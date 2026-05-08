using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.RoCm.Interop;
using DotLLM.Models;
using DotLLM.Models.Architectures;
namespace DotLLM.RoCm;

/// <summary>
/// GPU-accelerated transformer forward pass using AMD ROCm via HIP.
/// structurally with hip* API calls.
/// </summary>
public sealed class HipTransformerModel : IModel
{
    private readonly HipWeights _weights;
    private readonly HipForwardState _state;
    private readonly HipStream _stream;
    private readonly HipCublasHandle _cublas;
    private readonly HipContext _context;
    private readonly HipKernels _kernels;
    private readonly IModelContainer _container;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;

    public ModelConfig Config { get; }
    public long ComputeMemoryBytes => _state.AllocatedBytes;
    public string? VramWarning { get; }
    internal int DebugMaxLayers { get; set; }
    internal int DebugRopeTypeOverride { get; set; } = -1;
    internal bool DebugSkipBias { get; set; }

    private HipTransformerModel(ModelConfig config, HipWeights weights, HipForwardState state,
        HipStream stream, HipCublasHandle cublas, HipContext context, HipKernels kernels,
        IModelContainer container, int deviceId, float ropeTheta, int ropeDim, int ropeType, string? vramWarning)
    {
        Config = config; _weights = weights; _state = state; _stream = stream;
        _cublas = cublas; _context = context; _kernels = kernels; _container = container;
        _deviceId = deviceId; _ropeTheta = ropeTheta; _ropeDim = ropeDim;
        VramWarning = vramWarning; _ropeType = ropeType;
    }

 
    public static HipTransformerModel Load(IModelContainer container, int deviceId = 0, string? hsacoDir = null)
    {
        var config = container.Config;
        var cpuWeights = TransformerWeights.Load(container);
        var context = HipContext.Create(deviceId);
        var stream = HipStream.Create();
        var cublas = HipCublasHandle.Create();
        cublas.SetStream(stream.Handle);

        hsacoDir ??= Path.Combine(AppContext.BaseDirectory, "hsaco");
        var kernels = new HipKernels(hsacoDir);

        // Check VRAM estimation
        long estimatedWeightBytes = 0;
        foreach (var t in container.Tensors)
        {
            int innerDim = t.Shape[0];
            long outerDim = (long)t.Shape.ElementCount / innerDim;
            estimatedWeightBytes += Cpu.Kernels.Dequantize.RowByteSize(innerDim, t.QuantizationType) * outerDim;
        }

        string? vramWarning = null;

        if (HipApi.hipMemGetInfo(out nuint freeBefore, out nuint totalVram) == 0 && totalVram > 0 && estimatedWeightBytes > (long)freeBefore)
        {
            long modelMb = estimatedWeightBytes / (1024 * 1024);
            long freeMb = (long)freeBefore / (1024 * 1024);
            vramWarning = $"Model weights (~{modelMb} MB) exceed available VRAM ({freeMb} MB free). " +
                          "Performance will be degraded. Consider a smaller model or quantization format.";
        }

        var weights = HipWeights.Load(cpuWeights, container, kernels, stream.Handle);
        var state = new HipForwardState(config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        int ropeDimVal = config.RoPEConfig?.DimensionCount ?? config.HeadDim;

        if (ropeDimVal == 0)
        {
            ropeDimVal = config.HeadDim;
        }

        float ropeThetaVal = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeTypeVal = (int)(config.RoPEConfig?.Type ?? RoPEType.Norm);

        return new HipTransformerModel(config, weights, state, stream, cublas, context, kernels, container, deviceId, ropeThetaVal, ropeDimVal, ropeTypeVal, vramWarning);
    }


    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        _context.MakeCurrent();
        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize, numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads, headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize, vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        int h = sizeof(ushort);
        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;

        _state.EnsureCapacity(seqLen);

        fixed (int* tokenPtr = tokenIds)
        {
            HipApi.hipMemcpyHtoD(_state.TokenIdsDevice, (nint)tokenPtr, (nuint)(seqLen * sizeof(int))).ThrowOnError();
        }

        fixed (int* posPtr = positions)
        {
            HipApi.hipMemcpyHtoD(_state.PositionsDevice, (nint)posPtr, (nuint)(seqLen * sizeof(int))).ThrowOnError();
        }

        _kernels.LaunchEmbeddingLookup(_weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenState, seqLen, hiddenSize, s);

        long hiddenBytes = (long)seqLen * hiddenSize * h;

        HipApi.hipMemcpyDtoDAsync(_state.Residual, _state.HiddenState, (nuint)hiddenBytes, s).ThrowOnError();

        _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
            hiddenSize, eps, seqLen, s);

        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        if (numLayers == 0)
        {
            HipApi.hipMemcpyDtoDAsync(_state.HiddenState, _state.Residual, (nuint)hiddenBytes, s).ThrowOnError();
        }

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly HipLayerWeights lw = ref _weights.Layers[layer];

            Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
            Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
            Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);

            if (lw.QBias != nint.Zero && !DebugSkipBias)
            {
                _kernels.LaunchBiasAdd(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            }

            if (lw.KBias != nint.Zero && !DebugSkipBias)
            {
                _kernels.LaunchBiasAdd(_state.K, lw.KBias, lw.KOutputDim, seqLen, s);
            }

            if (lw.VBias != nint.Zero && !DebugSkipBias)
            {
                _kernels.LaunchBiasAdd(_state.V, lw.VBias, lw.VOutputDim, seqLen, s);
            }

            if (lw.QNormWeight != nint.Zero)
            {
                _kernels.LaunchPerHeadRmsNorm(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            }

            if (lw.KNormWeight != nint.Zero)
            {
                _kernels.LaunchPerHeadRmsNorm(_state.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);
            }

            int effectiveRope = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;

            _kernels.LaunchRoPE(_state.Q, _state.K, _state.PositionsDevice, seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, effectiveRope, s);

            if (kvCache is HipQuantizedKvCache hipQKvCache)
            {
                hipQKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s, _kernels);

                int seqKv = hipQKvCache.CurrentLength;

                var (kPtr, vPtr) = hipQKvCache.PrepareAttentionScratch(layer, s, _kernels);

                _kernels.LaunchAttention(_state.Q, kPtr, vPtr, _state.AttnOutput, seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], slidingWindow, s);
            }
            else if (kvCache is HipKvCache hipKvCache)
            {
                hipKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s);
                int seqKv = hipKvCache.CurrentLength;
                _kernels.LaunchAttention(_state.Q, hipKvCache.GetKeysPtr(layer),
                    hipKvCache.GetValuesPtr(layer), _state.AttnOutput, seqLen, seqKv,
                    numHeads, numKvHeads, headDim, positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_state.Q, _state.K, _state.V, _state.AttnOutput, seqLen, seqLen,
                    numHeads, numKvHeads, headDim, 0, slidingWindow, s);
            }

            Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);

            if (lw.OBias != nint.Zero)
            {
                _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);
            }

            _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);

            Project(lw.GateQuant, lw.GatewayQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
            Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (lw.GateBias != nint.Zero)
            {
                _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            }

            if (lw.UpBias != nint.Zero)
            {
                _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);
            }

            _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput, intermediateSize, seqLen, s);

            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);

            if (lw.DownBias != nint.Zero)
            {
                _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);
            }

            if (layer < numLayers - 1)
            {
                ref readonly HipLayerWeights nextLw = ref _weights.Layers[layer + 1];

                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput, hiddenSize, eps, seqLen, s);
            }
            else
            {
                _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.HiddenState, seqLen * hiddenSize, s);
            }
        }

        nint lastHidden = _state.HiddenState + (nint)((seqLen - 1) * hiddenSize * h);

        _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput, hiddenSize, eps, 1, s);

        Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
            _state.NormOutput, _state.LogitsF16, _weights.OutputOutputDim, _weights.OutputInputDim, 1);

        _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);

        _stream.Synchronize();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        HipApi.hipMemcpyDtoH(result.DataPointer, _state.LogitsF32, (nuint)(vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    private void Project(nint quantWeight, QuantizationType qt, nint fp16Weight, nint input,
        nint output, int outputDim, int inputDim, int seqLen)
    {
        nint s = _stream.Handle;

        if (seqLen > 1)
        {
            nint w = fp16Weight;
            if (w == nint.Zero)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch, outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            HipGemm.LinearF16(_cublas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
        }
        else if (quantWeight != 0 && HipKernels.HasQuantizedGemv(qt))
        {
            _kernels.LaunchQuantizedGemv(quantWeight, qt, input, output, outputDim, inputDim, s);
        }
        else
        {
            nint w = fp16Weight;
            if (w == nint.Zero)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch, outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            HipGemm.GemvF16(_cublas.Handle, w, input, output, outputDim, inputDim, s);
        }
    }

    public HipKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new HipKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
    }

    public IKvCache CreateKvCache(int maxSeqLen, KvCacheConfig config)
    {
        _context.MakeCurrent();

        if (!config.IsQuantized)
        {
            return new HipKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
        }

        return new HipQuantizedKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen, config);
    }

    public void Dispose()
    {
        _state.Dispose();
        _weights.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
        _container.Dispose();
    }
}
