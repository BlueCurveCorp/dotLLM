using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models;
using DotLLM.RoCm.Interop;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
namespace DotLLM.RoCm;

/// <summary>
/// Hybrid CPU/GPU transformer model for AMD ROCm: first N layers run on GPU (FP16, HIP kernels),
/// remaining layers run on CPU (FP32, SIMD kernels). Hidden state is transferred D2H at
/// the layer boundary with FP16-to-FP32 conversion.
/// </summary>
public sealed unsafe class HipHybridTransformerModel : IModel
{
    private const int InterleavedMinRowBytes = 1024;

    // ── GPU resources ──
    private readonly HipWeights _gpuWeights;
    private readonly HipForwardState _gpuState;
    private readonly HipStream _stream;
    private readonly HipCublasHandle _cublas;
    private readonly HipContext _context;
    private readonly HipKernels _kernels;

    // ── CPU resources ──
    private readonly TransformerWeights _cpuWeights;
    private readonly TransformerForwardState _cpuState;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    // ── Shared ──
    private readonly IModelContainer _container;
    private readonly int _numGpuLayers;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _gpuRopeType;
    private readonly RoPEType _cpuRopeType;
    private readonly int? _slidingWindowSize;
    private nint _fp16TransferBuffer;
    private int _fp16TransferCapacity; // in elements

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _gpuState.AllocatedBytes + _cpuState.AllocatedBytes;

    /// <summary>Non-null when GPU-side weights exceed available VRAM.</summary>
    public string? VramWarning { get; }

    /// <summary>Number of transformer layers running on GPU.</summary>
    public int NumGpuLayers => _numGpuLayers;

    /// <summary>Debug: limit the number of transformer layers processed.</summary>
    internal int DebugMaxLayers { get; set; }

    private HipHybridTransformerModel(
        ModelConfig config, HipWeights gpuWeights, HipForwardState gpuState,
        HipStream stream, HipCublasHandle cublas, HipContext context,
        HipKernels kernels, TransformerWeights cpuWeights, TransformerForwardState cpuState,
        ComputeThreadPool? threadPool, bool ownsPool, IModelContainer container,
        int numGpuLayers, int deviceId, float ropeTheta, int ropeDim,
        int gpuRopeType, RoPEType cpuRopeType, int? slidingWindowSize,
        string? vramWarning)
    {
        Config = config;
        _gpuWeights = gpuWeights;
        _gpuState = gpuState;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _cpuWeights = cpuWeights;
        _cpuState = cpuState;
        _threadPool = threadPool;
        _ownsThreadPool = ownsPool;
        _container = container;
        _numGpuLayers = numGpuLayers;
        _deviceId = deviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _gpuRopeType = gpuRopeType;
        _cpuRopeType = cpuRopeType;
        _slidingWindowSize = slidingWindowSize;
        VramWarning = vramWarning;

        // Initial transfer buffer for decode (1 token)
        _fp16TransferCapacity = config.HiddenSize;
        _fp16TransferBuffer = (nint)NativeMemory.AlignedAlloc(
            (nuint)(_fp16TransferCapacity * sizeof(ushort)), 64);
    }

    public static HipHybridTransformerModel Load(IModelContainer container,
        int gpuLayers, int deviceId, ThreadingConfig threading)
    {
        var config = container.Config;
        if (gpuLayers <= 0 || gpuLayers >= config.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(gpuLayers));

        var cpuWeights = TransformerWeights.Load(container);
        cpuWeights.RepackWeights();

        var context = HipContext.Create(deviceId);
        var stream = HipStream.Create();
        var cublas = HipCublasHandle.Create();
        cublas.SetStream(stream.Handle);

        string? hsacoDir = Path.Combine(AppContext.BaseDirectory, "hsaco");
        var kernels = new HipKernels(hsacoDir);

        long estimatedGpuBytes = 0;
        foreach (var t in container.Tensors)
        {
            if (IsGpuTensor(t.Name, gpuLayers))
            {
                int innerDim = t.Shape[0];
                long outerDim = (long)t.Shape.ElementCount / innerDim;
                estimatedGpuBytes += Dequantize.RowByteSize(innerDim, t.QuantizationType) * outerDim;
            }
        }

        var gpuWeights = HipWeights.Load(cpuWeights, container, kernels, stream.Handle, gpuLayers);

        string? vramWarning = null;
        if (HipApi.hipMemGetInfo(out nuint freeAfter, out nuint totalVram) == 0 && totalVram > 0)
        {
            if (estimatedGpuBytes > (long)freeAfter)
            {
                vramWarning = $"Estimated VRAM requirements ({estimatedGpuBytes / 1024 / 1024} MB) " +
                              $"exceed free VRAM ({freeAfter / 1024 / 1024} MB).";
            }
        }

        var gpuState = new HipForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType cpuRopeType = config.RoPEConfig?.Type ?? RoPEType.Norm;
        int gpuRopeType = (int)cpuRopeType;

        var cpuState = new TransformerForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize,
            config.MaxSequenceLength, ropeDim, ropeTheta);

        ComputeThreadPool? pool = null;
        if (threading.IsParallel)
        {
            int effectiveThreads = threading.EffectiveThreadCount;
            pool = new ComputeThreadPool(effectiveThreads, topology: null, threading);
        }

        return new HipHybridTransformerModel(
            config, gpuWeights, gpuState, stream, cublas, context, kernels,
            cpuWeights, cpuState, pool, ownsPool: pool is not null, container,
            gpuLayers, deviceId, ropeTheta, ropeDim, gpuRopeType, cpuRopeType,
            config.SlidingWindowSize, vramWarning);
    }

    private static bool IsGpuTensor(string name, int numGpuLayers)
    {
        if (name.StartsWith("layers."))
        {
            int layerIdx = int.Parse(name.Split('.')[1]);
            return layerIdx < numGpuLayers;
        }
        return name.StartsWith("tok_embeddings") || name.StartsWith("norm");
    }

    public HipHybridKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new HipHybridKvCache(
            new HipKvCache(_numGpuLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen),
            new SimpleKvCache(Config.NumLayers - _numGpuLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen),
            _numGpuLayers);
    }

    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        _context.MakeCurrent();
        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        int h = sizeof(ushort); 

        int totalLayers = Config.NumLayers;
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => totalLayers,
            _ => Math.Min(DebugMaxLayers, totalLayers)
        };

        int gpuLayers = Math.Min(_numGpuLayers, numLayers);
        int cpuLayers = numLayers - gpuLayers;

        var hybridKvCache = kvCache as HipHybridKvCache;

        // PHASE 1: GPU
        nint s = _stream.Handle;
        _gpuState.EnsureCapacity(seqLen);

        fixed (int* tokenPtr = tokenIds)
            HipApi.hipMemcpyHtoD(_gpuState.TokenIdsDevice, (nint)tokenPtr, (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            HipApi.hipMemcpyHtoD(_gpuState.PositionsDevice, (nint)posPtr, (nuint)(seqLen * sizeof(int))).ThrowOnError();

        _kernels.LaunchEmbeddingLookup(
            _gpuWeights.TokenEmbedDevice, _gpuWeights.TokenEmbedQuantType,
            _gpuState.TokenIdsDevice, _gpuState.HiddenState,
            seqLen, hiddenSize, s);

        long hiddenBytes = (long)seqLen * hiddenSize * h;
        if (gpuLayers > 0)
        {
            HipApi.hipMemcpyDtoD(_gpuState.Residual, _gpuState.HiddenState, (nuint)hiddenBytes).ThrowOnError();
            _kernels.LaunchRmsNorm(_gpuState.HiddenState, _gpuWeights.Layers[0].AttnNormWeight,
                _gpuState.NormOutput, hiddenSize, eps, seqLen, s);
        }

        for (int layer = 0; layer < gpuLayers; layer++)
        {
            ref readonly var lw = ref _gpuWeights.Layers[layer];
            ProjectGpu(lw.QQuant, lw.QQuantType, lw.Q, _gpuState.NormOutput, _gpuState.Q, lw.QOutputDim, lw.QInputDim, seqLen);
            ProjectGpu(lw.KQuant, lw.KQuantType, lw.K, _gpuState.NormOutput, _gpuState.K, lw.KOutputDim, lw.KInputDim, seqLen);
            ProjectGpu(lw.VQuant, lw.VQuantType, lw.V, _gpuState.NormOutput, _gpuState.V, lw.VOutputDim, lw.VInputDim, seqLen);

            if (lw.QBias != 0) _kernels.LaunchBiasAdd(_gpuState.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0) _kernels.LaunchBiasAdd(_gpuState.K, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0) _kernels.LaunchBiasAdd(_gpuState.V, lw.VBias, lw.VOutputDim, seqLen, s);

            if (lw.QNormWeight != 0) _kernels.LaunchPerHeadRmsNorm(_gpuState.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0) _kernels.LaunchPerHeadRmsNorm(_gpuState.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            _kernels.LaunchRoPE(_gpuState.Q, _gpuState.K, _gpuState.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, _gpuRopeType, s);

            var gpuKvCache = hybridKvCache?.GpuCache;
            if (gpuKvCache != null)
            {
                gpuKvCache.UpdateDevice(_gpuState.K, _gpuState.V, positions, seqLen, layer, s);
                _kernels.LaunchAttention(_gpuState.Q, gpuKvCache.GetKeysPtr(layer),
                    gpuKvCache.GetValuesPtr(layer), _gpuState.AttnOutput,
                    seqLen, gpuKvCache.CurrentLength, numHeads, numKvHeads, headDim, positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_gpuState.Q, _gpuState.K, _gpuState.V, _gpuState.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim, 0, slidingWindow, s);
            }

            ProjectGpu(lw.OQuant, lw.OQuantType, lw.O, _gpuState.AttnOutput, _gpuState.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_gpuState.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

            _kernels.LaunchFusedAddRmsNorm(_gpuState.Residual, _gpuState.NormOutput,
                lw.FfnNormWeight, _gpuState.NormOutput, hiddenSize, eps, seqLen, s);

            ProjectGpu(lw.GateQuant, lw.GatewayQuantType, lw.Gate, _gpuState.NormOutput, _gpuState.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
            ProjectGpu(lw.UpQuant, lw.UpQuantType, lw.Up, _gpuState.NormOutput, _gpuState.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_gpuState.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_gpuState.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            _kernels.LaunchSwiGLU(_gpuState.FfnGate, _gpuState.FfnUp, _gpuState.SiluOutput, intermediateSize, seqLen, s);

            ProjectGpu(lw.DownQuant, lw.DownQuantType, lw.Down, _gpuState.SiluOutput, _gpuState.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_gpuState.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            if (layer < gpuLayers - 1)
            {
                ref readonly var nextLw = ref _gpuWeights.Layers[layer + 1];
                _kernels.LaunchFusedAddRmsNorm(_gpuState.Residual, _gpuState.NormOutput,
                    nextLw.AttnNormWeight, _gpuState.NormOutput, hiddenSize, eps, seqLen, s);
            }
            else
            {
                _kernels.LaunchAdd(_gpuState.Residual, _gpuState.NormOutput, _gpuState.HiddenState, seqLen * hiddenSize, s);
            }
        }

        // PHASE 2: Boundary Transfer
        if (gpuLayers > 0 && cpuLayers > 0)
        {
            HipApi.hipStreamSynchronize(s).ThrowOnError();
            int transferElements = seqLen * hiddenSize;
            EnsureTransferCapacity(transferElements);
            HipApi.hipMemcpyDtoH(_fp16TransferBuffer, _gpuState.HiddenState, (nuint)(transferElements * h)).ThrowOnError();
            _cpuState.EnsureCapacity(seqLen);
            ConvertFp16ToFp32(_fp16TransferBuffer, _cpuState.HiddenState, transferElements);
        }

        // PHASE 3: CPU
        if (cpuLayers > 0)
        {
            float* hidden = (float*)_cpuState.HiddenState;
            float* residual = (float*)_cpuState.Residual;
            float* normOut = (float*)_cpuState.NormOutput;
            float* q = (float*)_cpuState.Q;
            float* k = (float*)_cpuState.K;
            float* v = (float*)_cpuState.V;
            float* attnOut = (float*)_cpuState.AttnOutput;
            float* ffnGate = (float*)_cpuState.FfnGate;
            float* ffnUp = (float*)_cpuState.FfnUp;
            float* siluOut = (float*)_cpuState.SiluOutput;
            byte* inputQ8Scratch = (byte*)_cpuState.InputQ8Scratch;

            if (gpuLayers == 0)
            {
                EmbeddingLookupCpu(tokenIds, hidden, hiddenSize);
            }

            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

            for (int layer = gpuLayers; layer < numLayers; layer++)
            {
                ref readonly var lw = ref _cpuWeights.Layers[layer];
                var rl = _cpuWeights.RepackedLayers?[layer];

                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.AttnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                if (seqLen == 1 && _threadPool != null)
                {
                    byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.QQuantType);
                    FusedQkvDecode(in lw, normOut, preQuantNorm, q, k, v);
                }
                else
                {
                    byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.QQuantType);
                    var rwQ = rl?.Q ?? default;
                    var rwK = rl?.K ?? default;
                    var rwV = rl?.V ?? default;

                    GemmInterleaved(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, seqLen, preQuantNorm, in rwQ);
                    GemmInterleaved(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, seqLen, preQuantNorm, in rwK);
                    GemmInterleaved(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, seqLen, preQuantNorm, in rwV);
                }

                AddBias(lw.QBias, q, lw.QOutputDim, seqLen);
                AddBias(lw.KBias, k, lw.KOutputDim, seqLen);
                AddBias(lw.VBias, v, lw.VOutputDim, seqLen);

                if (lw.QNormWeight is not null) ApplyPerHeadNorm(lw.QNormWeight, q, numHeads, headDim, seqLen, eps);
                if (lw.KNormWeight is not null) ApplyPerHeadNorm(lw.KNormWeight, k, numKvHeads, headDim, seqLen, eps);

                RoPE.Execute(new Span<float>(q, seqLen * numHeads * headDim),
                    new Span<float>(k, seqLen * numKvHeads * headDim),
                    positions, numHeads, numKvHeads, headDim, _ropeDim,
                    _cpuState.CosTable, _cpuState.SinTable, _cpuRopeType);

                if (hybridKvCache is not null)
                {
                    var kRef = new TensorRef(seqLen, numKvHeads * headDim, DType.Float32, -1, (nint)k);
                    var vRef = new TensorRef(seqLen, numKvHeads * headDim, DType.Float32, -1, (nint)v);
                    hybridKvCache.Update(kRef, vRef, positions, layer);
                    int seqKv = hybridKvCache.CpuCache.CurrentLength;
                    var cachedK = hybridKvCache.CpuCache.GetKeysRef(layer - _numGpuLayers);
                    var cachedV = hybridKvCache.CpuCache.GetValuesRef(layer - _numGpuLayers);
                    Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                        seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool, _slidingWindowSize);
                }
                else
                {
                    Attention.Execute(q, k, v, attnOut, seqLen, seqLen, numHeads, numKvHeads, headDim, 0, _threadPool, _slidingWindowSize);
                }

                byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, numHeads * headDim, seqLen, lw.OQuantType);
                var rwO = rl?.O ?? default;
                GemmInterleaved(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, seqLen, preQuantAttn, in rwO);
                AddBias(lw.OBias, normOut, lw.OOutputDim, seqLen);

                for (int t = 0; t < seqLen; t++)
                {
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                }
                new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

                if (seqLen == 1 && _threadPool != null)
                {
                    byte* preQuantFfn = null;
                    if (IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType))
                        preQuantFfn = FusedOps.RmsNormQuantize(hidden, lw.FfnNormWeight, eps, inputQ8Scratch, hiddenSize, lw.GateQuantType);

                    if (preQuantFfn == null)
                    {
                        RmsNorm.Execute(new ReadOnlySpan<float>(hidden, hiddenSize), lw.FfnNormWeight, eps, new Span<float>(normOut, hiddenSize));
                        preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.GateQuantType);
                    }
                    FusedGateUpDecode(in lw, normOut, preQuantFfn, ffnGate, ffnUp);
                }
                else
                {
                    for (int t = 0; t < seqLen; t++)
                        RmsNorm.Execute(new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize), lw.FfnNormWeight, eps, new Span<float>(normOut + t * hiddenSize, hiddenSize));

                    byte* preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.GateQuantType);
                    var rwGate = rl?.Gate ?? default;
                    var rwUp = rl?.Up ?? default;
                    GemmInterleaved(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen, preQuantFfn, in rwGate);
                    GemmInterleaved(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen, 
                        IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType) ? preQuantFfn : null, in rwUp);
                }
                AddBias(lw.GateBias, ffnGate, lw.GateOutputDim, seqLen);
                AddBias(lw.UpBias, ffnUp, lw.UpOutputDim, seqLen);

                for (int t = 0; t < seqLen; t++)
                    FusedOps.SwiGLU(new ReadOnlySpan<float>(ffnGate + t * intermediateSize, intermediateSize),
                        new ReadOnlySpan<float>(ffnUp + t * intermediateSize, intermediateSize),
                        new Span<float>(siluOut + t * intermediateSize, intermediateSize));

                byte* preQuantSilu = QuantizeInput(siluOut, inputQ8Scratch, intermediateSize, seqLen, lw.DownQuantType);
                var rwDown = rl?.Down ?? default;
                GemmInterleaved(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen, preQuantSilu, in rwDown);
                AddBias(lw.DownBias, normOut, lw.DownOutputDim, seqLen);

                for (int t = 0; t < seqLen; t++)
                    Add.Execute(new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            }
        }

        // Final Norm + LM Head
        if (numLayers == totalLayers)
        {
            var finalNormWeight = _cpuWeights.OutputNormWeight;
            var lmWeights = _cpuWeights.OutputWeight;
            var lmQt = _cpuWeights.OutputQuantType;
            int vocabSize = Config.VocabSize;

            float* finalHidden = (float*)_cpuState.HiddenState;
            float* finalNormOut = (float*)_cpuState.NormOutput;
            float* logits = (float*)_cpuState.Logits;

            for (int t = 0; t < seqLen; t++)
                RmsNorm.Execute(new ReadOnlySpan<float>(finalHidden + t * hiddenSize, hiddenSize), finalNormWeight, eps, new Span<float>(finalNormOut + t * hiddenSize, hiddenSize));

            byte* preQuantFinal = QuantizeInput(finalNormOut, (byte*)_cpuState.InputQ8Scratch, hiddenSize, seqLen, lmQt);
            var rwLm = _cpuWeights.RepackedOutput ?? default;
            GemmInterleaved(lmWeights, lmQt, finalNormOut, logits, vocabSize, hiddenSize, seqLen, preQuantFinal, in rwLm);

            var shape = new TensorShape(seqLen, vocabSize);
            var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
            new Span<float>(logits, seqLen * vocabSize).CopyTo(
                new Span<float>((void*)result.DataPointer, seqLen * vocabSize));

            return result;
        }

        return UnmanagedTensor.Allocate(new TensorShape(seqLen, hiddenSize), DType.Float32, deviceId: -1); // fallback/hidden return
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ProjectGpu(nint quantWeight, QuantizationType qt, nint fp16Weight,
                            nint input, nint output, int outputDim, int inputDim, int seqLen)
    {
        nint s = _stream.Handle;
        if (seqLen > 1)
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _gpuState.DequantScratch, outputDim * inputDim, s);
                w = _gpuState.DequantScratch;
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
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _gpuState.DequantScratch, outputDim * inputDim, s);
                w = _gpuState.DequantScratch;
            }
            HipGemm.GemvF16(_cublas.Handle, w, input, output, outputDim, inputDim, s);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemv(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        if (qt == QuantizationType.Q8_0) MatMul.GemvQ8_0((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q4_K) MatMul.GemvQ4_K((byte*)weights, x, y, m, k, _threadPool);
        else MatMul.GemvF32((float*)weights, x, y, m, k, _threadPool);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weights, QuantizationType qt, float* b, float* c, int m, int k, int n, byte* preQuant = null)
    {
        if (qt == QuantizationType.Q8_0) MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, _threadPool, preQuant);
        else if (qt == QuantizationType.Q4_K) MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, _threadPool, preQuant);
        else MatMul.GemmF32((float*)weights, b, c, m, k, n, _threadPool);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemmInterleaved(nint origWeights, QuantizationType qt, float* b, float* c, int m, int k, int n, byte* preQuant, in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || n > 1 || rw.RowBytes < InterleavedMinRowBytes) { Gemm(origWeights, qt, b, c, m, k, n, preQuant); return; }
        if (preQuant != null) { DispatchInterleavedComputeRows(qt, (byte*)rw.Ptr, preQuant, c, rw.FullGroupCount, rw.TailRows, k); return; }
        GemvInterleaved(origWeights, qt, b, c, m, k, in rw);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemvInterleaved(nint origWeights, QuantizationType qt, float* x, float* y, int m, int k, in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || rw.RowBytes < InterleavedMinRowBytes) { Gemv(origWeights, qt, x, y, m, k); return; }
        byte* scratch = (byte*)_cpuState.InputQ8Scratch;
        if (qt == QuantizationType.Q8_0)
        {
            MatMul.QuantizeF32ToQ8_0(x, scratch, k);
            MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, scratch, y, rw.FullGroupCount, rw.TailRows, k / 32, _threadPool);
        }
        else Gemv(origWeights, qt, x, y, m, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DispatchInterleavedComputeRows(QuantizationType qt, byte* repWeights, byte* preInput, float* result, int full, int tail, int k)
    {
        if (qt == QuantizationType.Q8_0) MatMul.ComputeRowsQ8_0Interleaved(repWeights, preInput, result, full, tail, k / 32, _threadPool);
        else if (qt == QuantizationType.Q4_K) MatMul.ComputeRowsQ4_KInterleaved(repWeights, preInput, result, full, tail, k / 256, _threadPool);
    }

    private void FusedQkvDecode(in TransformerLayerWeights lw, float* normOut, byte* preQuant, float* q, float* k, float* v)
        => MatMul.FusedDecodeGemv3((byte*)lw.QWeight, lw.QQuantType, q, lw.QOutputDim, (byte*)lw.KWeight, lw.KQuantType, k, lw.KOutputDim, (byte*)lw.VWeight, lw.VQuantType, v, lw.VOutputDim, normOut, preQuant, lw.QInputDim, _threadPool!);

    private void FusedGateUpDecode(in TransformerLayerWeights lw, float* normOut, byte* preQuant, float* gate, float* up)
        => MatMul.FusedDecodeGemv2((byte*)lw.GateWeight, lw.GateQuantType, gate, lw.GateOutputDim, (byte*)lw.UpWeight, lw.UpQuantType, up, lw.UpOutputDim, normOut, preQuant, lw.GateInputDim, _threadPool!);

    private void EnsureTransferCapacity(int elements)
    {
        if (elements <= _fp16TransferCapacity) return;
        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)elements);
        if (_fp16TransferBuffer != 0) NativeMemory.AlignedFree((void*)_fp16TransferBuffer);
        _fp16TransferBuffer = (nint)NativeMemory.AlignedAlloc((nuint)(newCapacity * sizeof(ushort)), 64);
        _fp16TransferCapacity = newCapacity;
    }

    private static void ConvertFp16ToFp32(nint src, nint dst, int count)
    {
        var s = new ReadOnlySpan<Half>((Half*)src, count);
        var d = new Span<float>((float*)dst, count);
        TensorPrimitives.ConvertToSingle(s, d);
    }

    private void EmbeddingLookupCpu(ReadOnlySpan<int> ids, float* hidden, int size)
    {
        nint emb = _cpuWeights.TokenEmbedWeight;
        var qt = _cpuWeights.TokenEmbedQuantType;
        for (int t = 0; t < ids.Length; t++)
        {
            float* dest = hidden + t * size;
            if (qt == QuantizationType.F32) new ReadOnlySpan<float>((float*)emb + (long)ids[t] * size, size).CopyTo(new Span<float>(dest, size));
            else Dequantize.ToFloat32(emb + (nint)((long)ids[t] * Dequantize.RowByteSize(size, qt)), size, qt, new Span<float>(dest, size));
        }
    }

    private static void AddBias(float[]? bias, float* output, int dim, int seq)
    {
        if (bias is null) return;
        for (int t = 0; t < seq; t++) TensorPrimitives.Add(new ReadOnlySpan<float>(output + t * dim, dim), bias, new Span<float>(output + t * dim, dim));
    }

    private static byte* QuantizeInput(float* input, byte* scratch, int dim, int seq, QuantizationType qt)
    {
        if (qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K)
        {
            int rowBytes = (dim / 256) * MatMul.Q8_K_BlockBytes;
            for (int t = 0; t < seq; t++) MatMul.QuantizeF32ToQ8_K(input + t * dim, scratch + t * rowBytes, dim);
            return scratch;
        }
        if (qt == QuantizationType.Q8_0)
        {
            int rowBytes = (dim / 32) * 34;
            for (int t = 0; t < seq; t++) MatMul.QuantizeF32ToQ8_0(input + t * dim, scratch + t * rowBytes, dim);
            return scratch;
        }
        return null;
    }

    private static bool IsCompatiblePreQuant(QuantizationType s, QuantizationType t) => s == t || ((s is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K) && (t is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K));

    private static void ApplyPerHeadNorm(float[] weight, float* qk, int heads, int dim, int seq, float eps)
    {
        int stride = heads * dim;
        for (int t = 0; t < seq; t++)
            for (int h = 0; h < heads; h++)
                RmsNorm.Execute(new ReadOnlySpan<float>(qk + t * stride + h * dim, dim), weight, eps, new Span<float>(qk + t * stride + h * dim, dim));
    }

    public void Dispose()
    {
        _gpuState.Dispose(); _gpuWeights.Dispose(); _kernels.Dispose(); _cublas.Dispose(); _stream.Dispose(); _context.Dispose();
        _container.Dispose();
        if (_ownsThreadPool) _threadPool?.Dispose();
        _cpuState.Dispose(); _cpuWeights.Dispose();
        if (_fp16TransferBuffer != 0) { NativeMemory.AlignedFree((void*)_fp16TransferBuffer); _fp16TransferBuffer = 0; }
    }
}
