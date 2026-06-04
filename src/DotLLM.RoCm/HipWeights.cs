using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

internal readonly struct HipLayerWeights
{
    public readonly nint Q, K, V, O, Gate, Up, Down;
    public readonly nint QQuant, KQuant, VQuant, OQuant, GateQuant, UpQuant, DownQuant;
    public readonly QuantizationType QQuantType, KQuantType, VQuantType, OQuantType;
    public readonly QuantizationType GatewayQuantType, UpQuantType, DownQuantType;
    public readonly int QOutputDim, QInputDim, KOutputDim, KInputDim;
    public readonly int VOutputDim, VInputDim, OOutputDim, OInputDim;
    public readonly int GateOutputDim, GateInputDim, UpOutputDim, UpInputDim;
    public readonly int DownOutputDim, DownInputDim;
    public readonly nint AttnNormWeight, FfnNormWeight;
    public readonly nint QNormWeight, KNormWeight;
    public readonly nint QBias, KBias, VBias, OBias;
    public readonly nint GateBias, UpBias, DownBias;

    public HipLayerWeights(
        nint q, int qOut, int qIn, nint k, int kOut, int kIn,
        nint v, int vOut, int vIn, nint o, int oOut, int oIn,
        nint gate, int gateOut, int gateIn, nint up, int upOut, int upIn,
        nint down, int downOut, int downIn,
        nint attnNorm, nint ffnNorm, nint qNorm, nint kNorm,
        nint qBias, nint kBias, nint vBias, nint oBias,
        nint gateBias, nint upBias, nint downBias,
        nint qQuant, QuantizationType qQt, nint kQuant, QuantizationType kQt,
        nint vQuant, QuantizationType vQt, nint oQuant, QuantizationType oQt,
        nint gateQuant, QuantizationType gateQt, nint upQuant, QuantizationType upQt,
        nint downQuant, QuantizationType downQt)
    {
        Q = q; QOutputDim = qOut; QInputDim = qIn;
        K = k; KOutputDim = kOut; KInputDim = kIn;
        V = v; VOutputDim = vOut; VInputDim = vIn;
        O = o; OOutputDim = oOut; OInputDim = oIn;
        Gate = gate; GateOutputDim = gateOut; GateInputDim = gateIn;
        Up = up; UpOutputDim = upOut; UpInputDim = upIn;
        Down = down; DownOutputDim = downOut; DownInputDim = downIn;
        AttnNormWeight = attnNorm; FfnNormWeight = ffnNorm;
        QNormWeight = qNorm; KNormWeight = kNorm;
        QBias = qBias; KBias = kBias; VBias = vBias; OBias = oBias;
        GateBias = gateBias; UpBias = upBias; DownBias = downBias;
        QQuant = qQuant; QQuantType = qQt; KQuant = kQuant; KQuantType = kQt;
        VQuant = vQuant; VQuantType = vQt; OQuant = oQuant; OQuantType = oQt;
        GateQuant = gateQuant; GatewayQuantType = gateQt;
        UpQuant = upQuant; UpQuantType = upQt;
        DownQuant = downQuant; DownQuantType = downQt;
    }
}

internal sealed class HipWeights : IDisposable
{
    public HipLayerWeights[] Layers { get; }
    public nint TokenEmbedDevice { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public nint OutputNormWeight { get; }
    public nint OutputWeight { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }
    public nint OutputWeightQuant { get; }
    public QuantizationType OutputQuantType { get; }

    private readonly List<nint> _allAllocations = new();

    private HipWeights(HipLayerWeights[] layers, nint tokenEmbed, QuantizationType tokenEmbedQt,
                        nint outputNorm, nint outputWeight, int outputOutDim, int outputInDim,
                        nint outputWeightQuant, QuantizationType outputQt,
                        List<nint> allocs)
    {
        Layers = layers;
        TokenEmbedDevice = tokenEmbed;
        TokenEmbedQuantType = tokenEmbedQt;
        OutputNormWeight = outputNorm;
        OutputWeight = outputWeight;
        OutputOutputDim = outputOutDim;
        OutputInputDim = outputInDim;
        OutputWeightQuant = outputWeightQuant;
        OutputQuantType = outputQt;
        _allAllocations = allocs;
    }

    public static HipWeights LoadFromGguf(TransformerWeights cpuWeights, ModelConfig config,
                                            HipKernels kernels, nint stream, int numGpuLayers = -1)
    {
        int layerCount = numGpuLayers < 0 ? config.NumLayers : Math.Min(numGpuLayers, config.NumLayers);
        bool isHybrid = layerCount < config.NumLayers;
        var allocs = new List<nint>();

        nint tokenEmbed;
        var tokenEmbedQt = cpuWeights.TokenEmbedQuantType;

        if (tokenEmbedQt is QuantizationType.F32 or QuantizationType.F16 or QuantizationType.Q8_0)
        {
            long bytes = Dequantize.RowByteSize(config.HiddenSize, tokenEmbedQt) * config.VocabSize;
            tokenEmbed = AllocAndUpload(cpuWeights.TokenEmbedWeight, bytes, allocs);
        }
        else
        {
            tokenEmbed = UploadAndDequant(cpuWeights.TokenEmbedWeight, tokenEmbedQt, config.VocabSize, config.HiddenSize, allocs, kernels, stream);
            tokenEmbedQt = QuantizationType.F16;
        }

        nint outputNorm = 0, outputWeight = 0, outputWeightQuant = 0;

        if (!isHybrid)
        {
            outputNorm = UploadNormWeight(cpuWeights.OutputNormWeight, allocs, kernels, stream);

            var lmHasGemv = HipKernels.HasQuantizedGemv(cpuWeights.OutputQuantType);

            outputWeight = (!IsQuantized(cpuWeights.OutputQuantType) || !lmHasGemv)
                ? UploadAndDequant(cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
                    cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs, kernels, stream)
                : 0;
        }

        var layers = new HipLayerWeights[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            ref readonly TransformerLayerWeights lw = ref cpuWeights.Layers[i];

            layers[i] = new HipLayerWeights(
                SkipFp16(lw.QQuantType) ? 0 : UploadAndDequant(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs, kernels, stream), lw.QOutputDim, lw.QInputDim,
                SkipFp16(lw.KQuantType) ? 0 : UploadAndDequant(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs, kernels, stream), lw.KOutputDim, lw.KInputDim,
                SkipFp16(lw.VQuantType) ? 0 : UploadAndDequant(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs, kernels, stream), lw.VOutputDim, lw.VInputDim,
                SkipFp16(lw.OQuantType) ? 0 : UploadAndDequant(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs, kernels, stream), lw.OOutputDim, lw.OInputDim,
                SkipFp16(lw.GateQuantType) ? 0 : UploadAndDequant(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs, kernels, stream), lw.GateOutputDim, lw.GateInputDim,
                SkipFp16(lw.UpQuantType) ? 0 : UploadAndDequant(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs, kernels, stream), lw.UpOutputDim, lw.UpInputDim,
                SkipFp16(lw.DownQuantType) ? 0 : UploadAndDequant(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs, kernels, stream), lw.DownOutputDim, lw.DownInputDim,
                UploadNormWeight(lw.AttnNormWeight, allocs, kernels, stream), UploadNormWeight(lw.FfnNormWeight, allocs, kernels, stream),
                lw.QNormWeight is not null ? UploadNormWeight(lw.QNormWeight, allocs, kernels, stream) : 0,
                lw.KNormWeight is not null ? UploadNormWeight(lw.KNormWeight, allocs, kernels, stream) : 0,
                UploadBias(lw.QBias, allocs, kernels, stream), UploadBias(lw.KBias, allocs, kernels, stream),
                UploadBias(lw.VBias, allocs, kernels, stream), UploadBias(lw.OBias, allocs, kernels, stream),
                UploadBias(lw.GateBias, allocs, kernels, stream), UploadBias(lw.UpBias, allocs, kernels, stream),
                UploadBias(lw.DownBias, allocs, kernels, stream),
                UploadQuantized(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs), lw.QQuantType,
                UploadQuantized(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs), lw.KQuantType,
                UploadQuantized(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs), lw.VQuantType,
                UploadQuantized(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs), lw.OQuantType,
                UploadQuantized(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs), lw.GateQuantType,
                UploadQuantized(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs), lw.UpQuantType,
                UploadQuantized(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs), lw.DownQuantType);
        }

        HipApi.hipStreamSynchronize(stream).ThrowOnError();

        if (!isHybrid)
        {
            outputWeightQuant = UploadQuantized(cpuWeights.OutputWeight, cpuWeights.OutputQuantType, cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs);
        }

        return new HipWeights(layers, tokenEmbed, tokenEmbedQt, outputNorm, outputWeight, cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, outputWeightQuant, cpuWeights.OutputQuantType, allocs);
    }

    private static nint UploadQuantized(nint hostPtr, QuantizationType qt, int outDim, int inDim, List<nint> allocs)
    {
        if (qt is QuantizationType.F16 or QuantizationType.F32)
        {
            return 0;
        }

        long bytes = Dequantize.RowByteSize(inDim, qt) * outDim;
        return AllocAndUpload(hostPtr, bytes, allocs);
    }

    private static nint UploadAndDequant(nint hostPtr, QuantizationType qt, int outDim, int inDim, List<nint> allocs, HipKernels kernels, nint stream)
    {
        if (qt == QuantizationType.F16)
        {
            long bytes = (long)(outDim * inDim) * sizeof(ushort);
            return AllocAndUpload(hostPtr, bytes, allocs);
        }

        if (qt == QuantizationType.F32)
        {
            long f32Bytes = (long)(outDim * inDim) * sizeof(float);
            nint devF32 = AllocAndUpload(hostPtr, f32Bytes, allocs);
            long f16Bytes = (long)(outDim * inDim) * sizeof(ushort);
            HipApi.hipMalloc(out nint devF16, (nuint)f16Bytes).ThrowOnError();
            allocs.Add(devF16);
            kernels.LaunchConvertF32ToF16(devF32, devF16, outDim * inDim, stream);
            return devF16;
        }

        long quantBytes = Dequantize.RowByteSize(inDim, qt) * outDim;
        nint devQuant = AllocAndUpload(hostPtr, quantBytes, allocs);
        long fp16Bytes = (long)(outDim * inDim) * sizeof(ushort);
        HipApi.hipMalloc(out nint devFp16, (nuint)fp16Bytes).ThrowOnError();
        allocs.Add(devFp16);
        kernels.LaunchDequantToF16(devQuant, qt, devFp16, outDim * inDim, stream);
        return devFp16;
    }

    private static unsafe nint UploadNormWeight(float[] weight, List<nint> allocs, HipKernels kernels, nint stream)
    {
        int n = weight.Length;
        long f32Bytes = (long)n * sizeof(float);
        long f16Bytes = (long)n * sizeof(ushort);
        HipApi.hipMalloc(out nint devF32, (nuint)f32Bytes).ThrowOnError();
        allocs.Add(devF32);

        fixed (float* ptr = weight)
        {
            HipApi.hipMemcpyHtoD(devF32, (nint)ptr, (nuint)f32Bytes).ThrowOnError();
        }

        HipApi.hipMalloc(out nint devF16, (nuint)f16Bytes).ThrowOnError();
        allocs.Add(devF16);
        kernels.LaunchConvertF32ToF16(devF32, devF16, n, stream);
        return devF16;
    }

    private static nint UploadBias(float[]? bias, List<nint> allocs, HipKernels kernels, nint stream)
        => bias is null ? 0 : UploadNormWeight(bias, allocs, kernels, stream);

    private static bool IsQuantized(QuantizationType qt) => qt is not QuantizationType.F16 and not QuantizationType.F32;
    private static bool SkipFp16(QuantizationType qt) => HipKernels.HasQuantizedGemv(qt);

    private static nint AllocAndUpload(nint hostPtr, long bytes, List<nint> allocs)
    {
        HipApi.hipMalloc(out nint devPtr, (nuint)bytes).ThrowOnError();

        allocs.Add(devPtr);

        HipApi.hipMemcpyHtoD(devPtr, hostPtr, (nuint)bytes).ThrowOnError();
        return devPtr;
    }

    public void Dispose()
    {
        foreach (nint ptr in _allAllocations)
        {
            if (ptr != nint.Zero)
            {
                HipApi.hipFree(ptr);
            }
        }
        _allAllocations.Clear();
    }
}
