using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Holds per-layer weight references for a single transformer layer.
/// Norm weights are dequantized to <c>float[]</c> at load time (small).
/// Linear projection weights remain as mmap pointers with their quantization type.
/// Bias arrays are nullable — null when the model has no biases (e.g. standard Llama/Mistral).
/// </summary>
internal readonly struct TransformerLayerWeights
{
    /// <summary>Pre-attention RMSNorm weight [hiddenSize].</summary>
    public readonly float[] AttnNormWeight;

    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to Q after projection, before RoPE. Null when absent (e.g. Qwen2, Llama).</summary>
    public readonly float[]? QNormWeight;
    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to K after projection, before RoPE. Null when absent.</summary>
    public readonly float[]? KNormWeight;

    /// <summary>Q projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint QWeight;
    public readonly QuantizationType QQuantType;
    public readonly int QOutputDim;
    public readonly int QInputDim;
    /// <summary>Optional Q projection bias [QOutputDim]. Null when absent.</summary>
    public readonly float[]? QBias;

    /// <summary>K projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint KWeight;
    public readonly QuantizationType KQuantType;
    public readonly int KOutputDim;
    public readonly int KInputDim;
    /// <summary>Optional K projection bias [KOutputDim]. Null when absent.</summary>
    public readonly float[]? KBias;

    /// <summary>V projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint VWeight;
    public readonly QuantizationType VQuantType;
    public readonly int VOutputDim;
    public readonly int VInputDim;
    /// <summary>Optional V projection bias [VOutputDim]. Null when absent.</summary>
    public readonly float[]? VBias;

    /// <summary>Output projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint OWeight;
    public readonly QuantizationType OQuantType;
    public readonly int OOutputDim;
    public readonly int OInputDim;
    /// <summary>Optional output projection bias [OOutputDim]. Null when absent.</summary>
    public readonly float[]? OBias;

    /// <summary>Pre-FFN RMSNorm weight [hiddenSize].</summary>
    public readonly float[] FfnNormWeight;

    /// <summary>SwiGLU gate projection.</summary>
    public readonly nint GateWeight;
    public readonly QuantizationType GateQuantType;
    public readonly int GateOutputDim;
    public readonly int GateInputDim;
    /// <summary>Optional gate projection bias [GateOutputDim]. Null when absent.</summary>
    public readonly float[]? GateBias;

    /// <summary>SwiGLU up projection.</summary>
    public readonly nint UpWeight;
    public readonly QuantizationType UpQuantType;
    public readonly int UpOutputDim;
    public readonly int UpInputDim;
    /// <summary>Optional up projection bias [UpOutputDim]. Null when absent.</summary>
    public readonly float[]? UpBias;

    /// <summary>Down projection.</summary>
    public readonly nint DownWeight;
    public readonly QuantizationType DownQuantType;
    public readonly int DownOutputDim;
    public readonly int DownInputDim;
    /// <summary>Optional down projection bias [DownOutputDim]. Null when absent.</summary>
    public readonly float[]? DownBias;

    public TransformerLayerWeights(
        float[] attnNormWeight,
        nint qWeight, QuantizationType qQuantType, int qOutputDim, int qInputDim,
        nint kWeight, QuantizationType kQuantType, int kOutputDim, int kInputDim,
        nint vWeight, QuantizationType vQuantType, int vOutputDim, int vInputDim,
        nint oWeight, QuantizationType oQuantType, int oOutputDim, int oInputDim,
        float[] ffnNormWeight,
        nint gateWeight, QuantizationType gateQuantType, int gateOutputDim, int gateInputDim,
        nint upWeight, QuantizationType upQuantType, int upOutputDim, int upInputDim,
        nint downWeight, QuantizationType downQuantType, int downOutputDim, int downInputDim,
        float[]? qBias = null, float[]? kBias = null, float[]? vBias = null, float[]? oBias = null,
        float[]? gateBias = null, float[]? upBias = null, float[]? downBias = null,
        float[]? qNormWeight = null, float[]? kNormWeight = null)
    {
        AttnNormWeight = attnNormWeight;
        QNormWeight = qNormWeight;
        KNormWeight = kNormWeight;
        QWeight = qWeight; QQuantType = qQuantType; QOutputDim = qOutputDim; QInputDim = qInputDim; QBias = qBias;
        KWeight = kWeight; KQuantType = kQuantType; KOutputDim = kOutputDim; KInputDim = kInputDim; KBias = kBias;
        VWeight = vWeight; VQuantType = vQuantType; VOutputDim = vOutputDim; VInputDim = vInputDim; VBias = vBias;
        OWeight = oWeight; OQuantType = oQuantType; OOutputDim = oOutputDim; OInputDim = oInputDim; OBias = oBias;
        FfnNormWeight = ffnNormWeight;
        GateWeight = gateWeight; GateQuantType = gateQuantType; GateOutputDim = gateOutputDim; GateInputDim = gateInputDim; GateBias = gateBias;
        UpWeight = upWeight; UpQuantType = upQuantType; UpOutputDim = upOutputDim; UpInputDim = upInputDim; UpBias = upBias;
        DownWeight = downWeight; DownQuantType = downQuantType; DownOutputDim = downOutputDim; DownInputDim = downInputDim; DownBias = downBias;
    }
}

/// <summary>
/// Holds R4-interleaved weight buffers for all projections in a single transformer layer.
/// Disposed when the parent <see cref="TransformerWeights"/> is disposed.
/// </summary>
internal sealed class RepackedLayerWeights : IDisposable
{
    public WeightRepacking.RepackedWeight Q, K, V, O, Gate, Up, Down;

    public void Dispose()
    {
        Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
        Gate.Dispose(); Up.Dispose(); Down.Dispose();
    }
}

/// <summary>
/// Organizes all weight tensor references from a loaded GGUF file for a transformer-family model.
/// Norm weights are dequantized to managed <c>float[]</c> at load time.
/// Linear projections remain as raw mmap pointers for zero-copy inference.
/// Optionally holds R4-interleaved weight buffers for improved cache locality in 4-row SIMD kernels.
/// </summary>
internal sealed class TransformerWeights : IDisposable
{
    /// <summary>Token embedding pointer and metadata.</summary>
    public nint TokenEmbedWeight { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public int VocabSize { get; }
    public int HiddenSize { get; }

    /// <summary>Per-layer weights.</summary>
    public TransformerLayerWeights[] Layers { get; }

    /// <summary>Final RMSNorm weight [hiddenSize].</summary>
    public float[] OutputNormWeight { get; }

    /// <summary>LM head (output projection) pointer and metadata.</summary>
    public nint OutputWeight { get; }
    public QuantizationType OutputQuantType { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    /// <summary>Per-layer R4-interleaved weights. Null until <see cref="RepackWeights"/> is called.</summary>
    public RepackedLayerWeights[]? RepackedLayers { get; private set; }

    /// <summary>R4-interleaved LM head weights. Null until <see cref="RepackWeights"/> is called or if type is not repackable.</summary>
    public WeightRepacking.RepackedWeight? RepackedOutput { get; private set; }

    private TransformerWeights(
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType, int vocabSize, int hiddenSize,
        TransformerLayerWeights[] layers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim)
    {
        TokenEmbedWeight = tokenEmbedWeight;
        TokenEmbedQuantType = tokenEmbedQuantType;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        Layers = layers;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputQuantType = outputQuantType;
        OutputOutputDim = outputOutputDim;
        OutputInputDim = outputInputDim;
    }

    /// <summary>
    /// Loads all weight references from a model container.
    /// Norm weights are dequantized to <c>float[]</c>. Linear projections stay as mmap pointers.
    /// </summary>
    public static TransformerWeights Load(IModelContainer container)
    {
        var config = container.Config;

        // Token embeddings
        if (!container.TryGetTensor("token_embd.weight", out var embDesc))
            throw new KeyNotFoundException("Token embeddings 'token_embd.weight' not found in model container.");

        nint embPtr = embDesc.Pointer;

        // Per-layer weights
        var layers = new TransformerLayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, container);
        }

        // Output norm
        if (!container.TryGetTensor("output_norm.weight", out var outNormDesc))
            throw new KeyNotFoundException("Output norm 'output_norm.weight' not found in model container.");

        float[] outputNormWeight = DequantizeNorm(outNormDesc, config.HiddenSize);

        // LM head — may be tied to token embeddings
        nint outputPtr;
        QuantizationType outputQt;
        int outputM, outputK;

        if (container.TryGetTensor("output.weight", out var outDesc))
        {
            outputPtr = outDesc.Pointer;
            outputQt = outDesc.QuantizationType;
            // Convention: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
            outputK = outDesc.Shape[0];
            outputM = outDesc.Shape[1];
        }
        else
        {
            // Tied embeddings: alias token_embd.weight
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        return new TransformerWeights(
            embPtr, embDesc.QuantizationType, config.VocabSize, config.HiddenSize,
            layers,
            outputNormWeight,
            outputPtr, outputQt, outputM, outputK);
    }

    /// <summary>
    /// Repacks all linear projection weights into R4 interleaved layout for improved
    /// cache locality in 4-row SIMD kernels. Skips token embeddings (random row access)
    /// and non-block-structured types (F32, F16).
    /// </summary>
    public void RepackWeights()
    {
        var repacked = new RepackedLayerWeights[Layers.Length];
        for (int i = 0; i < Layers.Length; i++)
        {
            ref readonly var lw = ref Layers[i];
            repacked[i] = new RepackedLayerWeights
            {
                Q = TryRepack(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim),
                K = TryRepack(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim),
                V = TryRepack(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim),
                O = TryRepack(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim),
                Gate = TryRepack(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim),
                Up = TryRepack(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim),
                Down = TryRepack(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim),
            };
        }
        RepackedLayers = repacked;

        if (WeightRepacking.IsRepackable(OutputQuantType))
            RepackedOutput = WeightRepacking.RepackR4(OutputWeight, OutputQuantType, OutputOutputDim, OutputInputDim);
    }

    private static WeightRepacking.RepackedWeight TryRepack(nint ptr, QuantizationType qt, int m, int k)
    {
        if (!WeightRepacking.IsRepackable(qt))
            return default;
        return WeightRepacking.RepackR4(ptr, qt, m, k);
    }

    /// <summary>Frees all R4-interleaved weight buffers.</summary>
    public void Dispose()
    {
        if (RepackedLayers is not null)
        {
            foreach (var rl in RepackedLayers)
                rl.Dispose();
            RepackedLayers = null;
        }
        RepackedOutput?.Dispose();
        RepackedOutput = null;
    }

    private static TransformerLayerWeights LoadLayer(
        int layerIdx,
        IModelContainer container)
    {
        ModelConfig config = container.Config;
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Attention norm — dequantize to float[]
        if (!container.TryGetTensor($"{prefix}.attn_norm.weight", out var attnNormDesc))
            throw new KeyNotFoundException($"Attention norm '{prefix}.attn_norm.weight' not found.");
        float[] attnNorm = DequantizeNorm(attnNormDesc, hiddenSize);

        // Q/K/V projections — check for fused attn_qkv.weight (Phi-3 style)
        nint qPtr, kPtr, vPtr;
        QuantizationType qQt, kQt, vQt;
        int qM, qK, kM, kK, vM, vK;

        if (container.TryGetTensor($"{prefix}.attn_qkv.weight", out var qkvDesc))
        {
            // Fused QKV — split by row offset
            nint qkvPtr = qkvDesc.Pointer;
            int inputDim = qkvDesc.Shape[0]; // hidden_size
            long rowBytes = Dequantize.RowByteSize(inputDim, qkvDesc.QuantizationType);

            int qDim = config.NumAttentionHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            qPtr = qkvPtr; qQt = qkvDesc.QuantizationType; qM = qDim; qK = inputDim;
            kPtr = qkvPtr + (nint)(qDim * rowBytes); kQt = qkvDesc.QuantizationType; kM = kvDim; kK = inputDim;
            vPtr = qkvPtr + (nint)((qDim + kvDim) * rowBytes); vQt = qkvDesc.QuantizationType; vM = kvDim; vK = inputDim;
        }
        else
        {
            // Separate Q/K/V (standard path)
            (qPtr, qQt, qM, qK) = LoadLinear(container, $"{prefix}.attn_q.weight");
            (kPtr, kQt, kM, kK) = LoadLinear(container, $"{prefix}.attn_k.weight");
            (vPtr, vQt, vM, vK) = LoadLinear(container, $"{prefix}.attn_v.weight");
        }

        var (oPtr, oQt, oM, oK) = LoadLinear(container, $"{prefix}.attn_output.weight");

        // Optional biases — check for fused attn_qkv.bias (Phi-3 style)
        float[]? qBias, kBias, vBias;
        if (container.TryGetTensor($"{prefix}.attn_qkv.bias", out var qkvBiasDesc))
        {
            // Fused QKV bias — split by element offset
            nint biasPtr = qkvBiasDesc.Pointer;
            int qDim = config.NumAttentionHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            qBias = new float[qDim];
            kBias = new float[kvDim];
            vBias = new float[kvDim];

            Dequantize.ToFloat32(biasPtr, qDim, qkvBiasDesc.QuantizationType, qBias);
            Dequantize.ToFloat32(biasPtr + qDim * sizeof(float), kvDim, qkvBiasDesc.QuantizationType, kBias);
            Dequantize.ToFloat32(biasPtr + (qDim + kvDim) * sizeof(float), kvDim, qkvBiasDesc.QuantizationType, vBias);
        }
        else
        {
            qBias = LoadOptionalBias(container, $"{prefix}.attn_q.bias");
            kBias = LoadOptionalBias(container, $"{prefix}.attn_k.bias");
            vBias = LoadOptionalBias(container, $"{prefix}.attn_v.bias");
        }
        float[]? oBias = LoadOptionalBias(container, $"{prefix}.attn_output.bias");

        // Optional QK-norms (Qwen3-style): per-head RMSNorm applied to Q/K after projection, before RoPE
        float[]? qNormWeight = LoadOptionalNorm(container, $"{prefix}.attn_q_norm.weight", config.HeadDim);
        float[]? kNormWeight = LoadOptionalNorm(container, $"{prefix}.attn_k_norm.weight", config.HeadDim);

        // FFN norm
        if (!container.TryGetTensor($"{prefix}.ffn_norm.weight", out var ffnNormDesc))
            throw new KeyNotFoundException($"FFN norm '{prefix}.ffn_norm.weight' not found.");
        float[] ffnNorm = DequantizeNorm(ffnNormDesc, hiddenSize);

        // FFN projections — check for fused gate+up (Phi-3 style: ffn_up.weight has 2x intermediate rows)
        nint gatePtr, upPtr, downPtr;
        QuantizationType gateQt, upQt, downQt;
        int gateM, gateK, upM, upK, downM, downK;
        float[]? gateBias, upBias, downBias;

        (downPtr, downQt, downM, downK) = LoadLinear(container, $"{prefix}.ffn_down.weight");
        downBias = LoadOptionalBias(container, $"{prefix}.ffn_down.bias");

        if (container.TryGetTensor($"{prefix}.ffn_gate.weight", out var gateDesc))
        {
            // Standard separate gate/up (Llama, Mistral, Qwen)
            (gatePtr, gateQt, gateM, gateK) = LoadLinear(container, gateDesc);
            (upPtr, upQt, upM, upK) = LoadLinear(container, $"{prefix}.ffn_up.weight");
            gateBias = LoadOptionalBias(container, $"{prefix}.ffn_gate.bias");
            upBias = LoadOptionalBias(container, $"{prefix}.ffn_up.bias");
        }
        else
        {
            // Fused gate+up in ffn_up.weight (Phi-3 style): output dim = 2 * intermediate_size
            // Split: first intermediate_size rows = gate, next intermediate_size rows = up
            if (!container.TryGetTensor($"{prefix}.ffn_up.weight", out var fusedDesc))
                throw new KeyNotFoundException($"FFN up '{prefix}.ffn_up.weight' not found.");

            nint fusedPtr = fusedDesc.Pointer;
            int inputDim = fusedDesc.Shape[0]; // hidden_size
            int fusedOutputDim = fusedDesc.Shape[1]; // 2 * intermediate_size
            int halfDim = fusedOutputDim / 2;
            long rowBytes = Dequantize.RowByteSize(inputDim, fusedDesc.QuantizationType);

            gatePtr = fusedPtr; gateQt = fusedDesc.QuantizationType; gateM = halfDim; gateK = inputDim;
            upPtr = fusedPtr + (nint)(halfDim * rowBytes); upQt = fusedDesc.QuantizationType; upM = halfDim; upK = inputDim;

            // Fused bias split (if present)
            if (container.TryGetTensor($"{prefix}.ffn_up.bias", out var fusedBiasDesc))
            {
                nint biasPtr = fusedBiasDesc.Pointer;
                gateBias = new float[halfDim];
                upBias = new float[halfDim];
                Dequantize.ToFloat32(biasPtr, halfDim, fusedBiasDesc.QuantizationType, gateBias);
                Dequantize.ToFloat32(biasPtr + halfDim * sizeof(float), halfDim, fusedBiasDesc.QuantizationType, upBias);
            }
            else
            {
                gateBias = null;
                upBias = null;
            }
        }

        return new TransformerLayerWeights(
            attnNorm,
            qPtr, qQt, qM, qK,
            kPtr, kQt, kM, kK,
            vPtr, vQt, vM, vK,
            oPtr, oQt, oM, oK,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias, kBias, vBias, oBias,
            gateBias, upBias, downBias,
            qNormWeight, kNormWeight);
    }

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) LoadLinear(
        IModelContainer container, string name)
    {
        if (!container.TryGetTensor(name, out var tensor))
            throw new KeyNotFoundException($"Linear weight tensor '{name}' not found.");
        return LoadLinear(container, tensor);
    }

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) LoadLinear(
        IModelContainer container, ModelTensor tensor)
    {
        // GGUF/dotLLM convention: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
        int k = tensor.Shape[0];
        int m = tensor.Shape[1];
        return (tensor.Pointer, tensor.QuantizationType, m, k);
    }

    private static float[] DequantizeNorm(ModelTensor tensor, int expectedSize)
    {
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(tensor.Pointer, expectedSize, tensor.QuantizationType, result);
        return result;
    }

    /// <summary>
    /// Loads an optional norm weight tensor. Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalNorm(IModelContainer container, string name, int expectedSize)
    {
        if (!container.TryGetTensor(name, out var tensor)) return null;
        return DequantizeNorm(tensor, expectedSize);
    }

    /// <summary>
    /// Loads an optional bias tensor. Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalBias(IModelContainer container, string name)
    {
        if (!container.TryGetTensor(name, out var tensor)) return null;
        int size = (int)tensor.Shape.ElementCount;
        float[] result = new float[size];
        Dequantize.ToFloat32(tensor.Pointer, size, tensor.QuantizationType, result);
        return result;
    }
}
