using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>Clover's bidirectional block preview on the Vulkan host (#1815); see <see cref="ICloverBlockPreview"/>.</summary>
/// <remarks>
/// <para>
/// The causal per-layer-submission forward with two substitutions: GDN layers record <see cref="RecordGdnPreviewLayer"/> (the causal
/// projections and decay, then <c>clover_bd_conv_silu_f32</c> and <c>clover_bd_softmax_f32</c> against the committed state, which no
/// kernel writes), and attention layers attend the whole prefix and block (<see cref="AttentionMaskMode.Bidirectional"/> on the per-token
/// kernel; the block's K/V are parked at cache rows <c>[start, start + T)</c> and rolled back). Embedding, MoE (plus the block expert
/// coreset mask between the router and the top-k, #1828) and the LM head are the causal path's, so the decode-fused path, MTP capture and verify snapshots are never involved.
/// </para>
/// <para>
/// Device memory: the four preview tensors per GDN layer are uploaded on the first preview (F32 as stored:
/// <c>((DConv-1) * convDim + 3 * NVHead) * 4</c> bytes per GDN layer, ~97 KiB at Clover-35B-A3B scale, ~2.9 MiB for 30 layers). The
/// logits read-back buffer grows to <c>T</c> rows (32 x 248320 x 4 B = 30.3 MiB, the existing multi-row logits buffer). No other storage.
/// </para>
/// </remarks>
public sealed partial class VulkanQwen3MoeHybridTransformerModel : ICloverBlockPreview
{
    private string? _cloverSpvDir;
    private CloverBlockPreviewKernels? _cloverKernels;
    private PreviewLayerBuffers?[]? _cloverBuffers;

    private sealed class PreviewLayerBuffers(VulkanDevice.Buffer right, VulkanDevice.Buffer scale, VulkanDevice.Buffer bias, VulkanDevice.Buffer c)
        : IDisposable
    {
        public readonly VulkanDevice.Buffer Right = right, Scale = scale, Bias = bias, C = c;

        public void Dispose()
        {
            Right.Dispose();
            Scale.Dispose();
            Bias.Dispose();
            C.Dispose();
        }
    }

    /// <inheritdoc/>
    public bool SupportsBlockPreview
    {
        get
        {
            if (Config.Clover is null) return false;
            foreach (var lw in _cpuLayers)
                if (lw.Gdn is { Preview: null }) return false;
            return true;
        }
    }

    private void EnsureCloverPreviewResources()
    {
        _cloverKernels ??= CloverBlockPreviewKernels.Create(_device,
            _cloverSpvDir ?? throw new InvalidOperationException("The SPIR-V directory of this model is unknown."));
        if (_cloverBuffers is not null) return;
        var buffers = new PreviewLayerBuffers?[_cpuLayers.Length];
        try
        {
            for (int l = 0; l < _cpuLayers.Length; l++)
            {
                if (_cpuLayers[l].Gdn?.Preview is not { } p) continue;
                buffers[l] = new PreviewLayerBuffers(Upload(p.ConvRightWeight), Upload(p.LogitScale), Upload(p.HistoryBias), Upload(p.HistoryC));
            }
        }
        catch
        {
            foreach (var b in buffers) b?.Dispose();
            throw;
        }
        _cloverBuffers = buffers;

        VulkanDevice.Buffer Upload(float[] data)
        {
            var buf = _device.AllocateDeviceLocal((long)data.Length * sizeof(float));
            _device.Upload(data, buf);
            return buf;
        }
    }

    private void DisposeCloverPreview()
    {
        if (_cloverBuffers is not null)
            foreach (var b in _cloverBuffers) b?.Dispose();
        _cloverBuffers = null;
        _cloverKernels?.Dispose();
        _cloverKernels = null;
    }

    /// <inheritdoc/>
    public ITensor ForwardBlockPreview(ReadOnlySpan<int> blockTokenIds, int startPosition, IKvCache? kvCache, IGdnState? gdnState, int deviceId)
    {
        var clover = Config.Clover ?? throw new NotSupportedException("The block preview is a Clover (clover_moe) forward; this model is not Clover.");
        if (!SupportsBlockPreview)
            throw new NotSupportedException("This Clover file carries no bidirectional-preview tensors (llama.cpp's converter drops them); "
                                            + "use the official DiffuSpace GGUF.");
        int T = blockTokenIds.Length;
        if (T == 0 || T > Math.Min(clover.BlockSize, CloverBlockPreviewKernels.MaxBlock))
            throw new ArgumentException($"Preview block length {T} must be 1..{clover.BlockSize}.", nameof(blockTokenIds));
        if (_gdn.DState > CloverBlockPreviewKernels.MaxHeadDim || _gdn.DConv > CloverBlockPreviewKernels.MaxConvWidth)
            throw new NotSupportedException($"GDN head dim {_gdn.DState} / conv width {_gdn.DConv} exceed the preview kernels' bounds.");
        ArgumentOutOfRangeException.ThrowIfNegative(startPosition);
        if (kvCache is null && startPosition != 0)
            throw new ArgumentException("A preview after a non-empty prefix needs the prefix's KV cache.", nameof(kvCache));
        if (kvCache is not null && kvCache is not VulkanNemotronHKvCache)
            throw new ArgumentException($"VulkanQwen3MoeHybridTransformerModel requires a VulkanNemotronHKvCache; got {kvCache.GetType().Name}.", nameof(kvCache));
        if (kvCache is not null && kvCache.CurrentLength != startPosition)
            throw new ArgumentException($"startPosition {startPosition} must equal the committed KV length {kvCache.CurrentLength}.", nameof(startPosition));
        if (startPosition + T > Config.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(startPosition), "The block runs past the model's maximum sequence length.");
        if (kvCache is not null && !kvCache.TryGrowTo(startPosition + T))
            throw new InvalidOperationException($"The KV cache cannot hold {startPosition + T} positions.");

        VulkanGdnStateCache gdnCache = gdnState switch
        {
            null => _gdnCache,
            VulkanGdnStateCache vk when vk.NumGdnLayers == _gdnCache.NumGdnLayers => vk,
            VulkanGdnStateCache vk => throw new ArgumentException(
                $"GdnState NumGdnLayers ({vk.NumGdnLayers}) does not match model GDN-layer count ({_gdnCache.NumGdnLayers}).", nameof(gdnState)),
            _ => throw new ArgumentException($"VulkanQwen3MoeHybridTransformerModel requires a VulkanGdnStateCache; got {gdnState.GetType().Name}.",
                                             nameof(gdnState)),
        };
        if (kvCache is null && gdnState is null && startPosition == 0)
            ResetSequenceState();   // nothing is committed at position 0 (#1786 new-sequence rule, as the causal forward)

        EnsureCloverPreviewResources();
        int[] positions = new int[T];
        for (int i = 0; i < T; i++) positions[i] = startPosition + i;

        int hiddenSize = Config.HiddenSize, vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        bool resized = _state.EnsureCapacity(T);
        if (resized) { _kernels.InvalidateAll(); _iqF16Prefill?.InvalidateDescriptorCache(); }
        var logitsBuf = T == 1 ? _state.Logits : EnsureMultiRowLogits(T, vocabSize);
        // Handles may have been recycled by any re-allocation since the last preview (causal forwards included): rebuild the sets.
        _cloverKernels!.InvalidateDescriptorCaches();
        UploadPositions(positions);

        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        RecordEmbeddingGather(cmdBuf, blockTokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        _submit.SubmitAndWait();

        var kinds = _layout.LayerKind;
        float coresetTopP = clover.PreviewCoresetTopP;   // the block expert coreset (#1828), in every MoE layer of the preview
        try
        {
            for (int layer = 0; layer < _cpuLayers.Length; layer++)
            {
                var lw = _cpuLayers[layer];
                ref readonly var layerBuf = ref _weights.Layers[layer];
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
                RecordMoeHybridTokenMixing(cmdBuf, layer, layerBuf, kinds, T, hiddenSize, eps, positions, gdnCache, kvCache,
                    Config.NumAttentionHeads, Config.NumKvHeads, Config.HeadDim, cloverPreview: true);
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();

                if (_cpuMoeLayer[layer]) RunCpuPlacedMoeLayer(layer, layerBuf, T, hiddenSize, eps, coresetTopP);
                else RunGpuPlacedMoeLayer(layer, lw, layerBuf, T, hiddenSize, eps, coresetTopP);
            }
        }
        finally
        {
            // The block's K/V were parked at [start, start + T): drop them so the committed length is unchanged.
            if (kvCache is not null && kvCache.CurrentLength != startPosition)
                kvCache.Rollback(startPosition);
        }

        _submit.Begin();
        cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput, 0, 0, (ulong)((long)T * hiddenSize * sizeof(float)));
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        _kernels.RmsNorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput, rowCount: T, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType, _state.NormOutput, logitsBuf,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: T);
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        var result = UnmanagedTensor.Allocate(new TensorShape(T, vocabSize), DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, T * vocabSize);
            _device.Download(logitsBuf, dest);
            if ((uint)clover.MaskTokenId < (uint)vocabSize)
                for (int r = 0; r < T; r++) dest[r * vocabSize + clover.MaskTokenId] = float.NegativeInfinity;
        }
        return result;
    }

    /// <summary>
    /// One GDN layer of the preview: the causal projections and decay, the preview conv (clean conv state read, not advanced), the
    /// causal split + L2 norm, the block softmax against the clean matrix state (only read), then the causal gate and <c>ssm_out</c>.
    /// </summary>
    private void RecordGdnPreviewLayer(nint cmdBuf, int absoluteLayerIdx, VulkanQwen3MoeHybridWeights.GdnLayerBuffers gdnW,
                                       int seqLen, float eps, VulkanGdnStateCache gdnCache)
    {
        var pv = _cloverBuffers![absoluteLayerIdx] ?? throw new InvalidOperationException($"Layer {absoluteLayerIdx} has no preview tensors.");
        var kernels = _cloverKernels!;
        int nV = _gdn.NVHead, nK = _gdn.NKHead, dS = _gdn.DState, dC = _gdn.DConv;
        int convDim = (2 * nK + nV) * dS, vDim = nV * dS, kDim = nK * dS;
        int ordinal = _gdnLayerOrdinal[absoluteLayerIdx];
        var convStateBuf = gdnCache.GetConvStateBuffer(ordinal);
        var gdnStateBuf = gdnCache.GetGdnStateBuffer(ordinal);

        // 1. projections (beta is not used by the preview)
        RecordMatmul(cmdBuf, gdnW.QkvWeight, gdnW.QkvDeviceQuantType, _state.NormOutput, _state.GdnQkvBuf,
            outputDim: gdnW.QkvOutputDim, inputDim: gdnW.QkvInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.GateWeight, gdnW.GateDeviceQuantType, _state.NormOutput, _state.GdnZBuf,
            outputDim: gdnW.GateOutputDim, inputDim: gdnW.GateInputDim, seqLen: seqLen);
        RecordMatmul(cmdBuf, gdnW.AlphaWeight, gdnW.AlphaDeviceQuantType, _state.NormOutput, _state.GdnAlphaBuf,
            outputDim: gdnW.AlphaOutputDim, inputDim: gdnW.AlphaInputDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 2. decay exp(g) in place
        _kernels.GdnDecay.Record(cmdBuf, _state.GdnAlphaBuf, gdnW.DtBiasDevice, gdnW.ADevice, seqLen: seqLen, nVHead: nV);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 3. preview conv + right conv + SiLU into GdnConvInput; the conv state is only read
        kernels.RecordConvSilu(cmdBuf, convStateBuf, _state.GdnQkvBuf, gdnW.Conv1dWeight, gdnW.Conv1dBias, pv.Right, _state.GdnConvInput,
            dConv: dC, channels: convDim, seqLen: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 4. split Q/K/V, L2-normalise Q and K (the causal kernels)
        var convOut = _state.GdnConvInput;
        if (_kernels.GdnQkvSplit is { } qkvSplit)
        {
            qkvSplit.RecordGdnQkvSplit(cmdBuf, convOut, _state.GdnQBuf, _state.GdnKBuf, _state.GdnVBuf, seqLen, kDim, vDim);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }
        else
        {
            KernelSupport.ComputeToTransferBarrier(cmdBuf);
            long convDimBytes = (long)convDim * sizeof(float), kDimBytes = (long)kDim * sizeof(float), vDimBytes = (long)vDim * sizeof(float);
            for (int t = 0; t < seqLen; t++)
            {
                ulong rowBase = (ulong)((long)t * convDimBytes);
                RecordCopyBufferRange(cmdBuf, convOut, _state.GdnQBuf, rowBase, (ulong)((long)t * kDimBytes), (ulong)kDimBytes);
                RecordCopyBufferRange(cmdBuf, convOut, _state.GdnKBuf, rowBase + (ulong)kDimBytes, (ulong)((long)t * kDimBytes), (ulong)kDimBytes);
                RecordCopyBufferRange(cmdBuf, convOut, _state.GdnVBuf, rowBase + (ulong)(2 * kDimBytes), (ulong)((long)t * vDimBytes), (ulong)vDimBytes);
            }
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }
        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnQBuf, totalHeads: seqLen * nK, dState: dS, eps: 1e-6f);
        _kernels.GdnL2Normalize.Record(cmdBuf, _state.GdnKBuf, totalHeads: seqLen * nK, dState: dS, eps: 1e-6f);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 5. bidirectional block softmax + read-only history bucket
        kernels.RecordSoftmax(cmdBuf, gdnStateBuf, _state.GdnQBuf, _state.GdnKBuf, _state.GdnVBuf, _state.GdnAlphaBuf,
            pv.Scale, pv.Bias, pv.C, _state.GdnOut, nV, nK, dS, seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // 6. per-head RMSNorm x silu(z), 7. ssm_out (the causal kernels)
        _kernels.GdnPostScanGate.Record(cmdBuf, gdnOut: _state.GdnOut, z: _state.GdnZBuf, ssmNormWeight: gdnW.SsmNormWeight,
            seqLen: seqLen, nVHead: nV, dState: dS, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordMatmul(cmdBuf, gdnW.OutWeight, gdnW.OutDeviceQuantType, _state.GdnOut, _state.NormOutput,
            outputDim: gdnW.OutOutputDim, inputDim: gdnW.OutInputDim, seqLen: seqLen);
    }
}
