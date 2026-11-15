using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// The two GDN kernels of Clover's bidirectional block preview (#1815): <c>clover_bd_conv_silu_f32</c> (causal conv over the clean conv
/// state plus the in-block right conv, SiLU; the state is only read) and <c>clover_bd_softmax_f32</c> (per-head softmax over the block
/// plus the read-only history bucket). Everything else in the preview forward reuses the causal path's kernels.
/// </summary>
/// <remarks>Both read and write only existing F32 GDN scratch; the softmax row lives in shared memory (<c>T + 1</c> floats).</remarks>
public sealed class CloverBlockPreviewKernels : IDisposable
{
    /// <summary>Largest conv width the conv kernel supports.</summary>
    public const int MaxConvWidth = 8;

    /// <summary>Largest GDN head dimension the softmax kernel supports (shared <c>MAX_D</c>).</summary>
    public const int MaxHeadDim = 256;

    /// <summary>Largest block the softmax kernel supports (shared <c>MAX_T</c>).</summary>
    public const int MaxBlock = 64;

    private const int ConvPushBytes = 3 * sizeof(uint);
    private const int SoftmaxPushBytes = 4 * sizeof(uint) + 2 * sizeof(float);
    private const int ConvBuffers = 6;
    private const int SoftmaxBuffers = 9;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _convModule, _softmaxModule;
    private readonly ComputePipeline _conv, _softmax;
    private readonly nint _convPool, _softmaxPool;
    private readonly DescriptorSetCache _convCache, _softmaxCache;
    private bool _disposed;

    private CloverBlockPreviewKernels(VulkanDevice device, string spvDir)
    {
        _device = device;
        (_convModule, _conv) = Load(device, spvDir, "clover_bd_conv_silu_f32", ConvBuffers, ConvPushBytes);
        (_softmaxModule, _softmax) = Load(device, spvDir, "clover_bd_softmax_f32", SoftmaxBuffers, SoftmaxPushBytes);
        _convPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: ConvBuffers);
        _softmaxPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: SoftmaxBuffers);
        _convCache = new DescriptorSetCache(device, _convPool, _conv, buffersPerSet: ConvBuffers);
        _softmaxCache = new DescriptorSetCache(device, _softmaxPool, _softmax, buffersPerSet: SoftmaxBuffers);
    }

    private static (VulkanModule, ComputePipeline) Load(VulkanDevice device, string spvDir, string name, int buffers, int pushBytes)
    {
        string path = Path.Combine(spvDir, name + ".spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffers];
            for (int i = 0; i < buffers; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            return (module, module.CreateComputePipeline("main", bindings, (uint)pushBytes));
        }
        catch
        {
            module.Dispose();
            throw;
        }
    }

    /// <summary>Loads both pipelines from <paramref name="spvDir"/>.</summary>
    public static CloverBlockPreviewKernels Create(VulkanDevice device, string spvDir) => new(device, spvDir);

    /// <summary>Drops cached descriptor sets (call whenever any bound buffer may have been re-allocated).</summary>
    public void InvalidateDescriptorCaches()
    {
        _convCache.Reset();
        _softmaxCache.Reset();
    }

    /// <summary>Records <c>output = silu(causal_conv([state | qkv]) + right_conv(qkv))</c> over <paramref name="seqLen"/> block rows.</summary>
    public unsafe void RecordConvSilu(nint cmdBuf, VulkanDevice.Buffer state, VulkanDevice.Buffer qkv, VulkanDevice.Buffer weight,
        VulkanDevice.Buffer bias, VulkanDevice.Buffer right, VulkanDevice.Buffer output, int dConv, int channels, int seqLen)
    {
        if (dConv < 2 || dConv > MaxConvWidth) throw new ArgumentOutOfRangeException(nameof(dConv));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (ReferenceEquals(qkv, output)) throw new ArgumentException("output must not alias qkv.", nameof(output));
        if (right.Size < (long)(dConv - 1) * channels * sizeof(float)) throw new ArgumentException("right-conv buffer too small.", nameof(right));
        if (output.Size < (long)seqLen * channels * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[ConvBuffers] { state.Handle, qkv.Handle, weight.Handle, bias.Handle, right.Handle, output.Handle };
        nint set = _convCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _conv.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _conv.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[3] { (uint)dConv, (uint)channels, (uint)seqLen };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _conv.Layout, VkShaderStageFlags.Compute, 0, ConvPushBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((channels + 255) / 256), (uint)seqLen, 1);
    }

    /// <summary>Records the bidirectional block softmax with the history bucket (one workgroup per block row and value head).</summary>
    /// <param name="cmdBuf">Command buffer.</param>
    /// <param name="cleanState">The committed GDN matrix state <c>[nV, dS, dS]</c> (only read).</param>
    /// <param name="q">L2-normalised queries <c>[T, nK, dS]</c>.</param>
    /// <param name="k">L2-normalised keys <c>[T, nK, dS]</c>.</param>
    /// <param name="v">Values <c>[T, nV, dS]</c>.</param>
    /// <param name="decay">Per-token decay <c>exp(g)</c> <c>[T, nV]</c>.</param>
    /// <param name="logitScale">Per-head block-logit scale.</param>
    /// <param name="historyBias">Per-head history-logit bias.</param>
    /// <param name="historyC">Per-head history-logit coefficient.</param>
    /// <param name="output">Output <c>[T, nV, dS]</c>.</param>
    /// <param name="nVHead">Value heads.</param>
    /// <param name="nKHead">Key heads.</param>
    /// <param name="dState">Head dimension.</param>
    /// <param name="seqLen">Block length.</param>
    public unsafe void RecordSoftmax(nint cmdBuf, VulkanDevice.Buffer cleanState, VulkanDevice.Buffer q, VulkanDevice.Buffer k,
        VulkanDevice.Buffer v, VulkanDevice.Buffer decay, VulkanDevice.Buffer logitScale, VulkanDevice.Buffer historyBias,
        VulkanDevice.Buffer historyC, VulkanDevice.Buffer output, int nVHead, int nKHead, int dState, int seqLen)
    {
        if (dState <= 0 || dState > MaxHeadDim) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen <= 0 || seqLen > MaxBlock) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nKHead <= 0 || nVHead % nKHead != 0) throw new ArgumentException("nVHead must be a positive multiple of nKHead.");
        if (output.Size < (long)seqLen * nVHead * dState * sizeof(float)) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[SoftmaxBuffers]
        {
            cleanState.Handle, q.Handle, k.Handle, v.Handle, decay.Handle, logitScale.Handle, historyBias.Handle, historyC.Handle, output.Handle,
        };
        nint set = _softmaxCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _softmax.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _softmax.Layout, 0, 1, set, 0, 0);
        // Same host constants as the CPU reference (CloverPreviewGdnMixer): ln(T / 32) and 1 / sqrt(dS) as F32.
        float logRatio = MathF.Log(seqLen / 32f);
        float invSqrtD = 1f / MathF.Sqrt(dState);
        Span<uint> pc = stackalloc uint[6]
        {
            (uint)nVHead, (uint)nKHead, (uint)dState, (uint)seqLen, BitConverter.SingleToUInt32Bits(logRatio), BitConverter.SingleToUInt32Bits(invSqrtD),
        };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _softmax.Layout, VkShaderStageFlags.Compute, 0, SoftmaxPushBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)seqLen, (uint)nVHead, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_convPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _convPool, 0);
        if (_softmaxPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _softmaxPool, 0);
        _conv.Dispose();
        _softmax.Dispose();
        _convModule.Dispose();
        _softmaxModule.Dispose();
    }
}
