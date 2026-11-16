using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The two Clover block-preview kernels (#1815) against the CPU reference math on random data, at the tiny-oracle shape and at
/// Clover-35B-A3B's (nV 32 / nK 16 / dS 128, conv 8192 channels, T 32). Both shapes have nK &gt; 1 and nV/nK &gt; 1, so a grouped head map
/// would fail; T &lt; 32 cases make <c>ln(T/32)</c> non-zero.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanCloverPreviewKernelTests(ITestOutputHelper output)
{
    private static float[] Rand(Random r, int n, float scale, float offset = 0f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = offset + scale * (2f * r.NextSingle() - 1f);
        return a;
    }

    private static void L2(float[] x, int d)
    {
        for (int h = 0; h < x.Length / d; h++)
        {
            float s = 0f;
            for (int i = 0; i < d; i++) s += x[h * d + i] * x[h * d + i];
            float inv = 1f / (MathF.Sqrt(s) + 1e-6f);
            for (int i = 0; i < d; i++) x[h * d + i] *= inv;
        }
    }

    private static float MaxRel(float[] a, float[] b)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]) / (1f + MathF.Abs(b[i])));
        return m;
    }

    [SkippableTheory]
    [InlineData(4, 2, 16, 8)]
    [InlineData(4, 2, 16, 32)]
    [InlineData(32, 16, 128, 32)]
    [InlineData(32, 16, 128, 5)]
    public void Softmax_MatchesCpu(int nV, int nK, int dS, int T)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var r = new Random(1815 + T);
        float[] clean = Rand(r, nV * dS * dS, 0.5f), q = Rand(r, T * nK * dS, 1f), k = Rand(r, T * nK * dS, 1f), v = Rand(r, T * nV * dS, 1f);
        L2(q, dS); L2(k, dS);
        float[] decay = Rand(r, T * nV, 0.45f, 0.5f), scale = Rand(r, nV, 10f, 14f), bias = Rand(r, nV, 2.5f, 5.5f), c = Rand(r, nV, 1.1f, 0.1f);
        var cpu = new float[T * nV * dS];
        CloverPreviewGdnMixer.BlockSoftmax(clean, q, k, v, decay, scale, bias, c, cpu, nV, nK, dS, T);

        using var device = VulkanDevice.Create();
        using var kernels = CloverBlockPreviewKernels.Create(device, spvDir);
        VulkanDevice.Buffer Up(float[] a) { var b = device.Allocate((long)a.Length * 4); device.Upload(a, b); return b; }
        using var bClean = Up(clean); using var bq = Up(q); using var bk = Up(k); using var bv = Up(v); using var bd = Up(decay);
        using var bs = Up(scale); using var bb = Up(bias); using var bc = Up(c);
        using var bo = device.AllocateHostReadback((long)cpu.Length * 4);
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        kernels.RecordSoftmax(ctx.CommandBuffer, bClean, bq, bk, bv, bd, bs, bb, bc, bo, nV, nK, dS, T);
        KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
        ctx.SubmitAndWait();
        var gpu = new float[cpu.Length];
        device.Download(bo, gpu);
        float rel = MaxRel(gpu, cpu);
        output.WriteLine($"[clover bd softmax] nV={nV} nK={nK} dS={dS} T={T}: maxRel={rel:E2}");
        Assert.True(rel < 1e-5f, $"maxRel {rel:E3}");
    }

    [SkippableTheory]
    [InlineData(96, 8)]
    [InlineData(8192, 32)]
    [InlineData(8192, 2)]
    [InlineData(8192, 1)]
    public void ConvSilu_MatchesCpu(int channels, int T)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int dC = 4;
        var r = new Random(channels + T);
        float[] state = Rand(r, (dC - 1) * channels, 1f), qkv = Rand(r, T * channels, 1f), w = Rand(r, dC * channels, 0.5f),
                right = Rand(r, (dC - 1) * channels, 0.5f), bias = new float[channels];

        var convIn = new float[(dC - 1 + T) * channels];
        state.CopyTo(convIn, 0);
        qkv.CopyTo(convIn, (dC - 1) * channels);
        var cpu = new float[T * channels];
        DotLLM.Cpu.Kernels.Conv1dCausal.Execute(convIn, w, bias, cpu, dC, channels, T);
        CloverPreviewGdnMixer.AddRightConv(qkv, right, cpu, dC, channels, T);
        DotLLM.Cpu.Kernels.SiLu.Execute(cpu, cpu);

        using var device = VulkanDevice.Create();
        using var kernels = CloverBlockPreviewKernels.Create(device, spvDir);
        VulkanDevice.Buffer Up(float[] a) { var b = device.Allocate((long)a.Length * 4); device.Upload(a, b); return b; }
        using var bs = Up(state); using var bq = Up(qkv); using var bw = Up(w); using var bb = Up(bias); using var br = Up(right);
        using var bo = device.AllocateHostReadback((long)cpu.Length * 4);
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        kernels.RecordConvSilu(ctx.CommandBuffer, bs, bq, bw, bb, br, bo, dC, channels, T);
        KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
        ctx.SubmitAndWait();
        var gpu = new float[cpu.Length];
        device.Download(bo, gpu);
        float rel = MaxRel(gpu, cpu);
        output.WriteLine($"[clover bd conv] channels={channels} T={T}: maxRel={rel:E2}");
        Assert.True(rel < 1e-5f, $"maxRel {rel:E3}");
    }
}
