using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using CloverLayout = DotLLM.Models.Gguf.SyntheticQwen35MoeGguf.CloverLayout;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Clover bidirectional block preview (#1815) on Vulkan against the CPU host (and, with the generated oracle fixture, against HF).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanCloverPreviewTests(ITestOutputHelper output) : IDisposable
{
    private static readonly int[] Prefix = [1, 3, 0, 2, 5, 4, 6, 3];
    private static readonly int[] Block = [7, 3, 7, 7, 5, 7];
    private static readonly int[] Continuation = [4, 2, 6];
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-clover-preview-{Guid.NewGuid():N}");

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string SyntheticFixture()
    {
        Directory.CreateDirectory(_scratch);
        return SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "clover.gguf"), contextLength: 64, blockCount: 4, clover: CloverLayout.Official);
    }

    private static float[] Read(ITensor t)
    {
        var a = new float[t.Shape.ElementCount];
        unsafe { new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a); }
        return a;
    }

    private static int[] Range(int start, int n) => Enumerable.Range(start, n).ToArray();

    private static float[] CpuPreview(string path, int[] prefix, int[] block)
    {
        using var gguf = GgufFile.Open(path);
        using var m = (Qwen3MoeHybridTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, GgufModelConfigExtractor.Extract(gguf.Metadata));
        using var kv = new SimpleKvCache(m.AttentionLayerCount, m.Config.NumKvHeads, m.Config.HeadDim, m.Config.MaxSequenceLength);
        using var gdn = (GdnStateCache)m.CreateSequenceState()!;
        if (prefix.Length > 0) m.Forward(prefix, Range(0, prefix.Length), -1, kv, gdn).Dispose();
        using var t = m.ForwardBlockPreview(block, prefix.Length, kv, gdn, -1);
        return Read(t);
    }

    private static T WithVulkan<T>(string path, string spvDir, Func<VulkanQwen3MoeHybridTransformerModel, T> body)
    {
        using var gguf = GgufFile.Open(path);
        using var device = VulkanDevice.Create();
        using var m = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), spvDir);
        return body(m);
    }

    private static float[] VkPreview(VulkanQwen3MoeHybridTransformerModel m, int[] prefix, int[] block)
    {
        using var kv = m.CreateKvCache(m.Config.MaxSequenceLength);
        using var gdn = m.CreateGdnStateCache();
        if (prefix.Length > 0) m.Forward(prefix, Range(0, prefix.Length), -1, kv, gdn, null).Dispose();
        using var t = m.ForwardBlockPreview(block, prefix.Length, kv, gdn, -1);
        return Read(t);
    }

    /// <summary>Vulkan vs CPU on the CI fixture (F32 weights, so the bound is summation order only).</summary>
    [SkippableFact]
    public void Preview_MatchesCpu_SyntheticFixture()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = SyntheticFixture();
        float[] cpu = CpuPreview(path, Prefix, Block);
        float[] vk = WithVulkan(path, spvDir, m =>
        {
            Assert.True(m.SupportsBlockPreview);
            return VkPreview(m, Prefix, Block);
        });
        var cmp = CloverPreviewOracleTests.Compare(vk, cpu, SyntheticQwen35MoeGguf.VocabSize, SyntheticQwen35MoeGguf.VocabSize - 1);
        output.WriteLine($"[clover preview vk vs cpu, synthetic] maxAbs={cmp.MaxAbs:E2} maxKL={cmp.MaxKl:E2} top1={cmp.Top1:P0}");
        Assert.True(cmp.MaxAbs < 1e-4f, $"maxAbs {cmp.MaxAbs:E3}");
        Assert.Equal(1.0, cmp.Top1);
    }

    /// <summary>The stage-4 contract on the device: previews repeat bit for bit, the KV length is kept, and a causal continuation after
    /// two previews equals one without them.</summary>
    [SkippableFact]
    public void Preview_LeavesCommittedStateUnchanged()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = SyntheticFixture();
        WithVulkan(path, spvDir, m =>
        {
            using var kv = m.CreateKvCache(m.Config.MaxSequenceLength);
            using var gdn = m.CreateGdnStateCache();
            m.Forward(Prefix, Range(0, Prefix.Length), -1, kv, gdn, null).Dispose();
            float[] a, b;
            using (var t = m.ForwardBlockPreview(Block, Prefix.Length, kv, gdn, -1)) a = Read(t);
            using (var t = m.ForwardBlockPreview(Block, Prefix.Length, kv, gdn, -1)) b = Read(t);
            Assert.Equal(a, b);
            Assert.Equal(Prefix.Length, kv.CurrentLength);
            float[] after;
            using (var t = m.Forward(Continuation, Range(Prefix.Length, Continuation.Length), -1, kv, gdn, null)) after = Read(t);

            using var kv2 = m.CreateKvCache(m.Config.MaxSequenceLength);
            using var gdn2 = m.CreateGdnStateCache();
            m.Forward(Prefix, Range(0, Prefix.Length), -1, kv2, gdn2, null).Dispose();
            using var plain = m.Forward(Continuation, Range(Prefix.Length, Continuation.Length), -1, kv2, gdn2, null);
            Assert.Equal(Read(plain), after);
            return 0;
        });
    }

    /// <summary>Every case of the generated HF oracle (skips without it): Vulkan vs CPU, and both vs HF.</summary>
    [SkippableFact]
    public void Preview_MatchesCpuAndHf_OracleFixture()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? dir = CloverPreviewOracleTests.FixtureDir();
        Skip.If(dir is null, "clover-preview-oracle fixture not present (run scripts/clover_preview_oracle.py)");
        var oracle = CloverPreviewOracleTests.ReadOracle(dir!);
        string path = Path.Combine(dir!, "clover-preview-tiny.gguf");
        var cpu = oracle.Cases.Select(c => CpuPreview(path, c.Prefix, c.Block)).ToArray();
        float[]? vkCausal = null;
        var vk = WithVulkan(path, spvDir, m =>
        {
            Assert.True(m.TrySetAllRowLogitsLimit(oracle.CausalTokens.Length));
            using (var kv = m.CreateKvCache(m.Config.MaxSequenceLength))
            using (var g = m.CreateGdnStateCache())
            using (var t = m.Forward(oracle.CausalTokens, Range(0, oracle.CausalTokens.Length), -1, kv, g, null))
                vkCausal = Read(t);
            return oracle.Cases.Select(c => VkPreview(m, c.Prefix, c.Block)).ToArray();
        });
        // The device's causal forward on the same fixture is the precision floor the preview inherits (shared projections / MoE / head).
        var causal = CloverPreviewOracleTests.Compare(vkCausal!, oracle.CausalLogits, oracle.Vocab, mask: -1);
        output.WriteLine($"[clover causal vk vs HF] maxAbs={causal.MaxAbs:E2} maxKL={causal.MaxKl:E2} top1={causal.Top1:P0} (rows={vkCausal!.Length / oracle.Vocab})");
        for (int i = 0; i < oracle.Cases.Length; i++)
        {
            var c = oracle.Cases[i];
            var vc = CloverPreviewOracleTests.Compare(vk[i], cpu[i], oracle.Vocab, oracle.MaskTokenId);
            var vh = CloverPreviewOracleTests.Compare(vk[i], c.Logits, oracle.Vocab, oracle.MaskTokenId);
            output.WriteLine($"[clover preview vk] {c.Name}: vs CPU maxAbs={vc.MaxAbs:E2} maxKL={vc.MaxKl:E2} top1={vc.Top1:P0} | "
                             + $"vs HF maxAbs={vh.MaxAbs:E2} maxKL={vh.MaxKl:E2} top1={vh.Top1:P0}");
            // The device's CAUSAL forward on this fixture is already ~1e-2 maxAbs / ~1e-5 KL from HF (shared projection / MoE kernels at
            // hidden 64; the preview kernels themselves are pinned to ~1e-6 by VulkanCloverPreviewKernelTests). The bound is that floor
            // with margin; the mutants recorded in the PR land at KL 1e-1..1e0.
            Assert.True(vh.MaxKl < 1e-4, $"{c.Name}: vk vs HF maxKL {vh.MaxKl:E3}");
            Assert.True(vh.MaxAbs < 3e-2f, $"{c.Name}: vk vs HF maxAbs {vh.MaxAbs:E3}");
            Assert.True(vc.MaxKl < 1e-4, $"{c.Name}: vk vs CPU maxKL {vc.MaxKl:E3}");
            Assert.Equal(1.0, vh.Top1);
        }
    }
}
