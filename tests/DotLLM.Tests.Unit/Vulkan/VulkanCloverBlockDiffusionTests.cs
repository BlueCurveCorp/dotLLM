using DotLLM.Cpu.Kernels;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Clover block expert coreset (#1828) on Vulkan: the <c>clover_moe_coreset_f32</c> kernel against the CPU reference
/// (<see cref="MoeSwiGluMlp.ApplyBlockCoreset"/>), and the coreset-on previews and the block-diffusion generation against HF with the
/// generated oracle fixture (<see cref="CloverBlockDiffusionOracleTests"/>).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanCloverBlockDiffusionTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(32, 256, 8, 0.9f, 3f)]   // Clover-35B-A3B: 256 experts, top-8, one 32-row block
    [InlineData(32, 256, 8, 0.9f, 0.5f)] // flat routing: most experts kept
    [InlineData(5, 256, 8, 0.5f, 3f)]
    [InlineData(32, 8, 2, 0.9f, 3f)]     // the tiny oracle's shape
    [InlineData(1, 8, 2, 0.01f, 1f)]     // keeps a single expert (fewer than k)
    public void Coreset_MatchesCpu(int rows, int experts, int k, float topP, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var r = new Random(1828 + rows * experts);
        var logits = new float[rows * experts];
        // A shared per-expert bias makes the block agree on some experts (the coreset's premise), plus per-row noise.
        var bias = new float[experts];
        for (int e = 0; e < experts; e++) bias[e] = scale * (2f * r.NextSingle() - 1f);
        for (int i = 0; i < logits.Length; i++) logits[i] = bias[i % experts] + (2f * r.NextSingle() - 1f);
        var cpu = (float[])logits.Clone();
        MoeSwiGluMlp.ApplyBlockCoreset(cpu, rows, experts, k, topP);

        using var device = VulkanDevice.Create();
        using var kernels = CloverBlockPreviewKernels.Create(device, spvDir);
        using var buf = device.AllocateHostReadback((long)logits.Length * 4);
        device.Upload(logits, buf);
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
        kernels.RecordCoreset(ctx.CommandBuffer, buf, rows, experts, k, topP);
        KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
        ctx.SubmitAndWait();
        var gpu = new float[logits.Length];
        device.Download(buf, gpu);

        int kept = 0;
        for (int e = 0; e < experts; e++) if (!float.IsNegativeInfinity(cpu[e])) kept++;
        output.WriteLine($"[coreset] rows {rows} experts {experts} k {k} top_p {topP}: CPU keeps {kept}");
        for (int i = 0; i < logits.Length; i++)
        {
            bool cpuMasked = float.IsNegativeInfinity(cpu[i]), gpuMasked = gpu[i] <= -1e38f;
            Assert.True(cpuMasked == gpuMasked, $"element {i} (expert {i % experts}): CPU masked {cpuMasked}, GPU masked {gpuMasked}");
            if (!cpuMasked) Assert.Equal(logits[i], gpu[i]);
        }
        Assert.InRange(kept, 1, experts);
    }

    private static T WithVulkan<T>(string path, string spvDir, Func<VulkanQwen3MoeHybridTransformerModel, T> body)
    {
        using var gguf = GgufFile.Open(path);
        using var device = VulkanDevice.Create();
        using var m = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), spvDir);
        return body(m);
    }

    /// <summary>Coreset-on previews against HF (and so the Vulkan MoE routing with the mask in place).</summary>
    [SkippableFact]
    public void CoresetPreview_MatchesHf_OracleFixture()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? dir = CloverBlockDiffusionOracleTests.FixtureDir();
        Skip.If(dir is null, "clover generate oracle fixture not present (run scripts/clover_preview_oracle.py --mode generate)");
        var oracle = CloverBlockDiffusionOracleTests.Read(dir!);
        WithVulkan(Path.Combine(dir!, "clover-generate-tiny.gguf"), spvDir, m =>
        {
            foreach (var c in oracle.Previews)
            {
                using var kv = m.CreateKvCache(m.Config.MaxSequenceLength);
                using var gdn = m.CreateGdnStateCache();
                m.Forward(c.Prefix, Enumerable.Range(0, c.Prefix.Length).ToArray(), -1, kv, gdn, null).Dispose();
                using var t = m.ForwardBlockPreview(c.Block, c.Prefix.Length, kv, gdn, -1);
                var cmp = CloverPreviewOracleTests.Compare(CloverPreviewOracleTests.ToArray(t, c.Block.Length * oracle.Vocab), c.Logits,
                                                           oracle.Vocab, oracle.MaskTokenId);
                output.WriteLine($"[vk clover coreset preview vs HF] {c.Name}: maxAbs={cmp.MaxAbs:E2} meanKL={cmp.MeanKl:E2} maxKL={cmp.MaxKl:E2} "
                                 + $"top1={cmp.Top1:P0} (HF coreset on vs off: {c.CoresetOffDelta:F2})");
                // The Vulkan host's causal floor on this fixture family is ~1e-2 max-abs at the stage-3 logit scale (#1819), here times
                // the LM-head sharpening (8x logits, so ~64x KL): measured maxAbs 2.5e-2..1.7e-1, maxKL 8.8e-6..1.6e-3
                // (the same as Vulkan vs CPU, see CoresetPreview_VulkanVsCpu_AtTheHostFloor). The coreset on/off
                // difference is 28-48 max-abs, so a wrong keep set cannot hide under these bounds.
                Assert.True(cmp.MaxAbs < 4e-2f * oracle.LmScale, $"{c.Name}: maxAbs {cmp.MaxAbs:E3}");
                Assert.True(cmp.MaxKl < 5e-3, $"{c.Name}: maxKL {cmp.MaxKl:E3}");
                Assert.Equal(1.0, cmp.Top1);
            }
            return 0;
        });
    }

    /// <summary>
    /// Separates the coreset from the Vulkan host's numerical floor: Vulkan vs CPU with the coreset on and off, both at the same floor
    /// (same weights, same blocks). A wrong Vulkan keep set would move logits by the on/off difference (28-48).
    /// </summary>
    [SkippableFact]
    public void CoresetPreview_VulkanVsCpu_AtTheHostFloor()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? dir = CloverBlockDiffusionOracleTests.FixtureDir();
        Skip.If(dir is null, "clover generate oracle fixture not present (run scripts/clover_preview_oracle.py --mode generate)");
        var oracle = CloverBlockDiffusionOracleTests.Read(dir!);
        string path = Path.Combine(dir!, "clover-generate-tiny.gguf");
        foreach (bool coreset in new[] { true, false })
        {
            using var gguf = GgufFile.Open(path);
            var baseConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
            var config = baseConfig with { Clover = baseConfig.Clover! with { MoeBlockCoresetEnabled = coreset } };
            using var cpu = (DotLLM.Models.Architectures.Qwen3MoeHybridTransformerModel)DotLLM.Models.ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var device = VulkanDevice.Create();
            using var vk = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            float worst = 0f;
            foreach (var c in oracle.Previews)
            {
                using var kvC = new DotLLM.Engine.KvCache.SimpleKvCache(cpu.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
                using var gdnC = (DotLLM.Models.Architectures.GdnStateCache)cpu.CreateSequenceState()!;
                cpu.Forward(c.Prefix, Enumerable.Range(0, c.Prefix.Length).ToArray(), -1, kvC, gdnC).Dispose();
                using var tC = cpu.ForwardBlockPreview(c.Block, c.Prefix.Length, kvC, gdnC, -1);
                using var kvV = vk.CreateKvCache(config.MaxSequenceLength);
                using var gdnV = vk.CreateGdnStateCache();
                vk.Forward(c.Prefix, Enumerable.Range(0, c.Prefix.Length).ToArray(), -1, kvV, gdnV, null).Dispose();
                using var tV = vk.ForwardBlockPreview(c.Block, c.Prefix.Length, kvV, gdnV, -1);
                int n = c.Block.Length * oracle.Vocab;
                var cmp = CloverPreviewOracleTests.Compare(CloverPreviewOracleTests.ToArray(tV, n), CloverPreviewOracleTests.ToArray(tC, n),
                                                           oracle.Vocab, oracle.MaskTokenId);
                output.WriteLine($"[vk vs cpu, coreset {(coreset ? "on " : "off")}] {c.Name}: maxAbs={cmp.MaxAbs:E2} maxKL={cmp.MaxKl:E2} top1={cmp.Top1:P0}");
                worst = MathF.Max(worst, cmp.MaxAbs);
                Assert.Equal(1.0, cmp.Top1);
            }
            // Both arms sit at the host floor (measured worst: on 1.7e-1, off 6.2e-2; the on-arm excess is one row of prefix3_T32_half,
            // where five experts at 5-12% mass make per-row top-2 near-ties, and the keep-set margins there are >= 3.8e-3, far above
            // the drift). A keep-set fault moves logits by the on/off difference, 28-48.
            Assert.True(worst < 0.04f * oracle.LmScale, $"coreset {(coreset ? "on" : "off")}: Vulkan vs CPU maxAbs {worst:E2}");
        }
    }

    /// <summary>HF <c>generate</c> on Vulkan: the same committed ids and per-step fill order.</summary>
    [SkippableFact]
    public void Generate_MatchesHf_OracleFixture()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? dir = CloverBlockDiffusionOracleTests.FixtureDir();
        Skip.If(dir is null, "clover generate oracle fixture not present (run scripts/clover_preview_oracle.py --mode generate)");
        var oracle = CloverBlockDiffusionOracleTests.Read(dir!);
        WithVulkan(Path.Combine(dir!, "clover-generate-tiny.gguf"), spvDir, m =>
        {
            foreach (var c in oracle.Cases)
            {
                var decoder = new CloverBlockDiffusion(m, c.Threshold, c.BlockSize);
                using var kv = m.CreateKvCache(decoder.RequiredKvLength(c.Prompt.Length, c.MaxNew));
                var result = decoder.Generate(c.Prompt, c.MaxNew, kv, c.Eos);
                string want = CloverBlockDiffusionOracleTests.Trace(c.FillTrace), got = CloverBlockDiffusionOracleTests.Trace(result.Blocks);
                output.WriteLine($"[vk clover generate vs HF] {c.Name}: {result.Response.Length} tokens (min |conf - threshold| {c.MinThresholdGap:E1})"
                                 + $"\n  HF   {want}\n  ours {got}");
                Assert.Equal(c.Response, result.Response);
                Assert.Equal(want, got);
            }
            return 0;
        });
    }
}
