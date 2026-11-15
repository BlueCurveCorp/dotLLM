using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Smoke check of the Clover block preview (#1815) on the official 35B GGUF (<c>DOTLLM_CLOVER_GGUF</c> or the HF cache): a natural-text
/// prefix, then a block holding the true continuation with every other position masked. The preview should fill most masked positions
/// with the true token (it is a denoiser) and the Vulkan host must track the CPU host. Numerical parity against HF is the synthetic
/// oracle's job (<c>CloverPreviewOracleTests</c>); a full-size torch run is ~70 GB.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CloverRealGgufPreviewTests(ITestOutputHelper output)
{
    // Tokens 0..47 of CloverRealGgufVulkanTests' paragraph: 40 committed, 8 in the block.
    private static readonly int[] Text =
    [
        760, 12386, 19825, 557, 279, 1675, 12, 66139, 1528, 314, 13437, 21047, 13, 1049, 369, 8524, 15513, 310, 2977, 279, 4031, 321,
        17435, 20632, 539, 279, 47091, 2614, 4756, 57177, 579, 23571, 314, 12924, 5728, 1172, 279, 43051, 344, 303, 220, 17, 22, 17501,
        13, 1049, 5064, 37167,
    ];

    private const int PrefixLen = 40;
    private const int Mask = 151669;

    private static int[] BlockTokens()
    {
        var b = Text[PrefixLen..].ToArray();
        for (int i = 0; i < b.Length; i += 2) b[i] = Mask;   // mask positions 0, 2, 4, 6
        return b;
    }

    private static float[][] Rows(ITensor t, int vocab)
    {
        var rows = new float[t.Shape[0]][];
        unsafe
        {
            for (int r = 0; r < rows.Length; r++)
                rows[r] = new ReadOnlySpan<float>((float*)t.DataPointer + (long)r * vocab, vocab).ToArray();
        }
        return rows;
    }

    private static double KlFinite(float[] p, float[] q)
    {
        var pp = p.Select(x => float.IsFinite(x) ? x : -1e30f).ToArray();
        var qq = q.Select(x => float.IsFinite(x) ? x : -1e30f).ToArray();
        return CloverRealGguf.Kl(pp, qq);
    }

    [SkippableFact]
    public void Preview_FillsMaskedText_AndVulkanTracksCpu()
    {
        string? path = CloverRealGguf.Resolve();
        Skip.If(path is null, "Clover GGUF not found: set DOTLLM_CLOVER_GGUF or hf-download Diffuspace/Clover-35B-A3B-Preview-GGUF.");
        string? spvDir = CloverRealGgufVulkanTests.ResolveSpvDir();
        Skip.If(spvDir is null, "Vulkan SPV directory not found.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int vocab = config.VocabSize;
        int[] block = BlockTokens();
        int[] prefixPos = Enumerable.Range(0, PrefixLen).ToArray();

        float[][] cpu;
        int cpuCausalNext;
        using (var model = (Qwen3MoeHybridTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config))
        {
            Assert.True(model.SupportsBlockPreview);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, 128);
            using var gdn = (GdnStateCache)model.CreateSequenceState()!;
            using (var pre = model.Forward(Text.AsSpan(0, PrefixLen), prefixPos, -1, kv, gdn))
                cpuCausalNext = CloverRealGguf.Top1(CloverRealGguf.LastRow(pre, vocab)).Top;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using (var p = model.ForwardBlockPreview(block, PrefixLen, kv, gdn, -1)) cpu = Rows(p, vocab);
            output.WriteLine($"[clover preview cpu] {sw.ElapsedMilliseconds} ms for T={block.Length}");
            Assert.Equal(PrefixLen, kv.CurrentLength);
        }

        float[][] vk;
        using (var device = VulkanDevice.Create())
        using (var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, spvDir!))
        {
            using var kv = model.CreateKvCache(128);
            using var gdn = model.CreateGdnStateCache();
            model.Forward(Text.AsSpan(0, PrefixLen), prefixPos, -1, kv, gdn, null).Dispose();
            using (var warm = model.ForwardBlockPreview(block, PrefixLen, kv, gdn, -1)) { }
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using (var p = model.ForwardBlockPreview(block, PrefixLen, kv, gdn, -1)) vk = Rows(p, vocab);
            output.WriteLine($"[clover preview vk] {sw.ElapsedMilliseconds} ms for T={block.Length} (second call)");
            Assert.Equal(PrefixLen, kv.CurrentLength);
        }

        int filledCpu = 0, filledVk = 0, masked = 0, agree = 0;
        double kl = 0;
        for (int i = 0; i < block.Length; i++)
        {
            int truth = Text[PrefixLen + i];
            var (tc, mc) = CloverRealGguf.Top1(cpu[i]);
            var (tv, _) = CloverRealGguf.Top1(vk[i]);
            double k = KlFinite(cpu[i], vk[i]);
            kl += k;
            if (tc == tv) agree++;
            Assert.Equal(float.NegativeInfinity, cpu[i][Mask]);
            Assert.Equal(float.NegativeInfinity, vk[i][Mask]);
            if (block[i] == Mask) { masked++; if (tc == truth) filledCpu++; if (tv == truth) filledVk++; }
            output.WriteLine($"  pos {i} {(block[i] == Mask ? "MASK" : "tok ")} truth={truth} cpu={tc} (margin {mc:F2}) vk={tv} KL={k:E2}");
        }
        kl /= block.Length;
        output.WriteLine($"[clover preview] masked filled with the true token: cpu {filledCpu}/{masked}, vk {filledVk}/{masked}; "
                         + $"cpu causal next={cpuCausalNext} (truth {Text[PrefixLen]}); vk-vs-cpu meanKL={kl:E3} top1={agree}/{block.Length}");
        Assert.True(filledCpu * 2 >= masked, $"the preview recovered only {filledCpu}/{masked} masked tokens");
        Assert.True(kl < 0.08, $"mean KL(cpu || vk) {kl:E3}");   // the causal Vulkan-vs-CPU gate's bound (#1813)
        Assert.True(agree * 4 >= block.Length * 3, $"top-1 agreement {agree}/{block.Length}");
    }
}
