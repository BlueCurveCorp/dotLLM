using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Clover LowConfidence block-diffusion generation with the block expert coreset (#1828) on the CPU host against DiffuSpace's own HF
/// <c>CloverMoeModelLM.generate</c>. Fixture: <c>scripts/clover_preview_oracle.py --mode generate</c> writes
/// <c>clover-generate-tiny.gguf</c> (the preview fixture's weights with the LM head and router sharpened, coreset on, top_p 0.9) and
/// <c>generate.json</c> (HF's committed ids and per-step fill order for fixed prompts, plus coreset-on single-preview logits) to the
/// preview oracle's directory; every test skips when it is absent.
/// </summary>
public sealed class CloverBlockDiffusionOracleTests(ITestOutputHelper output)
{
    internal static string? FixtureDir()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_CLOVER_PREVIEW_ORACLE_DIR");
        string d = !string.IsNullOrEmpty(env) ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "clover-preview-oracle");
        return File.Exists(Path.Combine(d, "clover-generate-tiny.gguf")) && File.Exists(Path.Combine(d, "generate.json")) ? d : null;
    }

    internal sealed record GenCase(string Name, int[] Prompt, int MaxNew, float Threshold, int BlockSize, int[] Eos, int[] Response,
                                   int[][][] FillTrace, double MinThresholdGap);

    internal sealed record PreviewCase(string Name, int[] Prefix, int[] Block, float[] Logits, float CoresetOffDelta);

    internal sealed record GenOracle(int Vocab, int MaskTokenId, float LmScale, GenCase[] Cases, PreviewCase[] Previews);

    internal static GenOracle Read(string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "generate.json")));
        var root = doc.RootElement;
        static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        static float[] Rows(JsonElement e) => e.EnumerateArray()
            .SelectMany(r => r.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Null ? float.NegativeInfinity : x.GetSingle())).ToArray();
        var cases = root.GetProperty("cases").EnumerateArray().Select(c => new GenCase(
            c.GetProperty("name").GetString()!, Ints(c.GetProperty("prompt")), c.GetProperty("max_new_tokens").GetInt32(),
            c.GetProperty("threshold").GetSingle(), c.GetProperty("block_size").GetInt32(), Ints(c.GetProperty("eos_token_ids")),
            Ints(c.GetProperty("response")),
            c.GetProperty("blocks").EnumerateArray()
                .Select(b => b.GetProperty("steps").EnumerateArray().Select(s => Ints(s.GetProperty("filled"))).ToArray()).ToArray(),
            c.GetProperty("summary").GetProperty("min_threshold_gap").GetDouble())).ToArray();
        var previews = root.GetProperty("previews").EnumerateArray().Select(p => new PreviewCase(
            p.GetProperty("name").GetString()!, Ints(p.GetProperty("prefix")), Ints(p.GetProperty("block")), Rows(p.GetProperty("logits")),
            p.GetProperty("coreset_off_max_abs_delta").GetSingle())).ToArray();
        return new GenOracle(root.GetProperty("config").GetProperty("vocab_size").GetInt32(), root.GetProperty("mask_token_id").GetInt32(),
                             root.GetProperty("lm_scale").GetSingle(),
                             cases, previews);
    }

    internal static string Trace(int[][][] t) => string.Join(" | ", t.Select(b => string.Join(" ", b.Select(s => "[" + string.Join(",", s) + "]"))));

    internal static string Trace(IReadOnlyList<CloverResolvedBlock> blocks)
        => string.Join(" | ", blocks.Select(b => string.Join(" ", b.Steps.Select(s => "[" + string.Join(",", s.Filled) + "]"))));

    /// <summary>The coreset-on previews (localizes a coreset fault): HF logits within the F32 preview bound of the stage-3 oracle.</summary>
    [SkippableFact]
    public void CoresetPreview_MatchesHf()
    {
        string? dir = FixtureDir();
        Skip.If(dir is null, "clover generate oracle fixture not present (run scripts/clover_preview_oracle.py --mode generate)");
        var oracle = Read(dir!);
        using var gguf = GgufFile.Open(Path.Combine(dir!, "clover-generate-tiny.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.True(config.Clover!.MoeBlockCoresetEnabled);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        foreach (var c in oracle.Previews)
        {
            using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var gdn = (GdnStateCache)hybrid.CreateSequenceState()!;
            hybrid.Forward(c.Prefix, Enumerable.Range(0, c.Prefix.Length).ToArray(), deviceId: -1, kv, gdn).Dispose();
            using var logits = hybrid.ForwardBlockPreview(c.Block, c.Prefix.Length, kv, gdn, deviceId: -1);
            var cmp = CloverPreviewOracleTests.Compare(CloverPreviewOracleTests.ToArray(logits, c.Block.Length * oracle.Vocab), c.Logits,
                                                       oracle.Vocab, oracle.MaskTokenId);
            output.WriteLine($"[clover coreset preview vs HF] {c.Name}: maxAbs={cmp.MaxAbs:E2} meanKL={cmp.MeanKl:E2} top1={cmp.Top1:P0} "
                             + $"(HF coreset on vs off: {c.CoresetOffDelta:F2})");
            // The stage-3 F32 bound (2e-4), scaled by the fixture's LM-head sharpening (the logits are lm_scale times larger); still
            // five orders of magnitude below the coreset on/off difference.
            Assert.True(cmp.MaxAbs < 2e-4f * oracle.LmScale, $"{c.Name}: preview maxAbs {cmp.MaxAbs:E3}");
            Assert.True(cmp.MeanKl < 1e-8, $"{c.Name}: preview mean KL {cmp.MeanKl:E3}");
            Assert.Equal(1.0, cmp.Top1);
        }
    }

    /// <summary>HF <c>generate</c>: the same committed ids and the same per-step fill order, for every fixed prompt.</summary>
    [SkippableFact]
    public void Generate_MatchesHf()
    {
        string? dir = FixtureDir();
        Skip.If(dir is null, "clover generate oracle fixture not present (run scripts/clover_preview_oracle.py --mode generate)");
        var oracle = Read(dir!);
        using var gguf = GgufFile.Open(Path.Combine(dir!, "clover-generate-tiny.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        foreach (var c in oracle.Cases)
        {
            var decoder = new CloverBlockDiffusion(model, c.Threshold, c.BlockSize);
            using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim,
                                             decoder.RequiredKvLength(c.Prompt.Length, c.MaxNew));
            var result = decoder.Generate(c.Prompt, c.MaxNew, kv, c.Eos);
            string want = Trace(c.FillTrace), got = Trace(result.Blocks);
            output.WriteLine($"[clover generate vs HF] {c.Name}: {result.Response.Length} tokens in {result.Blocks.Sum(b => b.Steps.Count)} steps "
                             + $"(min |conf - threshold| {c.MinThresholdGap:E1})\n  HF   {want}\n  ours {got}");
            Assert.Equal(c.Response, result.Response);
            Assert.Equal(want, got);
        }
    }
}
