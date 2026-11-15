using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Clover bidirectional block preview (#1815) on the CPU host against DiffuSpace's own HF modeling code. The fixture is a synthetic tiny
/// Clover (6 layers = [GDN, GDN, attention] x 2; GDN nK 2 / nV 4 so grouped and tiled V-head orders differ) produced by
/// <c>scripts/clover_preview_oracle.py</c>: random F32 weights in torch, written to GGUF in the official file's layout, plus the HF
/// <c>preview=True</c> logits. Fixture dir: <c>~/.dotllm/test-cache/clover-preview-oracle/</c> (override
/// <c>DOTLLM_CLOVER_PREVIEW_ORACLE_DIR</c>); every test skips when it is absent.
/// </summary>
public sealed class CloverPreviewOracleTests(ITestOutputHelper output)
{
    internal static string? FixtureDir()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_CLOVER_PREVIEW_ORACLE_DIR");
        string d = !string.IsNullOrEmpty(env) ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "clover-preview-oracle");
        return File.Exists(Path.Combine(d, "clover-preview-tiny.gguf")) && File.Exists(Path.Combine(d, "oracle.json")) ? d : null;
    }

    internal sealed record OracleCase(string Name, int[] Prefix, int[] Block, float[] Logits);

    internal sealed record Oracle(int Vocab, int MaskTokenId, int[] CausalTokens, float[] CausalLogits, OracleCase[] Cases);

    internal static Oracle ReadOracle(string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "oracle.json")));
        var root = doc.RootElement;
        int vocab = root.GetProperty("config").GetProperty("vocab_size").GetInt32();
        static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        static float[] Rows(JsonElement e) => e.EnumerateArray()
            .SelectMany(r => r.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Null ? float.NegativeInfinity : x.GetSingle())).ToArray();
        var causal = root.GetProperty("causal");
        var cases = root.GetProperty("cases").EnumerateArray()
            .Select(c => new OracleCase(c.GetProperty("name").GetString()!, Ints(c.GetProperty("prefix")), Ints(c.GetProperty("block")),
                                        Rows(c.GetProperty("logits"))))
            .ToArray();
        return new Oracle(vocab, root.GetProperty("mask_token_id").GetInt32(), Ints(causal.GetProperty("tokens")), Rows(causal.GetProperty("logits")), cases);
    }

    /// <summary>Max |Δ| over finite entries, mean / max KL(ref || ours) per row, top-1 agreement; the mask column is excluded.</summary>
    internal static (float MaxAbs, double MeanKl, double MaxKl, double Top1) Compare(float[] ours, float[] reference, int vocab, int mask)
    {
        int rows = reference.Length / vocab;
        float maxAbs = 0f; double klSum = 0, klMax = 0; int agree = 0;
        for (int r = 0; r < rows; r++)
        {
            var a = ours.AsSpan(r * vocab, vocab);
            var b = reference.AsSpan(r * vocab, vocab);
            double ma = double.NegativeInfinity, mb = double.NegativeInfinity; int ia = 0, ib = 0;
            for (int i = 0; i < vocab; i++)
            {
                if (i == mask) continue;
                maxAbs = MathF.Max(maxAbs, MathF.Abs(a[i] - b[i]));
                if (a[i] > ma) { ma = a[i]; ia = i; }
                if (b[i] > mb) { mb = b[i]; ib = i; }
            }
            if (ia == ib) agree++;
            double za = 0, zb = 0;
            for (int i = 0; i < vocab; i++) { if (i == mask) continue; za += Math.Exp(a[i] - ma); zb += Math.Exp(b[i] - mb); }
            double kl = 0;
            for (int i = 0; i < vocab; i++)
            {
                if (i == mask) continue;
                double lp = b[i] - mb - Math.Log(zb), lq = a[i] - ma - Math.Log(za);
                kl += Math.Exp(lp) * (lp - lq);
            }
            klSum += kl; klMax = Math.Max(klMax, kl);
        }
        return (maxAbs, klSum / rows, klMax, (double)agree / rows);
    }

    internal static float[] ToArray(ITensor t, int count)
    {
        var a = new float[count];
        unsafe { new ReadOnlySpan<float>((void*)t.DataPointer, count).CopyTo(a); }
        return a;
    }

    /// <summary>The writer check: dotLLM's causal forward on the fixture GGUF matches HF's causal logits before the preview is compared.</summary>
    [SkippableFact]
    public void Causal_MatchesHf()
    {
        string? dir = FixtureDir();
        Skip.If(dir is null, "clover-preview-oracle fixture not present (run scripts/clover_preview_oracle.py)");
        var oracle = ReadOracle(dir!);
        using var gguf = GgufFile.Open(Path.Combine(dir!, "clover-preview-tiny.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        int n = oracle.CausalTokens.Length;
        using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var logits = model.Forward(oracle.CausalTokens, Enumerable.Range(0, n).ToArray(), deviceId: -1, kv);
        var cmp = Compare(ToArray(logits, n * oracle.Vocab), oracle.CausalLogits, oracle.Vocab, mask: -1);
        output.WriteLine($"[clover causal vs HF] maxAbs={cmp.MaxAbs:E2} meanKL={cmp.MeanKl:E2} maxKL={cmp.MaxKl:E2} top1={cmp.Top1:P0}");
        Assert.True(cmp.MaxAbs < 2e-4f, $"causal maxAbs {cmp.MaxAbs:E3}");
        Assert.Equal(1.0, cmp.Top1);
    }

    /// <summary>
    /// Every oracle case: prefix committed causally, then <see cref="ICloverBlockPreview.ForwardBlockPreview"/> over the block. Bounds are
    /// F32-vs-F32 (the only expected differences are summation order and the L2-norm epsilon form); the mutants recorded in the PR move
    /// the logits by orders of magnitude more.
    /// </summary>
    [SkippableFact]
    public void Preview_MatchesHf()
    {
        string? dir = FixtureDir();
        Skip.If(dir is null, "clover-preview-oracle fixture not present (run scripts/clover_preview_oracle.py)");
        var oracle = ReadOracle(dir!);
        using var gguf = GgufFile.Open(Path.Combine(dir!, "clover-preview-tiny.gguf"));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
        Assert.True(hybrid.SupportsBlockPreview);

        float worst = 0f;
        foreach (var c in oracle.Cases)
        {
            using var kv = new SimpleKvCache(hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var gdn = (GdnStateCache)hybrid.CreateSequenceState()!;
            if (c.Prefix.Length > 0)
                hybrid.Forward(c.Prefix, Enumerable.Range(0, c.Prefix.Length).ToArray(), deviceId: -1, kv, gdn).Dispose();
            using var logits = hybrid.ForwardBlockPreview(c.Block, c.Prefix.Length, kv, gdn, deviceId: -1);
            float[] ours = ToArray(logits, c.Block.Length * oracle.Vocab);
            for (int r = 0; r < c.Block.Length; r++)
                Assert.Equal(float.NegativeInfinity, ours[r * oracle.Vocab + oracle.MaskTokenId]);
            var cmp = Compare(ours, c.Logits, oracle.Vocab, oracle.MaskTokenId);
            output.WriteLine($"[clover preview vs HF] {c.Name}: maxAbs={cmp.MaxAbs:E2} meanKL={cmp.MeanKl:E2} maxKL={cmp.MaxKl:E2} top1={cmp.Top1:P0}");
            worst = MathF.Max(worst, cmp.MaxAbs);
            Assert.True(cmp.MaxAbs < 2e-4f, $"{c.Name}: preview maxAbs {cmp.MaxAbs:E3}");
            Assert.Equal(1.0, cmp.Top1);
        }
        output.WriteLine($"[clover preview vs HF] worst maxAbs={worst:E2}");
    }
}
