using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// CI coverage of Clover's LowConfidence block decoding (#1828) and the block expert coreset, independent of the generated HF fixture
/// (<c>CloverBlockDiffusionOracleTests</c> holds the HF parity): a scripted preview model whose predictions depend on every committed
/// token, and hand-computed coreset cases.
/// </summary>
public sealed class CloverBlockDiffusionTests
{
    private const int Vocab = 16, Mask = 15, Block = 4;

    /// <summary>
    /// Scripted Clover: row <c>i</c> of a preview predicts a token that hashes the committed history (so a skipped or misplaced causal
    /// commit changes every later prediction); its confidence is 0.99 when <c>(i + filled) % 3 == 0</c> (filled = resolved positions in
    /// the block), else <c>0.2 + 0.05 i</c> (distinct, so the fallback choice is unambiguous).
    /// </summary>
    private sealed class ScriptedClover : IModel, ICloverBlockPreview
    {
        public readonly List<(int[] Tokens, int Start)> Commits = [];
        public readonly List<int> PreviewStarts = [];
        private long _history = 17;

        public ModelConfig Config { get; } = new()
        {
            Architecture = Architecture.Qwen3MoeHybrid, VocabSize = Vocab, HiddenSize = 8, IntermediateSize = 8, NumLayers = 1,
            NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 8, MaxSequenceLength = 256, AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None, ActivationFunction = ActivationFunction.SiLU, NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            Clover = new CloverConfig { BlockSize = Block, MaskTokenId = Mask, MoeBlockCoresetEnabled = true, MoeBlockCoresetTopP = 0.9f },
        };

        public long ComputeMemoryBytes => 0;
        public bool SupportsBlockPreview => true;

        public static float HighConfidence => 0.99f;

        public static float Confidence(int row, int filled) => (row + filled) % 3 == 0 ? HighConfidence : 0.2f + 0.05f * row;

        public int Predicted(int row, int start) => (int)((_history * 31 + row * 7 + start) % (Vocab - 2)) + 1;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => Commit(tokenIds, positions);
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => Commit(tokenIds, positions);
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
            => Commit(tokenIds, positions);
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter,
            AttentionMaskSpec maskSpec) => Commit(tokenIds, positions);

        private ITensor Commit(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
        {
            Commits.Add((tokens.ToArray(), positions[0]));
            for (int i = 0; i < tokens.Length; i++) _history = (_history * 131 + tokens[i] * (positions[i] + 1)) % 1_000_003;
            return UnmanagedTensor.Allocate(new TensorShape(1, Vocab), DType.Float32, -1);
        }

        public unsafe ITensor ForwardBlockPreview(ReadOnlySpan<int> blockTokenIds, int startPosition, IKvCache? kvCache, IGdnState? gdnState, int deviceId)
        {
            PreviewStarts.Add(startPosition);
            Assert.Equal(Block, blockTokenIds.Length);   // every block is previewed at full length, also the last one
            int filled = 0;
            foreach (int t in blockTokenIds) if (t != Mask) filled++;
            var t2 = UnmanagedTensor.Allocate(new TensorShape(blockTokenIds.Length, Vocab), DType.Float32, -1);
            float* p = (float*)t2.DataPointer;
            for (int r = 0; r < blockTokenIds.Length; r++)
            {
                // One token at logit a, the other Vocab - 2 non-mask tokens at 0: p = e^a / (e^a + Vocab - 2).
                float c = Confidence(r, filled);
                float a = MathF.Log(c * (Vocab - 2) / (1 - c));
                for (int v = 0; v < Vocab; v++) p[r * Vocab + v] = 0f;
                p[r * Vocab + Predicted(r, startPosition)] = a;
                p[r * Vocab + Mask] = float.NegativeInfinity;
            }
            return t2;
        }

        public void Dispose() { }
    }

    private static SimpleKvCache Kv() => new(1, 1, 8, 64);

    [Fact]
    public void ResolveBlock_FillsAboveThreshold_ElseTheMostConfident()
    {
        var model = new ScriptedClover();
        var decoder = new CloverBlockDiffusion(model, threshold: 0.95f);
        using var kv = Kv();
        var steps = decoder.ResolveBlock(kv, committedLength: 0).Steps;

        // Independent replay of the rule over the scripted confidences.
        var pending = new HashSet<int> { 0, 1, 2, 3 };
        int filledSoFar = 0;
        foreach (var s in steps)
        {
            var above = pending.Where(r => ScriptedClover.Confidence(r, filledSoFar) > 0.95f).OrderBy(r => r).ToArray();
            int[] expected = above.Length > 0 ? above : [pending.OrderByDescending(r => ScriptedClover.Confidence(r, filledSoFar)).ThenBy(r => r).First()];
            Assert.Equal(expected, s.Filled);
            Assert.Equal(above.Length == 0, s.Fallback);
            foreach (int r in expected) pending.Remove(r);
            filledSoFar += expected.Length;
        }
        Assert.Empty(pending);
        Assert.Contains(steps, s => s.Fallback);
        Assert.Contains(steps, s => !s.Fallback);
    }

    [Fact]
    public void Threshold_IsStrict_AndChangesTheTrace()
    {
        using var kv = Kv();
        // At threshold == the high confidence nothing passes (strict >): every step is a single fallback fill.
        var atHigh = new CloverBlockDiffusion(new ScriptedClover(), threshold: ScriptedClover.HighConfidence).ResolveBlock(kv, 0).Steps;
        Assert.All(atHigh, s => Assert.True(s.Fallback && s.Filled.Length == 1));
        var below = new CloverBlockDiffusion(new ScriptedClover(), threshold: 0.95f).ResolveBlock(kv, 0).Steps;
        Assert.NotEqual(atHigh.Count, below.Count);
    }

    [Fact]
    public void Generate_CommitsEachBlockCausally_ButNotTheLast()
    {
        var model = new ScriptedClover();
        var decoder = new CloverBlockDiffusion(model);
        using var kv = Kv();
        int[] prompt = [3, 4, 5];
        var result = decoder.Generate(prompt, maxNewTokens: 10, kv, eosTokenIds: [Vocab + 100]);

        Assert.Equal(10, result.Response.Length);                     // the last block (4 + 4 + 2) is cut to the budget
        Assert.Equal(3, result.Blocks.Count);
        Assert.Equal([0, 3, 7], model.Commits.Select(c => c.Start).ToArray());   // prompt, block 1, block 2; no commit after block 3
        Assert.Equal(prompt, model.Commits[0].Tokens);
        Assert.Equal(result.Response[..4], model.Commits[1].Tokens);
        Assert.Equal(result.Response[4..8], model.Commits[2].Tokens);
        Assert.Equal(new[] { 3, 7, 11 }, model.PreviewStarts.Distinct().ToArray());
        // The predictions read the committed history: block 2's tokens are the scripted ones for the history after block 1.
        Assert.Equal(result.Blocks[1].Tokens, result.Response[4..8]);
    }

    [Fact]
    public void Generate_StopsAfterTheBlockHoldingEos_AndDropsWhatFollows()
    {
        var probe = new ScriptedClover();
        using (var kv0 = Kv())
            _ = new CloverBlockDiffusion(probe).Generate([3, 4, 5], 12, kv0, [Vocab + 100]);
        // An id that first appears mid-block in block 2 of the free run becomes the EOS.
        var free = probe.Commits.Count;   // 3 commits (prompt, block 1, block 2)
        Assert.Equal(3, free);
        int[] block2 = probe.Commits[2].Tokens;
        int eos = block2[1];

        var model = new ScriptedClover();
        using var kv = Kv();
        var result = new CloverBlockDiffusion(model).Generate([3, 4, 5], 12, kv, [eos]);
        Assert.True(result.StoppedOnEos);
        int at = Array.IndexOf(result.Response, eos);
        Assert.Equal(result.Response.Length - 1, at);                  // EOS is the last returned token
        Assert.True(result.Response.Length < 12);
        Assert.Equal(result.Blocks.Count, model.Commits.Count);       // the prompt + every block but the stopped one, never committed
        Assert.True(result.Blocks[^1].Tokens.Length == Block);        // the stopped block was fully resolved before the stop
    }

    private sealed class StubTokenizer : DotLLM.Tokenizers.ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 1;
        public int EosTokenId => 0;
        public int[] Encode(string text) => [3, 4, 5];
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(" ", tokenIds.ToArray().Select(i => $"t{i}"));
        public string DecodeToken(int tokenId) => $"t{tokenId} ";
        public int CountTokens(string text) => 3;
    }

    /// <summary>
    /// The shipped path: <see cref="TextGenerator"/> in block mode (what <c>run</c> / <c>chat</c> / <c>serve</c> call) returns the same
    /// ids as <see cref="CloverBlockDiffusion.Generate"/> and makes the same causal commits.
    /// </summary>
    [Fact]
    public void TextGenerator_BlockMode_MatchesTheDecoder()
    {
        var reference = new ScriptedClover();
        using (var kv = Kv())
            _ = new CloverBlockDiffusion(reference).Generate([3, 4, 5], 10, kv, [Vocab + 100]);
        var refIds = reference.Commits.Skip(1).SelectMany(c => c.Tokens).ToList();

        var model = new ScriptedClover();
        var generator = new TextGenerator(model, new StubTokenizer()) { CloverDecode = CloverDecodeMode.Block };
        var response = generator.Generate("prompt", new InferenceOptions { MaxTokens = 10, Temperature = 0f });
        Assert.Equal(10, response.GeneratedTokenIds.Length);
        Assert.Equal(refIds, response.GeneratedTokenIds.Take(8));
        Assert.Equal(reference.Commits.Select(c => (c.Start, string.Join(",", c.Tokens))),
                     model.Commits.Select(c => (c.Start, string.Join(",", c.Tokens))));
        Assert.Equal(3, generator.LastCloverBlockStats!.Value.Blocks);
    }

    [Fact]
    public void TextGenerator_CausalByDefault()
    {
        var model = new ScriptedClover();
        var generator = new TextGenerator(model, new StubTokenizer());
        Assert.Equal(CloverDecodeMode.Causal, generator.CloverDecode);
    }

    /// <summary>Sampling with top-k 1 is greedy, whatever the temperature and seed.</summary>
    [Fact]
    public void Sampling_TopK1_EqualsGreedy()
    {
        using var kv1 = Kv();
        using var kv2 = Kv();
        var greedy = new CloverBlockDiffusion(new ScriptedClover()).Generate([3, 4, 5], 12, kv1, [Vocab + 100]);
        var sampled = new CloverBlockDiffusion(new ScriptedClover(), sampling: new CloverBlockSampling(1.3f, TopK: 1, Seed: 7))
            .Generate([3, 4, 5], 12, kv2, [Vocab + 100]);
        Assert.Equal(greedy.Response, sampled.Response);
    }

    [Fact]
    public void Coreset_KeepsTheExpertsCoveringTopP()
    {
        // Row 0 top-2 = {0, 1}, row 1 top-2 = {1, 2}; softmax masses 0.644 / 0.237 per row. Normalised totals: e1 .500, e0 .365,
        // e2 .134, e3 0. Exclusive cumulative mass: e1 0, e0 .500, e2 .865, e3 1.0.
        float[] logits = [2, 1, 0, -1, 0, 2, 1, -1];
        var at09 = (float[])logits.Clone();
        MoeSwiGluMlp.ApplyBlockCoreset(at09, rows: 2, numExperts: 4, topK: 2, topP: 0.9f);
        Assert.Equal(new float[] { 2, 1, 0, float.NegativeInfinity, 0, 2, 1, float.NegativeInfinity }, at09);
        var at08 = (float[])logits.Clone();
        MoeSwiGluMlp.ApplyBlockCoreset(at08, rows: 2, numExperts: 4, topK: 2, topP: 0.8f);
        Assert.Equal(new float[] { 2, 1, float.NegativeInfinity, float.NegativeInfinity, 0, 2, float.NegativeInfinity, float.NegativeInfinity }, at08);
    }

    [Fact]
    public void Route_WithACoresetSmallerThanTopK_FillsTheSpareSlotsWithZeroWeight()
    {
        const int E = 4, K = 2, H = 2, T = 2;
        float[] gate = [1, 0, 0, 1, -1, 0, 0, -1];   // expert e: logit = <gate_e, x>
        float[] hidden = [3, 0, 2.5f, 0];            // both rows prefer expert 0 by far
        int[] ae = new int[T * K]; float[] aw = new float[T * K];
        int[] cursors = new int[E + 1], bt = new int[T * K], bs = new int[T * K], unique = new int[E];
        MoeSwiGluMlp.Route(hidden, gate, ae, aw, cursors, bt, bs, unique, E, K, H, T, normTopKProb: true, blockCoresetTopP: 0.01f);
        for (int t = 0; t < T; t++)
        {
            Assert.Equal(0, ae[t * K]);
            Assert.Equal(1f, aw[t * K]);
            Assert.InRange(ae[t * K + 1], 0, E - 1);
            Assert.Equal(0f, aw[t * K + 1]);
        }
        // Off: the second slot is a real expert with a positive weight.
        MoeSwiGluMlp.Route(hidden, gate, ae, aw, cursors, bt, bs, unique, E, K, H, T, normTopKProb: true);
        Assert.True(aw[1] > 0f);
    }
}
