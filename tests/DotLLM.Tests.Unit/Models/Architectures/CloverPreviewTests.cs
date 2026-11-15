using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using CloverLayout = DotLLM.Models.Gguf.SyntheticQwen35MoeGguf.CloverLayout;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Self-contained (CI) contract tests of the Clover block preview (#1815) on the synthetic official-layout fixture. Numerical parity with
/// HF is <see cref="CloverPreviewOracleTests"/> (needs the generated oracle); these pin the API contract: the committed state is left
/// exactly as found, the block is bidirectional, and files without the preview tensors refuse.
/// </summary>
public sealed class CloverPreviewTests : IDisposable
{
    private static readonly int[] Prefix = [1, 3, 0, 2, 5, 4, 6, 3];
    private static readonly int[] Block = [7, 3, 7, 7, 5, 7];   // 7 = <|mask|> in the fixture
    private static readonly int[] Continuation = [4, 2, 6];
    private readonly string _scratch;

    public CloverPreviewTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-clover-preview-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private (GgufFile Gguf, Qwen3MoeHybridTransformerModel Model) Load(CloverLayout layout)
    {
        string path = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, $"{layout}.gguf"), contextLength: 64, blockCount: 4, clover: layout);
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = Assert.IsType<Qwen3MoeHybridTransformerModel>(ModelLoader.CreateCpuModelFromGguf(gguf, config));
        return (gguf, model);
    }

    private static SimpleKvCache NewKv(Qwen3MoeHybridTransformerModel m) =>
        new(m.AttentionLayerCount, m.Config.NumKvHeads, m.Config.HeadDim, m.Config.MaxSequenceLength);

    private static float[] Read(ITensor t)
    {
        var a = new float[t.Shape.ElementCount];
        unsafe { new ReadOnlySpan<float>((void*)t.DataPointer, a.Length).CopyTo(a); }
        return a;
    }

    private static int[] Range(int start, int n) => Enumerable.Range(start, n).ToArray();

    private static float[] Snapshot(GdnStateCache s)
    {
        var all = new List<float>();
        for (int i = 0; i < s.NumGdnLayers; i++) { all.AddRange(s.GetConvState(i).ToArray()); all.AddRange(s.GetGdnState(i).ToArray()); }
        return all.ToArray();
    }

    private static float[] KvRows(SimpleKvCache kv, int layer, int rows, int stride)
    {
        var a = new float[2 * rows * stride];
        unsafe
        {
            new ReadOnlySpan<float>((void*)kv.GetKeysRef(layer).DataPointer, rows * stride).CopyTo(a);
            new ReadOnlySpan<float>((void*)kv.GetValuesRef(layer).DataPointer, rows * stride).CopyTo(a.AsSpan(rows * stride));
        }
        return a;
    }

    /// <summary>
    /// The stage-4 contract: a preview reads the committed state and leaves it bit-identical (GDN conv + matrix state, KV rows below the
    /// prefix, KV length), so repeated previews agree bit for bit and a causal continuation after previews equals one without them.
    /// </summary>
    [Fact]
    public void Preview_LeavesCommittedStateUnchanged()
    {
        var (gguf, model) = Load(CloverLayout.Official);
        using var _ = gguf;
        using var __ = model;
        int stride = model.Config.NumKvHeads * model.Config.HeadDim;

        using var kv = NewKv(model);
        using var gdn = (GdnStateCache)model.CreateSequenceState()!;
        model.Forward(Prefix, Range(0, Prefix.Length), -1, kv, gdn).Dispose();
        float[] stateBefore = Snapshot(gdn);
        var kvBefore = Enumerable.Range(0, model.AttentionLayerCount).Select(l => KvRows(kv, l, Prefix.Length, stride)).ToArray();

        float[] first, second;
        using (var p = model.ForwardBlockPreview(Block, Prefix.Length, kv, gdn, -1)) first = Read(p);
        using (var p = model.ForwardBlockPreview(Block, Prefix.Length, kv, gdn, -1)) second = Read(p);

        Assert.Equal(first, second);
        Assert.Equal(Prefix.Length, kv.CurrentLength);
        Assert.Equal(stateBefore, Snapshot(gdn));
        for (int l = 0; l < model.AttentionLayerCount; l++)
            Assert.Equal(kvBefore[l], KvRows(kv, l, Prefix.Length, stride));
        int mask = model.Config.Clover!.MaskTokenId;
        for (int r = 0; r < Block.Length; r++)
            Assert.Equal(float.NegativeInfinity, first[r * model.Config.VocabSize + mask]);

        using var after = model.Forward(Continuation, Range(Prefix.Length, Continuation.Length), -1, kv, gdn);
        using var kv2 = NewKv(model);
        using var gdn2 = (GdnStateCache)model.CreateSequenceState()!;
        model.Forward(Prefix, Range(0, Prefix.Length), -1, kv2, gdn2).Dispose();
        using var plain = model.Forward(Continuation, Range(Prefix.Length, Continuation.Length), -1, kv2, gdn2);
        Assert.Equal(Read(plain), Read(after));
    }

    /// <summary>
    /// The block is bidirectional and reads the prefix: changing the LAST block token moves the FIRST row's logits (a causal block could
    /// not), and changing the prefix moves every row. Also a negative control for the parity test: preview logits differ from causal ones.
    /// </summary>
    [Fact]
    public void Preview_IsBidirectionalAndReadsThePrefix()
    {
        var (gguf, model) = Load(CloverLayout.Official);
        using var _ = gguf;
        using var __ = model;
        int vocab = model.Config.VocabSize;

        float[] Run(int[] prefix, int[] block)
        {
            using var kvC = NewKv(model);
            using var gdnC = (GdnStateCache)model.CreateSequenceState()!;
            model.Forward(prefix, Range(0, prefix.Length), -1, kvC, gdnC).Dispose();
            using var t = model.ForwardBlockPreview(block, prefix.Length, kvC, gdnC, -1);
            return Read(t);
        }

        static float RowDiff(float[] a, float[] b, int row, int vocab)
        {
            float m = 0f;
            for (int i = 0; i < vocab; i++)
            {
                float x = a[row * vocab + i], y = b[row * vocab + i];
                if (float.IsFinite(x) && float.IsFinite(y)) m = MathF.Max(m, MathF.Abs(x - y));
            }
            return m;
        }

        float[] baseLogits = Run(Prefix, Block);
        int[] changedTail = (int[])Block.Clone();
        changedTail[^1] = 2;
        Assert.True(RowDiff(baseLogits, Run(Prefix, changedTail), 0, vocab) > 1e-5f, "row 0 ignores a later block token: the block is not bidirectional");

        int[] changedPrefix = (int[])Prefix.Clone();
        changedPrefix[0] = 6;
        float[] p2 = Run(changedPrefix, Block);
        for (int r = 0; r < Block.Length; r++)
            Assert.True(RowDiff(baseLogits, p2, r, vocab) > 1e-6f, $"row {r} does not read the prefix");

        using var kv = NewKv(model);
        using var gdn = (GdnStateCache)model.CreateSequenceState()!;
        model.Forward(Prefix, Range(0, Prefix.Length), -1, kv, gdn).Dispose();
        using var causal = model.Forward(Block, Range(Prefix.Length, Block.Length), -1, kv, gdn);
        Assert.True(RowDiff(baseLogits, Read(causal), 0, vocab) > 1e-4f, "preview logits equal the causal ones");
    }

    [Fact]
    public void Preview_WithoutPreviewTensors_IsUnsupported()
    {
        var (gguf, model) = Load(CloverLayout.LlamaCppConverter);
        using var _ = gguf;
        using var __ = model;
        Assert.False(model.SupportsBlockPreview);
        Assert.Throws<NotSupportedException>(() => model.ForwardBlockPreview(Block, 0, null, null, -1));
    }

    [Fact]
    public void Preview_RejectsAStartThatIsNotTheCommittedLength()
    {
        var (gguf, model) = Load(CloverLayout.Official);
        using var _ = gguf;
        using var __ = model;
        Assert.True(model.SupportsBlockPreview);
        using var kv = NewKv(model);
        model.Forward(Prefix, Range(0, Prefix.Length), -1, kv).Dispose();
        Assert.Throws<ArgumentException>(() => model.ForwardBlockPreview(Block, Prefix.Length - 1, kv, null, -1));
        Assert.Throws<ArgumentException>(() => model.ForwardBlockPreview(new int[33], Prefix.Length, kv, null, -1));
    }
}
