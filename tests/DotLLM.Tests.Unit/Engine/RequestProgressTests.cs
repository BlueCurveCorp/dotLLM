using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;

using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// #1822: a CPU prefill runs as ONE forward pass, so per-chunk progress read <c>0/N</c> until it ended. The CPU layer loop now reports
/// through <see cref="ForwardProgress"/>, and the request's <see cref="RequestProgress"/> and the log pulse both move DURING the
/// forward. Observed from inside the real <see cref="TransformerModel"/> forward through its residual-modifier hook, which runs between
/// layers on the forward's own thread, so every recorded value is one the forward had published before it returned.
/// </summary>
[Collection("RequestTimingLogSink")]   // RequestTimingLog.Sink is process-global
public sealed class RequestProgressTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("dotllm-1822-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, true); } catch { }
    }

    /// <summary>Records the counters each time the forward passes a layer boundary.</summary>
    private sealed class Recorder(RequestProgress progress, Func<PrefillPulse?> pulse) : IResidualModifier
    {
        public List<(string Phase, int Done, int Total, int Pulse, int Decoded, int Tokens)> Seen { get; } = new();

        public bool AppliesTo(int layerIndex) => true;

        public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount) =>
            Seen.Add((progress.Phase, progress.PrefillTokensDone, progress.PrefillTokensTotal, pulse()?.Completed ?? -1, progress.DecodeTokens, tokenCount));
    }

    [Fact]
    public void CpuSingleForwardPrefill_PublishesProgressBetweenLayers_ThenCountsDecodedTokens()
    {
        string path = Path.Combine(_dir, "tiny.gguf");
        File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama", byteLevelVocab: true));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var progress = new RequestProgress("req#test");
        PrefillPulse? pulse = null;
        var recorder = new Recorder(progress, () => pulse);
        model.ResidualModifier = recorder;

        var previousSink = RequestTimingLog.Sink;
        var previousInterval = RequestTimingLog.PulseInterval;
        RequestTimingLog.Sink = static _ => { };                     // logging on, so the pulse exists; lines are not asserted
        RequestTimingLog.PulseInterval = Timeout.InfiniteTimeSpan;      // no timer: read the pulse's count directly
        RequestTimingLog.t_pulseCreated = p => pulse = p;
        RequestProgress.Current = progress;
        try
        {
            new TextGenerator(model, tokenizer)   // no chunk size: the whole prompt is one forward (the CPU default)
                .Generate(new string('a', 40), new InferenceOptions { MaxTokens = 4, Temperature = 0f });
        }
        finally
        {
            RequestProgress.Current = null;
            RequestTimingLog.t_pulseCreated = null;
            RequestTimingLog.Sink = previousSink;
            RequestTimingLog.PulseInterval = previousInterval;
        }

        var prefill = recorder.Seen.Where(s => s.Tokens > 1).ToList();
        Assert.NotEmpty(prefill);
        int total = prefill[0].Total;
        Assert.True(total >= 40, $"total {total}");
        Assert.True(prefill.All(s => s.Phase == "prefill"));

        // The forward has not returned, yet progress is strictly between 0 and the total: per-layer, not per-chunk.
        Assert.Contains(prefill, s => s.Done > 0 && s.Done < total);
        // ... monotonic, and the log pulse reads the very same count.
        Assert.Equal(prefill.Select(s => s.Done).Order(), prefill.Select(s => s.Done));
        Assert.All(prefill, s => Assert.Equal(s.Done, s.Pulse));

        // Decode: the count moves while the request is still generating (a decode forward sees the tokens decoded so far).
        var decode = recorder.Seen.Where(s => s.Tokens == 1).ToList();
        Assert.NotEmpty(decode);
        Assert.All(decode, s => Assert.Equal("decode", s.Phase));
        Assert.Contains(decode, s => s.Decoded >= 1);
        Assert.Equal(total, progress.PrefillTokensDone);
        Assert.True(progress.DecodeTokens is >= 1 and <= 4, $"decoded {progress.DecodeTokens}");
    }

    [Fact]
    public void ForwardProgress_WithNoTarget_IsANoOp_AndSwapRestores()
    {
        ForwardProgress.LayersCompleted(1, 2);   // nothing installed: must not throw
        var seen = new List<(int, int)>();
        var target = new Target(seen);
        var previous = ForwardProgress.Swap(target);
        try { ForwardProgress.LayersCompleted(1, 4); }
        finally { ForwardProgress.Swap(previous); }
        ForwardProgress.LayersCompleted(2, 4);   // after restore: not seen
        Assert.Equal([(1, 4)], seen);
    }

    private sealed class Target(List<(int, int)> seen) : IForwardProgressTarget
    {
        public void OnLayersCompleted(int completed, int total) => seen.Add((completed, total));
    }

    [Fact]
    public void RequestProgress_ClampsAndNeverMovesBackwards()
    {
        var p = new RequestProgress(null);
        Assert.Equal("waiting", p.Phase);
        p.BeginPrefill(promptTokens: 100, cachedTokens: 30);
        Assert.Equal(70, p.PrefillTokensTotal);
        p.SetPrefillDone(50);
        p.SetPrefillDone(20);
        Assert.Equal(50, p.PrefillTokensDone);
        p.SetPrefillDone(500);
        Assert.Equal(70, p.PrefillTokensDone);
        var generated = new List<int> { 1, 2 };
        p.BeginDecode(generated);
        generated.Add(3);
        Assert.Equal(("decode", 3), (p.Phase, p.DecodeTokens));
    }
}
