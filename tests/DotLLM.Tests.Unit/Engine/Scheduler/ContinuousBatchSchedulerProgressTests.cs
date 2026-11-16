using DotLLM.Engine;
using DotLLM.Engine.Scheduler;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

public sealed partial class ContinuousBatchSchedulerTests
{
    /// <summary>
    /// #1822: the scheduler path (paged dense models, the common serving path) captures the submitter's <see cref="RequestProgress"/> at
    /// submission - which runs synchronously on the request's flow - and the run loop, on its own thread, fills it: prefill done/total,
    /// then the decode count step by step.
    /// </summary>
    [Fact]
    public async Task Progress_CapturedAtSubmission_IsFilledByTheRunLoop()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 3));
        var progress = new RequestProgress("req#sched");
        RequestProgress.Current = progress;
        ISchedulerRequest handle;
        try { handle = fix.Scheduler.Submit(MakeRequest(promptLen: 5, maxTokens: 16)); }
        finally { RequestProgress.Current = null; }   // the loop below runs with nothing current, like the real run loop

        Assert.Equal("waiting", progress.Phase);
        var decodeSeen = new List<int>();
        for (int i = 0; i < 1000 && !fix.Scheduler.IsIdle; i++)
        {
            fix.Scheduler.Step();
            if (progress.Phase == "decode") decodeSeen.Add(progress.DecodeTokens);
        }
        await handle.Completion;

        Assert.Equal(5, progress.PromptTokens);
        Assert.Equal(5, progress.PrefillTokensTotal);
        Assert.Equal(5, progress.PrefillTokensDone);
        Assert.Equal("decode", progress.Phase);
        Assert.True(decodeSeen.Count >= 2 && decodeSeen[0] < decodeSeen[^1], $"decode counts {string.Join(",", decodeSeen)}");
        Assert.True(progress.DecodeTokens >= 3);
    }
}
