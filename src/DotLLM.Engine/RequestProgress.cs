using DotLLM.Core.Diagnostics;

namespace DotLLM.Engine;

/// <summary>
/// Live progress counters for one in-flight request (#1822): prompt size, prefix-cache hits, prefill tokens done / total, and tokens
/// decoded. The server creates one per request and makes it <see cref="Current"/> on the request's flow; the engine (the
/// <see cref="TextGenerator"/> prefill and decode loops, and the continuous-batch scheduler, which captures it at submission) writes
/// it; <c>/v1/status</c> reads it from another thread. Counters only: nothing is allocated per token or per chunk.
/// </summary>
/// <remarks>
/// Reads are deliberately racy-but-safe: every field is an <see cref="int"/> written with <see cref="Volatile"/>, and the decode count
/// is the length of the request's own generated-token list, read without a lock (a torn read is impossible for an <see cref="int"/>,
/// and a value one token stale is fine for a status page).
/// </remarks>
public sealed class RequestProgress
{
    private static readonly AsyncLocal<RequestProgress?> s_current = new();

    /// <summary>The progress of the request running on this flow, or null when nothing installed one (tests, CLI).</summary>
    public static RequestProgress? Current
    {
        get => s_current.Value;
        set => s_current.Value = value;
    }

    private int _phase;          // 0 = waiting, 1 = prefill, 2 = decode
    private int _promptTokens;
    private int _cachedTokens;
    private int _prefillTotal;
    private int _prefillDone;
    private List<int>? _generated;

    /// <summary>Creates the counters for one request.</summary>
    /// <param name="tag">The request's log tag (<c>req#12</c>), when it has one.</param>
    public RequestProgress(string? tag)
    {
        Tag = tag;
        StartedUtc = DateTimeOffset.UtcNow;
    }

    /// <summary>The request's log tag (<c>req#12</c>), when it has one.</summary>
    public string? Tag { get; }

    /// <summary>When the request started running (UTC).</summary>
    public DateTimeOffset StartedUtc { get; }

    /// <summary><c>waiting</c> (admitted, no prefill yet), <c>prefill</c>, or <c>decode</c>.</summary>
    public string Phase => Volatile.Read(ref _phase) switch { 1 => "prefill", 2 => "decode", _ => "waiting" };

    /// <summary>Tokens in the prompt, including prefix-cache hits; 0 before the prefill starts.</summary>
    public int PromptTokens => Volatile.Read(ref _promptTokens);

    /// <summary>Prompt tokens reused from the prefix cache (not computed).</summary>
    public int CachedTokens => Volatile.Read(ref _cachedTokens);

    /// <summary>Prompt tokens this prefill computes (<see cref="PromptTokens"/> minus <see cref="CachedTokens"/>).</summary>
    public int PrefillTokensTotal => Volatile.Read(ref _prefillTotal);

    /// <summary>Computed prompt tokens the device has finished so far (interpolated per layer inside a CPU forward).</summary>
    public int PrefillTokensDone => Volatile.Read(ref _prefillDone);

    /// <summary>Tokens decoded so far.</summary>
    public int DecodeTokens => Volatile.Read(ref _generated)?.Count ?? 0;

    /// <summary>Starts a prefill: <paramref name="promptTokens"/> in the prompt, <paramref name="cachedTokens"/> of them reused.</summary>
    /// <param name="promptTokens">Tokens in the prompt.</param>
    /// <param name="cachedTokens">Of those, the prefix-cache hits.</param>
    public void BeginPrefill(int promptTokens, int cachedTokens)
    {
        Volatile.Write(ref _promptTokens, promptTokens);
        Volatile.Write(ref _cachedTokens, cachedTokens);
        Volatile.Write(ref _prefillDone, 0);
        Volatile.Write(ref _prefillTotal, Math.Max(0, promptTokens - cachedTokens));
        Volatile.Write(ref _phase, 1);
    }

    /// <summary>Records that <paramref name="done"/> computed prompt tokens are finished (absolute, clamped to the total, never decreasing).</summary>
    /// <param name="done">Tokens finished so far in this prefill.</param>
    public void SetPrefillDone(int done)
    {
        int total = Volatile.Read(ref _prefillTotal);
        if (done > total) done = total;
        if (done > Volatile.Read(ref _prefillDone))
            Volatile.Write(ref _prefillDone, done);
    }

    /// <summary>Marks the prefill finished and the request decoding; <paramref name="generated"/> is the list its tokens are appended to.</summary>
    /// <param name="generated">The request's own generated-token list (read, never written, by status).</param>
    public void BeginDecode(List<int> generated)
    {
        Volatile.Write(ref _prefillDone, Volatile.Read(ref _prefillTotal));
        Volatile.Write(ref _generated, generated);
        Volatile.Write(ref _phase, 2);
    }
}

/// <summary>
/// One prefill's tracker (#1822): turns per-chunk completion and per-layer reports from a CPU forward
/// (<see cref="ForwardProgress"/>) into a monotonic "tokens done" for both the log pulse and the request's <see cref="RequestProgress"/>,
/// so the two can never disagree. Allocated once per prefill, and only when one of them is live.
/// </summary>
internal sealed class PrefillTracker : IForwardProgressTarget
{
    private readonly PrefillPulse? _pulse;
    private readonly RequestProgress? _progress;
    private int _base;
    private int _chunk;
    private int _done;

    /// <summary>Creates a tracker; returns null (no allocation) when there is nothing to report to.</summary>
    /// <param name="pulse">The log pulse, or null.</param>
    /// <param name="progress">The request's counters, or null.</param>
    /// <returns>The tracker, or null.</returns>
    public static PrefillTracker? Create(PrefillPulse? pulse, RequestProgress? progress) =>
        pulse is null && progress is null ? null : new PrefillTracker(pulse, progress);

    private PrefillTracker(PrefillPulse? pulse, RequestProgress? progress)
    {
        _pulse = pulse;
        _progress = progress;
    }

    /// <summary>A forward over the next <paramref name="tokens"/> prompt tokens is about to run.</summary>
    /// <param name="tokens">Tokens in the chunk.</param>
    public void BeginChunk(int tokens) => _chunk = tokens;

    /// <summary>The chunk's forward returned (the device finished it).</summary>
    public void EndChunk()
    {
        _base += _chunk;
        _chunk = 0;
        Publish(_base);
    }

    /// <inheritdoc/>
    public void OnLayersCompleted(int completed, int total)
    {
        if (total <= 0 || _chunk <= 0 || completed <= 0) return;
        if (completed > total) completed = total;
        Publish(_base + (int)((long)_chunk * completed / total));
    }

    private void Publish(int done)
    {
        if (done <= _done) return;   // monotonic: a nested forward (MTP head, draft) restarting its layer count must not move it back
        _done = done;
        _pulse?.Set(done);
        _progress?.SetPrefillDone(done);
    }
}
