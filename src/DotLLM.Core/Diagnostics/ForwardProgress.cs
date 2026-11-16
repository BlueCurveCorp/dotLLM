using System.Runtime.CompilerServices;

namespace DotLLM.Core.Diagnostics;

/// <summary>Receives intra-forward progress from a model's layer loop (#1822). See <see cref="ForwardProgress"/>.</summary>
public interface IForwardProgressTarget
{
    /// <summary>Called as a layer starts: <paramref name="completed"/> of <paramref name="total"/> layers of this forward have finished.</summary>
    /// <param name="completed">Layers completed so far in this forward (0 at the first layer).</param>
    /// <param name="total">Layers this forward runs.</param>
    void OnLayersCompleted(int completed, int total);
}

/// <summary>
/// Intra-forward progress (#1822): lets a long CPU prefill, which runs as ONE forward pass, report how far through its layers it is,
/// so <c>/v1/status</c> and the 3 s prefill log line move during the forward instead of jumping from 0 to 100 % at the end.
/// </summary>
/// <remarks>
/// <para>The target is per thread: the caller installs it around a synchronous <c>Forward</c>, and the model's layer loop (which
/// runs on the calling thread; per-op work fans out to the pool, the loop itself does not) reports into it. With no target installed
/// a report is one thread-static read and a null check, so the decode hot path pays nothing measurable.</para>
/// <para>Only <b>CPU</b> layer loops report. A GPU forward records layers before the device executes them, so a per-layer count there
/// would be queued work, not completed work.</para>
/// </remarks>
public static class ForwardProgress
{
    [ThreadStatic] private static IForwardProgressTarget? t_target;

    /// <summary>Reports that <paramref name="completed"/> of <paramref name="total"/> layers have finished. Free when no target is installed.</summary>
    /// <param name="completed">Layers completed so far in this forward.</param>
    /// <param name="total">Layers this forward runs.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LayersCompleted(int completed, int total)
    {
        if (t_target is { } target)
            target.OnLayersCompleted(completed, total);
    }

    /// <summary>Installs <paramref name="target"/> for the calling thread and returns the previous one (restore it in a <c>finally</c>).</summary>
    /// <param name="target">The new target, or null to remove it.</param>
    /// <returns>The target that was installed before.</returns>
    public static IForwardProgressTarget? Swap(IForwardProgressTarget? target)
    {
        var previous = t_target;
        t_target = target;
        return previous;
    }
}
