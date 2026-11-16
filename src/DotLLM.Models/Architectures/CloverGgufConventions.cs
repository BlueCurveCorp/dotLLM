using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Load-time tensor conventions of DiffuSpace Clover GGUFs (<c>clover_moe</c>, issue #1794), shared by every Qwen3MoeHybrid host
/// (CPU, and through it Vulkan; CUDA loads its own copy).
/// </summary>
/// <remarks>
/// <para>
/// <b><c>ssm_a</c> holds the raw HF <c>A_log</c>.</b> qwen35moe GGUFs store <c>ssm_a = -exp(A_log)</c> (strictly negative; the
/// GDN gate is <c>g = ssm_a * softplus(alpha + dt_bias)</c>). The official <c>Diffuspace/Clover-35B-A3B-Preview-GGUF</c> files were
/// written by DiffuSpace's own converter, which stores <c>A_log</c> itself under the <c>ssm_a</c> name: layer 0's values span
/// -3.98..+4.28, and Qwen3.6-35B-A3B's <c>ssm_a</c> is exactly <c>-exp</c> of them (-exp(-3.3125) = -0.03643 vs -0.036425). Read as
/// qwen35moe, the model emits <c>GGGG…</c>. llama.cpp PR #31306's converter (<c>conversion/clover.py</c>) instead writes the qwen35moe
/// form and drops the bidirectional tensors, so both layouts exist.
/// </para>
/// <para>
/// Detection, per GDN layer: <c>-exp(x)</c> is strictly negative, so any positive entry proves <c>A_log</c>; the presence of the
/// DiffuSpace-only <c>ssm_conv1d.weight_right</c> tensor identifies the converter that writes <c>A_log</c> even if every entry
/// happened to be negative. A file with neither is the llama.cpp-converter form and is left alone.
/// </para>
/// </remarks>
public static class CloverGgufConventions
{
    /// <summary>The four bidirectional-preview GDN tensor suffixes: read by the block preview forward (#1815), never by the causal one.
    /// Order matters to <see cref="LoadPreviewWeights"/>.</summary>
    public static readonly string[] BidirectionalPreviewSuffixes =
    [
        "ssm_conv1d.weight_right",   // HF conv_right_weight            [3, convDim]
        "ssm_bd_logit_scale.weight", // HF bd_softmax_logit_scale       [nVHead]
        "ssm_bd_history.bias",       // HF bd_softmax_history_bias      [nVHead]
        "ssm_bd_history.weight",     // HF bd_softmax_history_c         [nVHead]
    ];

    /// <summary>
    /// Whether <paramref name="a"/> (a GDN layer's <c>ssm_a</c>, read verbatim) is the raw <c>A_log</c> that must be turned into
    /// <c>-exp(A_log)</c> before use. Always <see langword="false"/> for a non-Clover model.
    /// </summary>
    /// <param name="config">Model configuration (Clover iff <see cref="ModelConfig.Clover"/> is non-null).</param>
    /// <param name="tensors">The GGUF tensor directory.</param>
    /// <param name="layerPrefix">The layer prefix, e.g. <c>blk.0</c>.</param>
    /// <param name="a">The layer's <c>ssm_a</c> values as stored.</param>
    public static bool SsmAIsALog(ModelConfig config, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
                                  string layerPrefix, ReadOnlySpan<float> a)
    {
        if (config.Clover is null)
            return false;
        if (tensors.ContainsKey($"{layerPrefix}.ssm_conv1d.weight_right"))
            return true;
        foreach (float v in a)
            if (v > 0f)
                return true;
        return false;
    }

    /// <summary>In place: <c>a[i] = -exp(a[i])</c> (HF <c>-A_log.float().exp()</c>), the qwen35moe <c>ssm_a</c> form.</summary>
    /// <param name="a">The per-head <c>A_log</c> values; overwritten.</param>
    public static void ALogToSsmA(Span<float> a)
    {
        for (int i = 0; i < a.Length; i++)
            a[i] = -MathF.Exp(a[i]);
    }

    /// <summary>
    /// Reads one GDN layer's four bidirectional-preview tensors (#1815) as F32, or <see langword="null"/> when the model is not Clover or
    /// the file does not carry them (llama.cpp PR #31306's converter drops them). A partial set throws: the preview forward needs all four.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="tensors">The GGUF tensor directory.</param>
    /// <param name="layerPrefix">The layer prefix, e.g. <c>blk.0</c>.</param>
    /// <param name="convDim">GDN conv channels, <c>(2*NKHead + NVHead) * DState</c>.</param>
    /// <param name="read">Dequantises a tensor to F32 (<c>n</c> elements).</param>
    internal static CloverPreviewWeights? LoadPreviewWeights(ModelConfig config, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
                                                            string layerPrefix, int convDim, Func<GgufTensorDescriptor, int, float[]> read)
    {
        if (config.Clover is null)
            return null;
        var found = new GgufTensorDescriptor?[BidirectionalPreviewSuffixes.Length];
        int present = 0;
        for (int s = 0; s < found.Length; s++)
            if (tensors.TryGetValue($"{layerPrefix}.{BidirectionalPreviewSuffixes[s]}", out var d)) { found[s] = d; present++; }
        if (present == 0)
            return null;
        if (present != found.Length)
            throw new InvalidDataException($"{layerPrefix}: Clover file carries {present} of the {found.Length} bidirectional-preview tensors.");

        var gdn = config.GdnConfig!.Value;
        int nV = gdn.NVHead, taps = gdn.DConv - 1;
        if (found[0]!.Value.Shape.ElementCount != (long)taps * convDim)
            throw new InvalidDataException(
                $"{layerPrefix}.ssm_conv1d.weight_right has {found[0]!.Value.Shape.ElementCount} elements, expected (DConv-1)*convDim = {taps * convDim}.");
        for (int s = 1; s < found.Length; s++)
            if (found[s]!.Value.Shape.ElementCount != nV)
                throw new InvalidDataException(
                    $"{layerPrefix}.{BidirectionalPreviewSuffixes[s]} has {found[s]!.Value.Shape.ElementCount} elements, expected NVHead = {nV}.");
        return new CloverPreviewWeights
        {
            ConvRightWeight = read(found[0]!.Value, taps * convDim),
            LogitScale = read(found[1]!.Value, nV),
            HistoryBias = read(found[2]!.Value, nV),
            HistoryC = read(found[3]!.Value, nV),
        };
    }

    /// <summary>
    /// Writes one load-log line naming the Clover tensors the causal forward does not read (the bidirectional block preview, #1815, does).
    /// No-op for non-Clover models or files without them.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="tensors">The GGUF tensor directory.</param>
    /// <param name="host">Host name for the log line.</param>
    public static void LogUnconsumedTensors(ModelConfig config, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string host)
    {
        if (config.Clover is null)
            return;
        var counts = new int[BidirectionalPreviewSuffixes.Length];
        long bytes = 0;
        foreach (var (name, desc) in tensors)
        {
            for (int s = 0; s < BidirectionalPreviewSuffixes.Length; s++)
            {
                if (name.EndsWith("." + BidirectionalPreviewSuffixes[s], StringComparison.Ordinal))
                {
                    counts[s]++;
                    bytes += desc.Shape.ElementCount * sizeof(float);
                }
            }
        }
        int total = counts.Sum();
        if (total == 0)
            return;
        var parts = BidirectionalPreviewSuffixes.Select((sfx, i) => $"{sfx} x{counts[i]}");
        Console.Error.WriteLine(
            $"[clover] {host}: {total} bidirectional block-preview tensors ({string.Join(", ", parts)}; ~{bytes / 1024} KiB) are read "
            + "only by the block preview forward (#1815); the causal forward ignores them. Block-diffusion decoding (#1828) is opt-in: DOTLLM_CLOVER_DECODE=block.");
    }
}
