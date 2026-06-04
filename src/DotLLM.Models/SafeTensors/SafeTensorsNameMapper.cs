using System.Text.RegularExpressions;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Maps standard Hugging Face tensor names (used in SafeTensors) to DotLLM's internal GGUF-based naming scheme.
/// </summary>
public static partial class SafeTensorsNameMapper
{
    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")]
    private static partial Regex QProjRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$")]
    private static partial Regex KProjRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$")]
    private static partial Regex VProjRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")]
    private static partial Regex OProjRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$")]
    private static partial Regex GateProjRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$")]
    private static partial Regex UpProjRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$")]
    private static partial Regex DownProjRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.input_layernorm\.weight$")]
    private static partial Regex InputNormRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$")]
    private static partial Regex PostAttnNormRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$")]
    private static partial Regex QProjBiasRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$")]
    private static partial Regex KProjBiasRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$")]
    private static partial Regex VProjBiasRegex();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.self_attn\.o_proj\.bias$")]
    private static partial Regex OProjBiasRegex();

    /// <summary>
    /// Translates a Hugging Face SafeTensors name to the standard DotLLM abbreviated name.
    /// Returns the original name if no mapping is found.
    /// </summary>
    public static string MapToDotLlmName(string hfName)
    {
        // Embeddings
        if (hfName == "model.embed_tokens.weight") return "token_embd.weight";

        // Outputs
        if (hfName == "model.norm.weight") return "output_norm.weight";
        if (hfName == "lm_head.weight") return "output.weight";

        // Block mappings
        Match m;

        m = QProjRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_q.weight";

        m = KProjRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_k.weight";

        m = VProjRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_v.weight";

        m = OProjRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_output.weight";

        m = GateProjRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.ffn_gate.weight";

        m = UpProjRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.ffn_up.weight";

        m = DownProjRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.ffn_down.weight";

        m = InputNormRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_norm.weight";

        m = PostAttnNormRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.ffn_norm.weight";

        // Biases
        m = QProjBiasRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_q.bias";

        m = KProjBiasRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_k.bias";

        m = VProjBiasRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_v.bias";

        m = OProjBiasRegex().Match(hfName);
        if (m.Success) return $"blk.{m.Groups[1].Value}.attn_output.bias";

        return hfName; // Fallback
    }
}
