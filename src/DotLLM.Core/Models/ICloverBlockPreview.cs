using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.Models;

/// <summary>
/// Clover's bidirectional block preview (#1815, stage 3 of #1794): the <c>preview=True</c> forward of DiffuSpace
/// <c>modeling_clover_moe.py</c> over one block of tokens (some of them <c>&lt;|mask|&gt;</c>) after a committed causal prefix.
/// </summary>
/// <remarks>
/// <para>
/// Inside the block every token sees every other: the GDN layers add a right conv and replace the delta-rule recurrence by a per-head
/// softmax over the block plus a read-only history bucket of the prefix's recurrent state; the full-attention layers attend the whole
/// prefix and the whole block. Logits are same-position: row <c>i</c> predicts the token AT block position <c>i</c>, and the
/// <c>&lt;|mask|&gt;</c> column is <c>-inf</c>.
/// </para>
/// <para>
/// <b>The committed state is not mutated.</b> The recurrent state (conv windows and GDN matrices) is only read. The block's K/V are written
/// to the KV cache rows <c>[startPosition, startPosition + T)</c> for the attention and the cache is rolled back to
/// <c>startPosition</c> before returning (also on failure), so rows below <c>startPosition</c> and <see cref="IKvCache.CurrentLength"/>
/// are unchanged; rows at and above it are scratch the next causal forward overwrites. A block decoder (stage 4) therefore calls this
/// repeatedly on the same committed state while denoising, then commits the resolved block with an ordinary causal forward.
/// </para>
/// <para>
/// The block expert coreset (<c>moe_block_coreset_enabled</c>, <see cref="CloverConfig.PreviewCoresetTopP"/>) is applied in every MoE
/// layer of the preview, as HF does (#1828): a router-logit mask before top-k that restricts the whole block to one expert subset covering
/// <c>top_p</c> of its aggregated top-k routing mass. The causal forward never applies it.
/// </para>
/// </remarks>
public interface ICloverBlockPreview
{
    /// <summary>Whether the loaded weights carry the bidirectional-preview tensors (an official DiffuSpace Clover GGUF).</summary>
    bool SupportsBlockPreview { get; }

    /// <summary>Runs the preview forward over one block.</summary>
    /// <param name="blockTokenIds">The block, <c>1..CloverConfig.BlockSize</c> tokens; masked positions hold <c>CloverConfig.MaskTokenId</c>.</param>
    /// <param name="startPosition">Position of the block's first token = the committed prefix length.</param>
    /// <param name="kvCache">The committed KV cache (<see cref="IKvCache.CurrentLength"/> must equal <paramref name="startPosition"/>), or
    /// <see langword="null"/> when <paramref name="startPosition"/> is 0.</param>
    /// <param name="gdnState">The committed recurrent state of the same backend (<see langword="null"/>: the model-owned one).</param>
    /// <param name="deviceId">Target device for the returned tensor.</param>
    /// <returns>A caller-owned <c>[T, VocabSize]</c> F32 logits tensor with the mask column at <c>-inf</c>.</returns>
    ITensor ForwardBlockPreview(ReadOnlySpan<int> blockTokenIds, int startPosition, IKvCache? kvCache, IGdnState? gdnState, int deviceId);
}
