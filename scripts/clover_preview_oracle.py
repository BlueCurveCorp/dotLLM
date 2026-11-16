#!/usr/bin/env python3
"""Synthetic tiny-Clover oracle for the bidirectional block preview (issue #1815, stage 3 of #1794).

Instantiates DiffuSpace's own HF modeling code (``modeling_clover_moe.py`` from ``Diffuspace/Clover-35B-A3B-Preview``, read from the
default HF hub cache; fetch the code files only with
``hf download Diffuspace/Clover-35B-A3B-Preview configuration_clover_moe.py modeling_clover_moe.py sampling_clover_moe.py``)
with a small random config in torch on CPU, F32 end to end, and writes:

* ``clover-preview-tiny.gguf``: the same weights in the OFFICIAL ``Clover-35B-A3B-Preview-GGUF`` tensor layout: arch ``clover_moe``,
  V heads reordered from HF's grouped order to llama.cpp's tiled order (every per-V-head tensor, the four bidirectional ones included;
  verified against the official Q4_K_M file), raw ``A_log`` under ``ssm_a``, zero-centred norms stored as ``1 + w``.
* ``oracle.json``: HF causal logits for a prefix, and HF ``preview=True`` logits for blocks after it.

Both go to ``~/.dotllm/test-cache/clover-preview-oracle/`` (override with ``--out``); nothing is written into the repository. The
dotLLM test ``CloverPreviewOracleTests`` reads them and skips when they are absent.

Run with a venv outside the repo that has torch (CPU), transformers>=5, safetensors, numpy and llama.cpp's ``gguf-py``:
    python -I scripts/clover_preview_oracle.py
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HF_REPO_DIR = "models--Diffuspace--Clover-35B-A3B-Preview"
CODE_FILES = ("configuration_clover_moe.py", "modeling_clover_moe.py", "sampling_clover_moe.py")

# Tiny but non-degenerate geometry: nK=2 / nV=4 makes the grouped (HF, vh // r) and tiled (GGUF, vh % nK) head mappings differ;
# dS=16 keeps the SIMD scan; 6 layers = [GDN, GDN, attn] x 2 so attention layers sit between GDN layers.
CFG = dict(
    hidden_size=64,
    num_hidden_layers=6,
    full_attention_interval=3,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    partial_rotary_factor=0.25,
    rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 10000.0, "rope_type": "default"},
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_shared_experts=1,
    vocab_size=512,
    mask_token_id=500,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    block_size=32,
    # Stage 3 implements the preview forward without the block expert coreset (a stage-4 decision, #1794); HF applies it in preview
    # whenever this flag is set, so the oracle turns it off.
    moe_block_coreset_enabled=False,
    attention_backend="eager",
    dtype="float32",
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
)


def find_hf_code() -> Path:
    roots = [os.environ.get("HF_HUB_CACHE"), os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
             os.path.expanduser("~/.cache/huggingface/hub")]
    for root in roots:
        if not root:
            continue
        for snap in sorted(glob.glob(os.path.join(root, HF_REPO_DIR, "snapshots", "*"))):
            if all(os.path.exists(os.path.join(snap, f)) for f in CODE_FILES):
                return Path(snap)
    sys.exit("Clover HF code not found in the HF hub cache; run: hf download Diffuspace/Clover-35B-A3B-Preview " + " ".join(CODE_FILES))


def import_hf(code_dir: Path, out: Path):
    # The modeling file uses package-relative imports: stage the three files as a package under the output dir.
    pkg = out / "_hf" / "clover_hf"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for f in CODE_FILES:
        shutil.copyfile(code_dir / f, pkg / f)
    sys.path.insert(0, str(pkg.parent))
    cfg_mod = importlib.import_module("clover_hf.configuration_clover_moe")
    mdl_mod = importlib.import_module("clover_hf.modeling_clover_moe")
    return cfg_mod.CloverMoeConfig, mdl_mod.CloverMoeModelLM


def randomise(model, c, gen: torch.Generator) -> None:
    """HF _init_weights leaves the preview parameters at their (identical-per-head) inits and right conv at zero: replace everything."""
    def n(*shape, std=1.0):
        return torch.randn(*shape, generator=gen) * std

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen)

    with torch.no_grad():
        m = model.model
        m.embed_tokens.weight.copy_(n(*m.embed_tokens.weight.shape))
        model.lm_head.weight.copy_(n(*model.lm_head.weight.shape, std=2.0 / math.sqrt(c.hidden_size)))
        m.norm.weight.copy_(n(c.hidden_size, std=0.1))
        for layer in m.layers:
            layer.input_layernorm.weight.copy_(n(c.hidden_size, std=0.1))
            layer.post_attention_layernorm.weight.copy_(n(c.hidden_size, std=0.1))
            if layer.preview_type == "linear_attention":
                g = layer.linear_attn
                for lin in (g.in_proj_qkv, g.in_proj_z, g.in_proj_a, g.in_proj_b, g.out_proj):
                    lin.weight.copy_(n(*lin.weight.shape, std=1.0 / math.sqrt(lin.weight.shape[1])))
                g.conv1d.weight.copy_(n(*g.conv1d.weight.shape, std=0.4))
                g.conv_right_weight.copy_(n(*g.conv_right_weight.shape, std=0.4))
                g.A_log.copy_(u(-2.0, 1.5, g.vh))
                g.dt_bias.copy_(n(g.vh, std=0.5))
                g.norm.weight.copy_(1.0 + n(g.vd, std=0.1))
                # Ranges of the official file (logit_scale 4.5..30, history_bias 3..9, history_c -1..1.4), distinct per head.
                g.bd_softmax_logit_scale.copy_(u(4.0, 25.0, g.vh))
                g.bd_softmax_history_bias.copy_(u(3.0, 8.0, g.vh))
                g.bd_softmax_history_c.copy_(u(-1.0, 1.3, g.vh))
            else:
                a = layer.self_attn
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    lin.weight.copy_(n(*lin.weight.shape, std=1.0 / math.sqrt(lin.weight.shape[1])))
                a.q_norm.weight.copy_(n(c.head_dim, std=0.1))
                a.k_norm.weight.copy_(n(c.head_dim, std=0.1))
            mlp = layer.mlp
            mlp.gate.weight.copy_(n(*mlp.gate.weight.shape, std=2.0 / math.sqrt(c.hidden_size)))
            mlp.experts.gate_up_proj.copy_(n(*mlp.experts.gate_up_proj.shape, std=1.0 / math.sqrt(c.hidden_size)))
            mlp.experts.down_proj.copy_(n(*mlp.experts.down_proj.shape, std=1.0 / math.sqrt(c.moe_intermediate_size)))
            se = mlp.shared_expert
            for lin in (se.gate_proj, se.up_proj, se.down_proj):
                lin.weight.copy_(n(*lin.weight.shape, std=1.0 / math.sqrt(lin.weight.shape[1])))
            mlp.shared_expert_gate.weight.copy_(n(*mlp.shared_expert_gate.weight.shape, std=1.0 / math.sqrt(c.hidden_size)))


def tiled(x: np.ndarray, num_k: int, num_v_per_k: int, head_dim: int, dim: int = 0) -> np.ndarray:
    """llama.cpp ``_reorder_v_heads``: grouped (by K head) -> tiled order along ``dim``."""
    x = np.moveaxis(x, dim, 0)
    shape = x.shape
    y = x.reshape(num_k, num_v_per_k, head_dim, *shape[1:])
    y = np.swapaxes(y, 0, 1).reshape(shape)
    return np.ascontiguousarray(np.moveaxis(y, 0, dim))


def write_gguf(model, c, path: Path) -> None:
    import gguf

    arch = "clover_moe"
    f32 = lambda t: np.ascontiguousarray(t.detach().float().numpy().astype(np.float32))
    nk, nv, kd, vd = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
    r = nv // nk
    qk = 2 * nk * kd

    w = gguf.GGUFWriter(str(path), arch)
    w.add_string("general.name", "clover-preview-tiny")
    w.add_uint32(f"{arch}.context_length", c.max_position_embeddings)
    w.add_uint32(f"{arch}.embedding_length", c.hidden_size)
    w.add_uint32(f"{arch}.block_count", c.num_hidden_layers)
    w.add_uint32(f"{arch}.attention.head_count", c.num_attention_heads)
    w.add_uint32(f"{arch}.attention.head_count_kv", c.num_key_value_heads)
    w.add_uint32(f"{arch}.attention.key_length", c.head_dim)
    w.add_uint32(f"{arch}.attention.value_length", c.head_dim)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", c.rms_norm_eps)
    w.add_uint32(f"{arch}.vocab_size", c.vocab_size)
    w.add_float32(f"{arch}.rope.freq_base", float(c.rope_parameters["rope_theta"]))
    w.add_uint32(f"{arch}.rope.dimension_count", int(c.head_dim * c.partial_rotary_factor))
    w.add_uint32(f"{arch}.full_attention_interval", c.full_attention_interval)
    w.add_uint32(f"{arch}.ssm.inner_size", nv * vd)
    w.add_uint32(f"{arch}.ssm.state_size", kd)
    w.add_uint32(f"{arch}.ssm.time_step_rank", nv)
    w.add_uint32(f"{arch}.ssm.group_count", nk)
    w.add_uint32(f"{arch}.ssm.conv_kernel", c.linear_conv_kernel_dim)
    w.add_uint32(f"{arch}.expert_count", c.num_experts)
    w.add_uint32(f"{arch}.expert_used_count", c.num_experts_per_tok)
    w.add_uint32(f"{arch}.expert_feed_forward_length", c.moe_intermediate_size)
    w.add_uint32(f"{arch}.expert_shared_feed_forward_length", c.shared_expert_intermediate_size)
    w.add_array(f"{arch}.attention.recurrent_layers", [t == "linear_attention" for t in c.layer_types])
    w.add_uint32(f"{arch}.block_size", c.block_size)
    w.add_bool(f"{arch}.moe_block_coreset_enabled", bool(c.moe_block_coreset_enabled))
    w.add_float32(f"{arch}.moe_block_coreset_top_p", float(c.moe_block_coreset_top_p))
    w.add_uint32("tokenizer.ggml.mask_token_id", c.mask_token_id)
    w.add_string("tokenizer.ggml.model", "llama")
    w.add_array("tokenizer.ggml.tokens", ["<unk>", "<bos>", "<eos>"] + [f"tok{i}" for i in range(3, c.vocab_size)])
    w.add_array("tokenizer.ggml.scores", [0.0] * c.vocab_size)
    w.add_array("tokenizer.ggml.token_type", [2, 3, 3] + [1] * (c.vocab_size - 3))
    w.add_uint32("tokenizer.ggml.bos_token_id", 1)
    w.add_uint32("tokenizer.ggml.eos_token_id", 2)
    w.add_uint32("tokenizer.ggml.unknown_token_id", 0)

    m = model.model
    w.add_tensor("token_embd.weight", f32(m.embed_tokens.weight))
    w.add_tensor("output.weight", f32(model.lm_head.weight))
    w.add_tensor("output_norm.weight", f32(1 + m.norm.weight))
    vt = lambda x, hd, dim=0: tiled(x, nk, r, hd, dim)
    for i, layer in enumerate(m.layers):
        p = f"blk.{i}"
        w.add_tensor(f"{p}.attn_norm.weight", f32(1 + layer.input_layernorm.weight))
        w.add_tensor(f"{p}.post_attention_norm.weight", f32(1 + layer.post_attention_layernorm.weight))
        if layer.preview_type == "linear_attention":
            g = layer.linear_attn
            qkv = f32(g.in_proj_qkv.weight)
            w.add_tensor(f"{p}.attn_qkv.weight", np.concatenate([qkv[:qk], vt(qkv[qk:], vd)], 0))
            w.add_tensor(f"{p}.attn_gate.weight", vt(f32(g.in_proj_z.weight), vd))
            w.add_tensor(f"{p}.ssm_alpha.weight", vt(f32(g.in_proj_a.weight), 1))
            w.add_tensor(f"{p}.ssm_beta.weight", vt(f32(g.in_proj_b.weight), 1))
            w.add_tensor(f"{p}.ssm_a", vt(f32(g.A_log), 1))   # official layout: the raw A_log, not -exp(A_log)
            w.add_tensor(f"{p}.ssm_dt.bias", vt(f32(g.dt_bias), 1))
            conv = f32(g.conv1d.weight)[:, 0, :]               # [C, dConv] -> ggml [dConv, C]
            w.add_tensor(f"{p}.ssm_conv1d.weight", np.concatenate([conv[:qk], vt(conv[qk:], vd)], 0))
            right = f32(g.conv_right_weight)[:, 0, :]          # [C, dConv-1] -> ggml [dConv-1, C]
            w.add_tensor(f"{p}.ssm_conv1d.weight_right", np.concatenate([right[:qk], vt(right[qk:], vd)], 0))
            w.add_tensor(f"{p}.ssm_bd_logit_scale.weight", vt(f32(g.bd_softmax_logit_scale), 1))
            w.add_tensor(f"{p}.ssm_bd_history.bias", vt(f32(g.bd_softmax_history_bias), 1))
            w.add_tensor(f"{p}.ssm_bd_history.weight", vt(f32(g.bd_softmax_history_c), 1))
            w.add_tensor(f"{p}.ssm_norm.weight", f32(g.norm.weight))   # not zero-centred: stored as is
            w.add_tensor(f"{p}.ssm_out.weight", vt(f32(g.out_proj.weight), vd, dim=1))
        else:
            a = layer.self_attn
            w.add_tensor(f"{p}.attn_q.weight", f32(a.q_proj.weight))   # per head [q(d) | gate(d)], as dotLLM de-interleaves
            w.add_tensor(f"{p}.attn_k.weight", f32(a.k_proj.weight))
            w.add_tensor(f"{p}.attn_v.weight", f32(a.v_proj.weight))
            w.add_tensor(f"{p}.attn_output.weight", f32(a.o_proj.weight))
            w.add_tensor(f"{p}.attn_q_norm.weight", f32(1 + a.q_norm.weight))
            w.add_tensor(f"{p}.attn_k_norm.weight", f32(1 + a.k_norm.weight))
        mlp = layer.mlp
        inter = c.moe_intermediate_size
        w.add_tensor(f"{p}.ffn_gate_inp.weight", f32(mlp.gate.weight))
        gu = f32(mlp.experts.gate_up_proj)                     # [E, 2I, H]: chunk(2, -1) of the output -> gate first
        w.add_tensor(f"{p}.ffn_gate_exps.weight", np.ascontiguousarray(gu[:, :inter, :]))
        w.add_tensor(f"{p}.ffn_up_exps.weight", np.ascontiguousarray(gu[:, inter:, :]))
        w.add_tensor(f"{p}.ffn_down_exps.weight", f32(mlp.experts.down_proj))
        se = mlp.shared_expert
        w.add_tensor(f"{p}.ffn_gate_shexp.weight", f32(se.gate_proj.weight))
        w.add_tensor(f"{p}.ffn_up_shexp.weight", f32(se.up_proj.weight))
        w.add_tensor(f"{p}.ffn_down_shexp.weight", f32(se.down_proj.weight))
        w.add_tensor(f"{p}.ffn_gate_inp_shexp.weight", f32(mlp.shared_expert_gate.weight).reshape(-1))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.expanduser("~/.dotllm/test-cache/clover-preview-oracle"))
    ap.add_argument("--seed", type=int, default=1815)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Config, ModelLM = import_hf(find_hf_code(), out)
    torch.set_grad_enabled(False)
    c = Config(**CFG)
    model = ModelLM(c).float().eval()
    gen = torch.Generator().manual_seed(args.seed)
    randomise(model, c, gen)
    write_gguf(model, c, out / "clover-preview-tiny.gguf")

    mask = c.mask_token_id
    rng = np.random.default_rng(args.seed)
    def ids(n):  # ordinary (non-mask, non-special) tokens
        return [int(x) for x in rng.integers(3, 480, size=n)]

    def preview(prefix, block):
        cache = model.model(torch.tensor([prefix]), use_cache=True).past_key_values if prefix else None
        logits = model(torch.tensor([block]), past_key_values=cache, use_cache=True, preview=True).logits[0].float().clone()
        logits[:, mask] = float("-inf")
        return logits

    def causal(seq):
        return model(torch.tensor([seq]), use_cache=False).logits[0].float()

    p12 = ids(12)
    cases = []
    specs = [
        ("prefix12_T8", p12, [mask, ids(1)[0], mask, mask, *ids(1), mask, mask, mask]),
        ("prefix12_T32", p12, [mask if (i % 3) else t for i, t in enumerate(ids(32))]),
        ("prefix12_T4_allmask", p12, [mask] * 4),
        ("prefix0_T8", [], [mask if i in (1, 4, 6) else t for i, t in enumerate(ids(8))]),
        ("prefix5_T16", ids(5), [mask if i % 2 else t for i, t in enumerate(ids(16))]),
    ]
    for name, prefix, block in specs:
        lg = preview(prefix, block)
        cases.append({"name": name, "prefix": prefix, "block": block,
                      "logits": [[(None if math.isinf(v) else round(v, 7)) for v in row] for row in lg.tolist()]})
        print(f"{name}: prefix {len(prefix)} block {len(block)} top1 {lg.argmax(-1).tolist()}")

    # Causal reference over prefix + one block (checks the writer before preview is compared at all).
    cseq = p12 + specs[0][2]
    causal_logits = causal(cseq)
    doc = {
        "generator": "scripts/clover_preview_oracle.py",
        "seed": args.seed,
        "hf_code": str(find_hf_code()),
        "torch": torch.__version__,
        "config": {k: v for k, v in CFG.items()},
        "mask_token_id": mask,
        "causal": {"tokens": cseq, "logits": [[round(v, 7) for v in row] for row in causal_logits.tolist()]},
        "cases": cases,
    }
    (out / "oracle.json").write_text(json.dumps(doc))
    print(f"wrote {out / 'clover-preview-tiny.gguf'} and {out / 'oracle.json'}")


if __name__ == "__main__":
    main()
