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


# ── Stage 4 (#1828): LowConfidence block-diffusion generation oracle ───────────────────────────────────────────────────────────────
# The generation fixture is the preview fixture's weights (same seed) with two changes that make the tests able to fail:
#  * lm_head scaled up, so top-1 confidences straddle the 0.95 threshold (with the stage-3 scale every step is a single fallback
#    fill, and a wrong-threshold mutant would be invisible);
#  * router gate scaled up, so the block expert coreset (on, top_p 0.9, as the official config) actually drops experts.
GEN_LM_SCALE = 8.0
GEN_ROUTER_SCALE = 3.0
GEN_THRESHOLD = 0.95
GEN_PENALTIES_OFF = dict(presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                         dllm_length_penalty=0.0, dllm_think_end_bias=0.0)


def mirror_generate(model, c, prompt, max_new, threshold, block_size, eos_ids):
    """HF ``CloverMoeModelLM.generate`` (greedy, penalties off) re-stated step by step so the per-step fill order can be recorded.
    The committed ids are asserted equal to HF's own ``generate`` by the caller."""
    mask = c.mask_token_id
    cache = model.model(torch.tensor([prompt]), use_cache=True).past_key_values
    response, blocks, stopped = [], [], False
    while len(response) < max_new and not stopped:
        remaining = min(block_size, max_new - len(response))
        block = torch.full((1, block_size), mask, dtype=torch.long)
        valid = torch.ones_like(block, dtype=torch.bool)
        pending = valid.clone()
        steps = []
        for _ in range(block_size):
            logits = model(block, attention_mask=valid, past_key_values=cache, use_cache=True, preview=True).logits[0].float()
            logits[:, mask] = -float("inf")
            prediction = logits.argmax(-1)
            confidence = logits.softmax(-1).gather(-1, prediction[:, None]).squeeze(-1)
            confidence = confidence.masked_fill(~pending[0], -float("inf"))
            transfer = (confidence > threshold) & pending[0]
            fallback = not bool(transfer.any())
            if fallback:
                transfer[confidence.argmax()] = True
            # Margins that decide whether a faithful re-implementation can flip: the threshold distance over the pending rows, the
            # top-1/top-2 logit gap of the rows filled, and (fallback) the gap between the best and second-best pending confidence.
            pend = pending[0]
            top2 = logits.topk(2, -1).values
            filled = transfer.nonzero().flatten().tolist()
            conf_p = confidence[pend]
            srt = conf_p.sort(descending=True).values
            steps.append({
                "filled": filled,
                "tokens": [int(prediction[i]) for i in filled],
                "fallback": fallback,
                "min_threshold_gap": float((conf_p - threshold).abs().min()),
                "min_logit_margin": float(min(top2[i, 0] - top2[i, 1] for i in filled)),
                "fallback_conf_gap": float(srt[0] - srt[1]) if fallback and srt.numel() > 1 else None,
            })
            block[0, transfer] = prediction[transfer]
            pending[0, transfer] = False
            if not bool(pending.any()):
                break
        assert not bool(pending.any())
        new = block[0, :remaining]
        committed = []
        for token in new.tolist():
            response.append(token)
            committed.append(token)
            if token in eos_ids:
                stopped = True
                break
        blocks.append({"block": block[0].tolist(), "committed": committed, "steps": steps})
        if len(response) < max_new and not stopped:
            cache = model.model(new.view(1, -1), past_key_values=cache, use_cache=True).past_key_values
    return response, blocks


def generate_oracle(Config, ModelLM, args, out: Path) -> None:
    import clover_hf.modeling_clover_moe as mdl

    cfg = dict(CFG, moe_block_coreset_enabled=True, moe_block_coreset_top_p=0.9,
               denoising_config={"algorithm": "low_confidence", "threshold": GEN_THRESHOLD})
    torch.set_grad_enabled(False)
    c = Config(**cfg)
    model = ModelLM(c).float().eval()
    gen = torch.Generator().manual_seed(args.seed)
    randomise(model, c, gen)
    with torch.no_grad():
        model.lm_head.weight.mul_(args.lm_scale)
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(args.router_scale)
    write_gguf(model, c, out / "clover-generate-tiny.gguf")

    # Coreset observability: how many experts each preview MoE call keeps (all 8 kept everywhere = an inert coreset).
    kept = []
    real_coreset = mdl.coreset_torch

    def counting_coreset(logits, valid, top_k, top_p):
        masked = real_coreset(logits, valid, top_k, top_p)
        kept.append(int(torch.isfinite(masked[0, 0]).sum()))
        return masked
    mdl.coreset_torch = counting_coreset

    def with_coreset(enabled, top_p, fn):
        c.moe_block_coreset_enabled, c.moe_block_coreset_top_p = enabled, top_p
        try:
            return fn()
        finally:
            c.moe_block_coreset_enabled, c.moe_block_coreset_top_p = True, 0.9

    rng = np.random.default_rng(args.seed + 1)
    def ids(n):
        return [int(x) for x in rng.integers(3, 480, size=n)]

    def trace(bs):
        return [[s["filled"] for s in b["steps"]] for b in bs]

    cases = []
    base_specs = [
        ("p10_n80", ids(10), 80, GEN_THRESHOLD, 32),
        ("p3_n70", ids(3), 70, GEN_THRESHOLD, 32),
        ("p17_n40_b8", ids(17), 40, GEN_THRESHOLD, 8),
    ]
    for name, prompt, max_new, threshold, block_size in base_specs:
        for eos_case in (False, True):
            eos = [c.eos_token_id]
            if eos_case:
                # A stop token that first appears mid-block (not in the first block) in the eos-free run, re-run as an extra EOS id.
                free, _ = mirror_generate(model, c, prompt, max_new, threshold, block_size, eos)
                cand = [t for i, t in enumerate(free) if i >= block_size and i % block_size not in (0, block_size - 1)
                        and t not in free[:i] and t != c.eos_token_id]
                if not cand:
                    continue
                eos = [c.eos_token_id, cand[0]]
            kept.clear()
            resp, blocks = mirror_generate(model, c, prompt, max_new, threshold, block_size, eos)
            kept_counts = list(kept)
            seq = model.generate(torch.tensor([prompt]), max_new_tokens=max_new, do_sample=False, threshold=threshold,
                                 block_size=block_size, eos_token_id=eos, pad_token_id=0, **GEN_PENALTIES_OFF)
            hf = seq[0, len(prompt):].tolist()
            assert hf == resp, f"{name}: mirror {resp} != HF generate {hf}"
            # Substitutions (the mutants, applied to the reference itself): each must change the committed ids or the fill trace.
            subs = {}
            for sub, enabled, top_p, th in (("coreset_off", False, 0.9, threshold), ("top_p_0.7", True, 0.7, threshold),
                                            ("threshold_0.9", True, 0.9, 0.9)):
                r2, b2 = with_coreset(enabled, top_p, lambda: mirror_generate(model, c, prompt, max_new, th, block_size, eos))
                subs[sub] = {"ids_differ": r2 != resp, "trace_differs": trace(b2) != trace(blocks)}
            steps = [s for b in blocks for s in b["steps"]]
            summary = {
                "steps": len(steps),
                "fallback_steps": sum(s["fallback"] for s in steps),
                "multi_fill_steps": sum(len(s["filled"]) > 1 for s in steps),
                "min_threshold_gap": min(s["min_threshold_gap"] for s in steps),
                "min_logit_margin": min(s["min_logit_margin"] for s in steps),
                "min_fallback_conf_gap": min((s["fallback_conf_gap"] for s in steps if s["fallback_conf_gap"] is not None), default=None),
                "coreset_kept_min": min(kept_counts), "coreset_kept_max": max(kept_counts),
                "coreset_kept_mean": sum(kept_counts) / len(kept_counts),
            }
            case = {"name": name + ("_eos" if eos_case else ""), "prompt": prompt, "max_new_tokens": max_new, "threshold": threshold,
                    "block_size": block_size, "eos_token_ids": eos, "response": resp, "blocks": blocks, "summary": summary,
                    "substitutions": subs}
            cases.append(case)
            print(f"{case['name']}: {len(resp)} tokens, {summary}\n    subs {subs}")

    # Coreset-on single previews (localize a coreset bug far better than end-to-end ids).
    previews = []
    mask = c.mask_token_id
    for pname, prefix, block in (("prefix10_T32_allmask", base_specs[0][1], [mask] * 32),
                                 ("prefix3_T32_half", base_specs[1][1], [mask if i % 2 else t for i, t in enumerate(ids(32))]),
                                 ("prefix17_T8", base_specs[2][1], [mask, *ids(2), mask, mask, *ids(1), mask, mask])):
        cache = model.model(torch.tensor([prefix]), use_cache=True).past_key_values
        run = lambda: model(torch.tensor([block]), past_key_values=cache, use_cache=True, preview=True).logits[0].float().clone()
        kept.clear()
        lg = run()
        kept_counts = list(kept)
        lg_off = with_coreset(False, 0.9, run)
        lg[:, mask] = float("-inf")
        lg_off[:, mask] = float("-inf")
        fin = torch.isfinite(lg)
        delta = float((lg[fin] - lg_off[fin]).abs().max())
        previews.append({"name": pname, "prefix": prefix, "block": block, "coreset_kept": kept_counts,
                         "coreset_off_max_abs_delta": delta,
                         "logits": [[(None if math.isinf(v) else round(v, 7)) for v in row] for row in lg.tolist()]})
        print(f"preview {pname}: kept {kept_counts}, |coreset on - off| max {delta:.3f}")
    mdl.coreset_torch = real_coreset

    doc = {
        "generator": "scripts/clover_preview_oracle.py --mode generate",
        "seed": args.seed, "lm_scale": args.lm_scale, "router_scale": args.router_scale,
        "hf_code": str(find_hf_code()), "torch": torch.__version__,
        "config": cfg,
        "mask_token_id": mask,
        "cases": cases,
        "previews": previews,
    }
    (out / "generate.json").write_text(json.dumps(doc))
    print(f"wrote {out / 'clover-generate-tiny.gguf'} and {out / 'generate.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.expanduser("~/.dotllm/test-cache/clover-preview-oracle"))
    ap.add_argument("--seed", type=int, default=1815)
    ap.add_argument("--mode", choices=("preview", "generate"), default="preview",
                    help="preview: the stage-3 preview oracle (#1815); generate: the stage-4 block-diffusion generation oracle (#1828)")
    ap.add_argument("--lm-scale", type=float, default=GEN_LM_SCALE)
    ap.add_argument("--router-scale", type=float, default=GEN_ROUTER_SCALE)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Config, ModelLM = import_hf(find_hf_code(), out)
    if args.mode == "generate":
        generate_oracle(Config, ModelLM, args, out)
        return
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
