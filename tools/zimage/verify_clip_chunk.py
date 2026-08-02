#!/usr/bin/env python3
"""Check that the split, static Qwen3 chain reproduces `hidden_states[-2]`.

Runs on a tiny random-weight Qwen3 in a few seconds, so it can be run after any
transformers bump. Three things are being checked, and each has already been a
real bug in the DiT equivalent of this file:

  1. `usable_layers()` -- that `hidden_states[-2]` really is the output of layer
     N-1, so exporting N-1 layers is neither too few nor one too many.
  2. The additive causal + padding mask matches what the reference builds from
     a `[1, S]` attention mask.
  3. Splitting the layers across parts is lossless.

    python tools/zimage/verify_clip_chunk.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_clip import StaticQwen3Chunk, plan_parts, usable_layers  # noqa: E402

SEQ = 24
TRUE_LEN = 17          # the rest is right padding, as the app produces


def build(cfg_layers=6, dim=64):
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=dim, intermediate_size=2 * dim,
                      num_hidden_layers=cfg_layers, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, vocab_size=64,
                      max_position_embeddings=128)
    return cfg, Qwen3ForCausalLM(cfg).eval()


def reference(model, ids, mask):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask,
                     output_hidden_states=True).hidden_states[-2]


def chain(cfg, model, embeds, mask_f, cuts, dim):
    """Run the exported parts back to back, exactly as the app would."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    rot = Qwen3RotaryEmbedding(cfg)
    with torch.no_grad():
        cos, sin = rot(torch.zeros(1, SEQ, dim), torch.arange(SEQ).unsqueeze(0))
    h = embeds
    for pi, (a, b) in enumerate(cuts):
        part = StaticQwen3Chunk(model.model.layers[a:b], cos, sin,
                                first=(pi == 0), last=(pi == len(cuts) - 1),
                                seq=SEQ).eval()
        with torch.no_grad():
            h = part(h, mask_f)
    return h


def main():
    dim = 64
    cfg, model = build(dim=dim)
    n_usable = usable_layers(cfg.num_hidden_layers)
    print(f"{cfg.num_hidden_layers} layers, {n_usable} reachable from "
          f"hidden_states[-2]")

    ids = torch.randint(0, 64, (1, SEQ))
    mask = torch.zeros(1, SEQ, dtype=torch.long)
    mask[:, :TRUE_LEN] = 1
    ref = reference(model, ids, mask)

    with torch.no_grad():
        embeds = model.model.embed_tokens(ids)
    mask_f = mask.to(torch.float32)

    # Only the real-token rows are compared: the reference discards padded rows
    # (prompt_embeds[i][prompt_masks[i]]) and so does the app, so what a pad row
    # holds is unobservable.
    ok = True
    for n_parts in (1, 2, 3, n_usable):
        cuts = plan_parts(n_parts, n_usable)
        got = chain(cfg, model, embeds, mask_f, cuts, dim)
        d = (got[:, :TRUE_LEN] - ref[:, :TRUE_LEN]).abs().max().item()
        flag = "OK" if d < 2e-5 else "MISMATCH"
        if flag != "OK":
            ok = False
        print(f"  {n_parts:2d}-part chain {str(cuts):40.40} max abs {d:.3e}  {flag}")

    # A one-layer-too-many chain must NOT match, or the check above proves
    # nothing about usable_layers() being right.
    cuts = [(0, cfg.num_hidden_layers)]
    got = chain(cfg, model, embeds, mask_f, cuts, dim)
    d = (got[:, :TRUE_LEN] - ref[:, :TRUE_LEN]).abs().max().item()
    print(f"  control: all {cfg.num_hidden_layers} layers          "
          f"max abs {d:.3e}  {'differs, as it must' if d > 1e-4 else 'UNEXPECTED MATCH'}")
    if d <= 1e-4:
        ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
