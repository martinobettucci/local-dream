#!/usr/bin/env python3
"""Export the Qwen3-4B text encoder to per-part ONNX graphs, plus token_emb.bin.

Three facts shape this, all of them measured rather than assumed:

  * The reference reads `hidden_states[-2]`, which is the output of layer
    **N-1** -- the last decoder layer and the final RMSNorm are never
    evaluated. So 35 of Qwen3's 36 layers are exported, not 36.
  * The token embedding stays out of the graph. The app looks tokens up on the
    CPU against `token_emb.bin` so prompt weighting can scale individual rows,
    exactly as the Anima path does.
  * 35 layers is ~3.5 B parameters, ~14 GB of fp32. That cannot be held for
    export, let alone quantized, so the encoder is a chain like the DiT. It is
    far cheaper to split than the DiT: it runs once per generation and the
    result is prompt-cached, so the extra context switches are amortised over a
    whole image instead of paid on every step.

    python tools/zimage/export_clip.py --work /scratch --parts 6
    python tools/zimage/export_clip.py --work /scratch --token-emb

Weights are read over HTTP ranges (remote_safetensors.py), so no shard is ever
downloaded. Resumable: any part whose ONNX exists is skipped.
"""
import argparse
import gc
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "Tongyi-MAI/Z-Image-Turbo"
SUBDIR = "text_encoder"
SEQ = 512               # zimage_text_seq_len
DIM = 2560              # zimage_text_embedding_size

# Masked positions get a large negative rather than -inf: activations are
# quantized to 16 bits, and -inf survives neither the encoding search nor a
# fully-masked softmax row.
NEG = -1e4


def log(msg):
    print(f"[export_clip] {msg}", flush=True)


def usable_layers(n_hidden_layers):
    """How many decoder layers `hidden_states[-2]` actually depends on.

    hidden_states has n+1 entries: embeddings, then the output of layers 1..n-1,
    then norm(output of layer n). The last layer's raw output never appears. So
    [-2] is layer n-1's output and layer n is dead. Verified by replaying every
    stage of a small random Qwen3 and matching -- see docs/zimage.md.
    """
    return n_hidden_layers - 1


def plan_parts(n_parts, n_layers):
    base, extra = divmod(n_layers, n_parts)
    cuts, at = [], 0
    for i in range(n_parts):
        take = base + (1 if i < extra else 0)
        cuts.append((at, at + take))
        at += take
    return cuts


class StaticQwen3Chunk(nn.Module):
    """A contiguous run of Qwen3 decoder layers at a fixed [1, SEQ, DIM].

    Rotary tables are precomputed for positions 0..SEQ-1 and baked in as
    buffers: the app always pads right to SEQ, so positions are compile-time
    constants and there is nothing to feed at runtime. The attention mask is
    still an input, because it is the only thing that varies with the prompt.
    """

    def __init__(self, layers, cos, sin, first, last, seq=SEQ):
        super().__init__()
        self.layers = layers
        self.first, self.last, self.seq = bool(first), bool(last), int(seq)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("pos", torch.arange(seq).unsqueeze(0), persistent=False)
        # Causal part of the mask is constant; only the padding part is an input.
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        self.register_buffer("causal", causal.view(1, 1, seq, seq), persistent=False)

    def forward(self, hidden, attention_mask):
        # attention_mask is [1, SEQ], 1.0 real / 0.0 pad. Broadcast it over query
        # rows and combine with the causal triangle into one additive mask.
        keep = self.causal & (attention_mask.view(1, 1, 1, self.seq) > 0.5)
        mask = torch.where(keep, torch.zeros((), dtype=hidden.dtype),
                           torch.full((), NEG, dtype=hidden.dtype))
        pe = (self.cos, self.sin)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=mask, position_ids=self.pos,
                           position_embeddings=pe)
        return hidden

    @property
    def input_names(self):
        # "hidden_in" rather than "hidden" for the same reason the DiT parts use
        # it: ONNX cannot have an input and an output of the same name, and
        # torch renames the input silently instead of failing.
        return ["input_embedding" if self.first else "hidden_in", "attention_mask"]

    @property
    def output_names(self):
        return ["context"] if self.last else ["hidden"]

    def example_inputs(self):
        return (torch.randn(1, self.seq, DIM),
                torch.ones(1, self.seq, dtype=torch.float32))


def load_config():
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3Config

    path = hf_hub_download(REPO, f"{SUBDIR}/config.json")
    return Qwen3Config(**json.load(open(path)))


def build_chunk(cfg, readers, weight_map, a, b, first, last):
    """Instantiate only this part's decoder layers and load only their weights."""
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer, Qwen3RotaryEmbedding)

    from remote_safetensors import RemoteSafetensors  # noqa: F401  (typing only)

    # layer_idx is used for cache bookkeeping we never exercise, but keep the
    # real index so anything that logs it stays honest.
    layers = nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(a, b)])

    wanted = {}
    for i in range(a, b):
        for name, shard in weight_map.items():
            if name.startswith(f"model.layers.{i}."):
                local = f"{i - a}." + name[len(f"model.layers.{i}."):]
                wanted.setdefault(shard, {})[name] = local
    sd = {}
    for shard, names in wanted.items():
        got = readers[shard].get_tensors(list(names), dtype=torch.float32)
        for ckpt, tensor in got.items():
            sd[names[ckpt]] = tensor
        del got
        gc.collect()
    missing, unexpected = layers.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    if missing or unexpected:
        raise RuntimeError(f"layers {a}..{b}: missing={missing[:4]} "
                           f"unexpected={unexpected[:4]}")
    del sd
    gc.collect()

    rot = Qwen3RotaryEmbedding(cfg)
    with torch.no_grad():
        cos, sin = rot(torch.zeros(1, SEQ, cfg.hidden_size),
                       torch.arange(SEQ).unsqueeze(0))
    chunk = StaticQwen3Chunk(layers, cos, sin, first, last).eval()
    return chunk


def dump_token_emb(work, readers, weight_map):
    """fp16 [vocab, hidden] row-major, exactly what the app mmaps."""
    out = os.path.join(work, "token_emb.bin")
    if os.path.exists(out):
        log(f"token_emb.bin already present ({os.path.getsize(out) / 1e6:.0f} MB)")
        return out
    name = "model.embed_tokens.weight"
    shard = weight_map[name]
    log(f"fetching {name} from {shard}")
    t = readers[shard].get_tensors([name], dtype=torch.float16)[name]
    log(f"  {tuple(t.shape)} fp16 -> {out}")
    t.numpy().tofile(out)
    del t
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--token-emb", action="store_true",
                    help="dump token_emb.bin and exit")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    cfg = load_config()
    n_layers = usable_layers(cfg.num_hidden_layers)
    cuts = plan_parts(args.parts, n_layers)
    log(f"{cfg.num_hidden_layers} layers, {n_layers} reachable from "
        f"hidden_states[-2]; {args.parts} parts {cuts}")
    if args.plan_only:
        return

    from remote_safetensors import open_shards

    # The text encoder ships as model*.safetensors, not diffusion_pytorch_model*.
    from huggingface_hub import hf_hub_download
    index = hf_hub_download(REPO, f"{SUBDIR}/model.safetensors.index.json")
    weight_map = json.load(open(index))["weight_map"]
    from remote_safetensors import RemoteSafetensors
    readers = {s: RemoteSafetensors(REPO, f"{SUBDIR}/{s}")
               for s in sorted(set(weight_map.values()))}
    del open_shards

    if args.token_emb:
        dump_token_emb(args.work, readers, weight_map)
        return

    odir = os.path.join(args.work, "onnx")
    os.makedirs(odir, exist_ok=True)
    manifest = []
    for pi, (a, b) in enumerate(cuts):
        n = pi + 1
        entry = {"part": n, "layers": [a, b]}
        if args.only and n != args.only:
            manifest.append(entry)
            continue
        pdir = os.path.join(odir, f"clip_part{n}")
        os.makedirs(pdir, exist_ok=True)
        out = os.path.join(pdir, f"clip_part{n}.onnx")
        if os.path.exists(out):
            log(f"part{n} already exported, skipping")
            manifest.append({**entry, "onnx": out})
            continue
        log(f"building clip part{n} (layers {a}..{b})")
        chunk = build_chunk(cfg, readers, weight_map, a, b,
                            first=(pi == 0), last=(pi == len(cuts) - 1))
        torch.onnx.export(chunk, chunk.example_inputs(), out, opset_version=17,
                          dynamo=False, input_names=chunk.input_names,
                          output_names=chunk.output_names)
        from export_dit import check_io_names
        check_io_names(out, chunk.input_names, chunk.output_names)
        size = sum(os.path.getsize(os.path.join(pdir, f)) for f in os.listdir(pdir))
        log(f"  clip part{n} onnx {size / 1e9:.2f} GB")
        manifest.append({**entry, "onnx": out, "inputs": chunk.input_names,
                         "outputs": chunk.output_names})
        del chunk
        gc.collect()

    mpath = os.path.join(args.work, "clip_manifest.json")
    json.dump({"parts": manifest, "n_parts": len(cuts), "seq": SEQ, "dim": DIM},
              open(mpath, "w"), indent=2)
    log(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
