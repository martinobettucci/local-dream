---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
tags:
  - text-to-image
  - qualcomm
  - qnn
  - npu
  - android
  - local-dream
library_name: local-dream
---

# Z-Image Turbo — Qualcomm NPU (QNN) build for Local Dream

> ## ⚠️ UNTESTED
>
> **These artifacts have never been run on a Snapdragon device.** They were
> converted and compiled on a machine with no NPU, so nothing here has produced
> an image. They are published so they *can* be tested, not because they work.
>
> What has been verified, and what has not:
>
> | | |
> |---|---|
> | Static export matches the reference pipeline | ✅ bit-exact on CPU |
> | ONNX matches PyTorch | ✅ ~1e-6 |
> | Graphs accepted by `qairt-converter` | ✅ |
> | Quantization error after w4a16 | ❌ never measured |
> | Runs on an NPU at all | ❌ never attempted |
| Whether 8 Gen 2 (v73) has the headroom for a 6B DiT | ❌ unknown |
> | Output image quality | ❌ unknown |
>
> If it does not work, that is expected rather than surprising. Please open an
> issue with logs. This banner will be removed once someone confirms a
> successful generation.

A conversion of [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
— a 6B Scalable Single-Stream DiT with a Qwen3-4B text encoder and the Flux VAE
— to Qualcomm QNN context binaries, for the `zimage` backend in
[Local Dream](https://github.com/xororz/local-dream).

## The APK

`LocalDream-zimage-2.8.1-arm64-v8a-UNTESTED-debug.apk` in this repo is a build
of Local Dream with `--type zimage` support compiled in — the stock releases do
not have it, so the weights below need this build (or your own from source).

It is a **debug** build: installable and signed with the standard Android debug
key, so it coexists with a Play/release install rather than upgrading it.
arm64-v8a only. Sideload with `adb install <apk>` or a file manager.

The Z-Image runner inside it has never executed a single graph. Installing it
and reaching the model list proves the app works; it proves nothing about
Z-Image.

## Requirements

- Snapdragon 8 Gen 2 or newer. The context binaries are built for **Hexagon
  v73** (8 Gen 2), which also runs on v75 (8 Gen 3) and v79 (8 Elite) — a
  binary built for a *newer* Hexagon will not load on an older one, which is
  why v73 is the target rather than v75.
- **16 GB RAM** recommended; 12 GB needs *DiT sequential loading* in settings
- A Local Dream build with `--type zimage` support

## Install

Download and unzip into the app's model directory, or use the in-app download.
The archive unpacks to the model directory root — not into a subfolder.

## `partial/` — conversion in progress

`partial/` holds pieces that have been converted but do not yet add up to a
runnable model. **It is not installable.** Each file is uploaded the moment it
is built, because the conversion runs on ephemeral machines with less free disk
than the finished model needs — publishing every piece immediately is what stops
a reclaimed container from costing the whole run.

`partial/stats/partN.tsv` records peak RSS and wall clock per stage
(`stage`, `peak_rss_kb`, `seconds`, `exit_code`) for the machine that built each
part. That is what the split count is chosen from: `qairt-quantizer` holds a
whole part plus a calibration sample's activations, and fewer blocks per part is
the only lever on it.

When `partial/` is complete it is repackaged as the archive above and this
section goes away.

## What is in the archive

| File | What it is |
|---|---|
| `tokenizer.json` | Qwen2Tokenizer |
| `token_emb.bin` | fp16 `[vocab, 2560]` token embeddings; the lookup runs on CPU so prompt weighting can scale rows |
| `clip.bin` | Qwen3-4B text encoder, tapped at `hidden_states[-2]` |
| `unet_part1..N.bin` | the S3-DiT, split across contexts (6B does not fit one) |
| `vae_decoder.bin` / `vae_encoder.bin` | Flux AutoencoderKL, 16 latent channels |
| `config.json` | 8 steps, cfg 1.0, euler |

## Generation settings

Turbo is distilled to **8 steps** and trained **guidance-free**, so keep
**cfg = 1.0**. At cfg 1.0 the pipeline skips the unconditional pass entirely,
halving the work per step; raising it both doubles generation time and degrades
a distilled model. Prompts are natural language, not booru tags. Negative
prompts are not evaluated at cfg 1.0.

Fixed 1024×1024 canvas; other aspect ratios go through inpaint padding. Prompt
limit is 512 Qwen tokens, and short prompts cost nothing in quality — the static
512-slot caption graph is bit-exact against the reference's variable-length one.

## Quantization

`w4a16` — 4-bit weights, 16-bit activations.

Note that 2-bit weights are also supported by the toolchain
(`qairt-quantizer --weights_bitwidth` accepts 2, 4, 8 or 16), so a w2a16 build
is possible and would roughly halve the size again. It has not been built or
measured here, and no quality comparison exists.

## Conversion

Reproducible from `tools/zimage/` in the Local Dream repo. See `docs/zimage.md`
for the graph IO contract and the non-obvious parts — RoPE has to be
de-complexified, the sequence is ordered `[image, caption]`, positions must be
graph inputs, and the attention mask has to be applied to the caption refiner as
well as the main blocks.

## Credits

Model: [Tongyi-MAI](https://github.com/Tongyi-MAI/Z-Image), Apache 2.0.
App: [xororz/local-dream](https://github.com/xororz/local-dream).
