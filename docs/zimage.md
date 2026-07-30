# Z-Image on Local Dream (`--type zimage`)

Runner support for [Z-Image](https://github.com/Tongyi-MAI/Z-Image) /
Z-Image-Turbo (Tongyi-MAI), reusing the QNN/HTP stack this app already runs
SD1.5, SDXL and Anima on.

This document covers the two things the code cannot: **what the converted model
files must look like**, and **what is and isn't realistic to quantize**.

---

## 1. Status

Implemented and wired end to end:

- `--type zimage` backend format, file layout, and pipeline
  (`PipelineZImage.hpp`).
- Qwen3 chat-template prompt handling, token-embedding lookup, prompt weighting,
  `/tokenize` budget accounting (`TextEncoder::processZImagePrompt`).
- Rectified-flow sampling with the **diffusers** `FlowMatchEulerDiscreteScheduler`
  schedule (shift 3.0, `t_scale` 1000).
- Flux VAE latent scaling, 16-channel latents, img2img / inpaint /
  aspect-ratio padding, prompt caching, low-RAM and one-part-at-a-time loading.
- Android side: `ZIMAGE` marker detection, model listing, backend launch flags,
  remote (LAN) host/client propagation, Turbo-appropriate defaults.

Not included: **the converted model binaries.** Producing them needs the
Qualcomm AI Engine Direct SDK, the original weights, and a machine to run the
conversion on. Section 4 is the contract they must satisfy.

Not supported, on purpose: UltraFix (tiling a 6B DiT is not worth it), textual
inversion (Qwen3's embedding space is not CLIP's), and LoRA.

---

## 2. About "Q2"

Short version: **there is no 2-bit path on the NPU, and the CPU path that does
have one is far too slow to use.** The practical target is `w4a16` on the HTP.

**NPU (QNN/HTP).** The HTP quantizer supports 8/16-bit activations and 4/8-bit
weights. There is no 2-bit weight format, so a "Z-Image Q2" NPU export is not
something that can be built today regardless of how the graph is cut. The floor
is 4-bit weights:

| Component | Params | `w8a16` | `w4a16` |
|---|---|---|---|
| S3-DiT | ~6.0 B | ~6.0 GB | ~3.0 GB |
| Qwen3-4B body (embedding excluded, see below) | ~3.6 B | ~3.6 GB | ~1.8 GB |
| Flux VAE (enc + dec) | ~0.08 B | ~0.16 GB | — |

Plus `token_emb.bin`: 151936 × 2560 fp16 ≈ **778 MB**, kept out of the graph and
mmap'd (the token lookup runs on the CPU, as it does for Anima).

So a `w4a16` model is roughly **5.5–6 GB on disk**. Nothing holds all of it at
once — the pipeline loads one stage at a time under `--lowram`, and
`--anima_seq_dit` drops the peak further to a single DiT part.

**CPU (MNN).** MNN's `--weightQuantBits` does go down to 2, so a genuine Q2
build is possible there — and it still isn't usable. One DiT step is
~2 × 6e9 × 4608 tokens ≈ **55 TFLOP**; a phone CPU delivering tens of GFLOPS
needs on the order of ten minutes *per step*, so eight steps is hours. That is
why this format is NPU-only. (For scale: the same step on an HTP doing tens of
T-MAC/s lands in the seconds, which is what makes 8-step Turbo viable at all.)

If you want a smaller file, the lever that actually exists is mixed precision —
keep the first/last blocks and the modulation paths at 8-bit and push the bulk
of the attention/FFN weights to 4-bit — not a lower uniform bit width.

---

## 3. File layout

Everything goes in one model directory under the app's `models/`:

```
<model_dir>/
  ZIMAGE              # empty marker file; makes the app list it as a Z-Image model
  tokenizer.json      # Qwen2Tokenizer (from Z-Image-Turbo/tokenizer/)
  token_emb.bin       # fp16 [vocab, 2560] token embedding table, row-major
  clip.bin            # QNN context binary: Qwen3-4B text encoder
  unet_part1.bin      # QNN context binaries: the S3-DiT, cut into N pieces
  unet_part2.bin      #   numbered from 1, contiguous, N discovered on disk
  ...
  vae_decoder.bin     # QNN context binary
  vae_encoder.bin     # optional; enables img2img / inpaint / aspect ratios
  config.json         # optional defaults (see below)
```

A recommended `config.json` for Turbo:

```json
{
  "default_steps": 8,
  "default_cfg": 1.0,
  "default_scheduler": "euler",
  "default_prompt": "",
  "default_negative_prompt": ""
}
```

`cfg` must stay at **1.0** for Turbo: it is distilled guidance-free, and at
cfg 1.0 the pipeline skips the unconditional pass entirely, halving the work per
step. Raising it doubles generation time *and* degrades a distilled model.

Run it manually with:

```
stable_diffusion_core --type zimage --model_dir <dir> --lib_dir <qnn libs> \
    [--lowram] [--anima_seq_dit] [--no_img2img]
```

---

## 4. Graph IO contract

All graphs are converted with `--preserve_io_datatype`, so every IO tensor is
fp32 and the app memcpys directly, as for the SDXL and Anima paths. **Tensors
are bound by name, not by position** (except where noted), so the ONNX
`input_names` / `output_names` below are load-bearing.

Constants referenced here live in `Config.hpp`: `S = 512` (context length),
`D = 2560` (Qwen3-4B hidden / DiT `cap_feat_dim`), `C = 16` (latent channels),
`H = W = 128` (latent grid at 1024×1024).

### `clip.bin` — Qwen3-4B text encoder

| | name | shape |
|---|---|---|
| in | `input_embedding` | `[1, S, D]` |
| in | `attention_mask` | `[1, S]` (1.0 = real token, 0.0 = pad) |
| out | `context` | `[1, S, D]` |

Two things are easy to get wrong here:

- **Take `hidden_states[-2]`, not the final layer.** Z-Image's reference
  pipeline reads the second-to-last hidden state. Exporting `last_hidden_state`
  will produce plausible-looking but consistently wrong images.
- **The token embedding is not in this graph.** The app does the lookup on the
  CPU against `token_emb.bin` so prompt weighting can scale individual rows.
  Export the model from the embedding output inward, and dump the embedding
  matrix separately as fp16.

The app builds the sequence with Qwen3's chat template
(`add_generation_prompt=True`, `enable_thinking=True`), i.e.
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`, pads to `S`
with `<|endoftext|>` (151643) and marks the padding in `attention_mask`. Build
the graph's causal mask from `attention_mask` — do not assume a fixed prompt
length.

### `unet_part1.bin` … `unet_partN.bin` — the S3-DiT

The DiT is cut between transformer blocks. The chain is uniform:

**Part 1**

| | name | shape |
|---|---|---|
| in | `sample` | `[1, C, H, W]` |
| in | `timestep` | `[1]` — this is `sigma * 1000`, not sigma |
| in | `context` | `[1, S, D]` |
| in | `text_mask` | `[1, S]` |
| out | `hidden` | `[1, T, 3840]` — the fused caption+image token stream |
| out | `emb` | `[1, 3840]` — the timestep modulation vector |

**Every later part**

| | name | shape |
|---|---|---|
| in | `hidden` | `[1, T, 3840]` |
| in | `emb` | `[1, 3840]` |
| in | `timestep` | `[1]` |
| in | `text_mask` | `[1, S]` |
| out | `hidden` | `[1, T, 3840]` (non-terminal parts) |
| out | `out_sample` | `[1, C, H, W]` (terminal part only) |

Rules the runner enforces:

- A part is **terminal** iff it exposes an output named `out_sample`. There is
  no separate flag, and no special case for `N = 1` — a single-context export
  where part 1 is terminal works as-is.
- Non-terminal parts must emit `hidden`. They may re-emit `emb`, but do not have
  to: `emb` is constant across the chain and the host re-supplies the copy part
  1 produced.
- `timestep` and `text_mask` are re-supplied to every part rather than threaded
  through the handoff. This is deliberate, and follows what the Anima split
  found the hard way: passing precomputed adaLN/RoPE tensors as flat graph
  inputs forces the residual stream into a slow HTP layout. Recompute them
  inside each part from `timestep`.
- Because the stream is single-stream, `context` is **not** an input past part 1
  — the caption tokens are already inside `hidden`.
- Handoff shapes are checked against the graph's declared tensor sizes at run
  time; a mismatch is reported rather than memcpy'd.

`T` is up to you (it is whatever your patchify + concat produces, nominally
`S + (H/2)·(W/2)` = 512 + 4096 = 4608). The runner never assumes a value for it;
it sizes the handoff from the graph.

### `vae_decoder.bin` / `vae_encoder.bin` — Flux AutoencoderKL

Bound **positionally**, matching the SDXL/Anima VAE paths.

| | | shape |
|---|---|---|
| decoder in | latents | `[1, C, H, W]` |
| decoder out | pixels | `[1, 3, 1024, 1024]`, −1…1 |
| encoder in | pixels | `[1, 3, 1024, 1024]`, −1…1 |
| encoder out 0 | mean | `[1, C, H, W]` |
| encoder out 1 | std | `[1, C, H, W]` |

The graphs work in **raw VAE latent space**. The scaling
(`latent / 0.3611 + 0.1159` on the way in, the inverse on the way out) is
applied by the pipeline — do not bake it into the graph, or it will be applied
twice.

---

## 5. Sampling, for reference

The pipeline reproduces `FlowMatchEulerDiscreteScheduler(shift=3.0,
num_train_timesteps=1000, use_dynamic_shifting=False)`:

```
sigmas    = linspace(1.0, 1/1000, steps)
sigmas    = 3.0 * sigmas / (1 + 2.0 * sigmas)      # SNR shift, applied ONCE
timesteps = sigmas * 1000                           # what the graph receives
denoised  = x - v * sigma                           # CONST parameterization
```

Note that this is **not** the same schedule Anima uses. Anima follows ComfyUI's
`normal_scheduler`, which linspaces between the already-shifted `sigma_max` and
`sigma_min` and then shifts a second time. At 8 steps the two disagree
substantially (final timestep 2.99 vs 8.93), which is why
`FlowMatchScheduler::SigmaSchedule` selects between them rather than one being
reused for both.

Sampling is deterministic Euler by default. Selecting `euler_a` in the UI
switches to the ancestral variant (`eta = 1.0`); for a distilled 8-step model
the deterministic one is what the model was trained for.

---

## 6. Memory and speed expectations

On a 16 GB device, `--lowram` (the default for this format in the app) is
enough: each stage is loaded around its use and the DiT parts stay resident
together for the denoising loop, sharing one HTP spill-fill buffer.

On 12 GB, add `--anima_seq_dit` ("DiT sequential loading" in settings). Peak
memory drops to a single part, at the cost of reloading each part on every step
— for an N-part Z-Image that is `8 × N` context loads per image, so expect this
to dominate the wall clock.

If context creation fails with a message about the required spill-fill size,
set `LOCALDREAM_ZIMAGE_SPILL_FILL_BYTES` to the largest value the log reports
across the parts. Setting it to `0` disables buffer sharing entirely.

---

## 7. Publishing the weights

The app ships a built-in entry for Z-Image Turbo (`ModelRepository.createZImageTurboModel`)
that points at a Hugging Face repo which **does not exist yet**. Until the
archive below is published, tapping Download fails; everything else about the
entry is wired.

What the entry expects:

| | |
|---|---|
| Model id | `zimage_turbo` (reserved, so a side-loaded model cannot shadow it) |
| URL | `<baseUrl>/P2Enjoy/z-image-turbo-qnn/resolve/main/z_image_turbo_w4a16_qnn2.39_8gen3.zip` |
| `baseUrl` | user-selected: `https://huggingface.co/`, an hf-mirror, or a custom host |
| Shown on | 8 Gen 3 / 8 Elite class SoC **and** ≥ 11 GB reported RAM |
| Listed size | `5.8GB` — update `approximateSize` if the real archive differs |

The zip's contents are extracted directly into the model directory, so it must
unpack to exactly the layout in section 3 — the files at the archive root, not
nested inside a folder. `ZIMAGE` and `config.json` are not needed for the
built-in entry (the marker only matters for side-loaded models, and steps / cfg
/ scheduler are set in code because `codeDefaults` outrank a bundled
`config.json`). They are harmless to include, and worth including if you also
want the same archive to work as a manual import.

To point the app somewhere else, change `fileUri` in
`createZImageTurboModel()`; to widen or narrow which devices see it, change
`isZImageCapableDevice()`.

---

## 8. Building the APK

**Gradle does not build the native code.** There is no `externalNativeBuild`
block in `app/build.gradle.kts` — `libstable_diffusion_core.so` and the QNN
runtime libraries are produced by a separate CMake build and are *gitignored*
(`app/src/main/jniLibs`, `app/src/main/assets/qnnlibs`). Running `./gradlew`
on a fresh clone therefore succeeds but yields an APK **with no native library
in it**, which cannot run any model. The native build has to happen first.

Prerequisites:

- Android SDK (`compileSdk 37`), JDK 21, Gradle 9.3.1 (via the wrapper)
- Android NDK **r28** at `/data/android-ndk-r28`, or `ANDROID_NDK_ROOT` set
  (see `app/src/main/cpp/CMakePresets.json`)
- Qualcomm AI Engine Direct SDK **2.39.0.250926** at `/data/qairt/2.39.0.250926`
  (path hardcoded as `QNN_SDK_ROOT` in `app/src/main/cpp/CMakeLists.txt`).
  This one is not optional and not publicly downloadable — it requires a
  Qualcomm Developer account. Without it the CMake configure step fails
  immediately at the `file(COPY ${QNN_SDK_ROOT}/...)` calls.
- `ninja`, `ccache`
- A Rust toolchain (for the `tokenizers-cpp` submodule)

Steps:

```bash
git submodule update --init --recursive     # MNN, tokenizers-cpp, zstd, xtensor, ...
cd app/src/main/cpp && ./build.sh            # -> jniLibs/arm64-v8a/ + assets/qnnlibs/
cd ../../../.. && ./gradlew assembleBasicRelease
```

Flavors are `basic` and `filter` (the latter bundles the NSFW checker), so the
tasks are `assembleBasicRelease` / `assembleFilterRelease`. Release signing
reads `RELEASE_STORE_FILE`, `RELEASE_STORE_PASSWORD`, `RELEASE_KEY_ALIAS` and
`RELEASE_KEY_PASSWORD` from Gradle properties; use `assembleBasicDebug` if you
just want something installable.

Only `arm64-v8a` is built — the QNN backend has no other target.
