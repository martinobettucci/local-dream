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

This has been wrong twice. First it said 2-bit was impossible; then it said
2-bit was supported and would halve the size again. **Both were reasoning from
the `--help` text rather than from an artifact.** Section 10 has the measurement
that settles it, and the short version is:

`--weights_bitwidth 2` is accepted, converts, and passes Hexagon v73 op
validation — but it produces a DLC **byte-identical** to `--weights_bitwidth 4`,
because the bit width is a metadata field on an 8-bit container, not the
storage. There is no 2-bit datatype in QNN. The real 2x comes from the
undocumented `--pack_4_bit_weights`, which switches to the genuine packed
`UFIXED_POINT_4` type. **4 bits per weight is the floor on this hardware**, and
w2 buys accuracy loss for zero bytes.

Sizes at each width, DiT plus Qwen3-4B body:

| Component | Params | `w8a16` | `w4a16` unpacked | `w4a16` packed | `w2a16` |
|---|---|---|---|---|---|
| S3-DiT | ~6.0 B | ~6.0 GB | ~6.0 GB | ~3.0 GB | ~6.0 GB |
| Qwen3-4B body (embedding excluded) | ~3.6 B | ~3.6 GB | ~3.6 GB | ~1.8 GB | ~3.6 GB |
| Flux VAE (enc + dec) | ~0.08 B | ~0.16 GB | — | — | — |

The unpacked and `w2a16` columns are not a mistake — see section 10.

Plus `token_emb.bin`: 151936 x 2560 fp16 = **778 MB**, kept out of the graph and
mmap'd (the token lookup runs on CPU, as it does for Anima). At w4a16 the whole
model is roughly 5.5-6 GB; at w2a16 nearer 3.5 GB.

**Bit width does not change the conversion's memory cost.** `qairt-quantizer`
runs the graph to collect activation statistics regardless of the weight width,
so w2, w4 and w8 all need the same RAM. `--enable_float_fallback
--float_bitwidth 16` is the only mode that skips calibration (and it forbids
`--input_list`), but it produces fp16 — ~12 GB for the DiT, which defeats the
point.

**CPU (MNN).** MNN's `--weightQuantBits` also goes down to 2, and it still is
not usable: one DiT step is ~2 x 6e9 x 4608 tokens = **55 TFLOP**, which is
minutes per step on a phone CPU. This format is NPU-only.

---

## 3. File layout

Everything goes in one model directory under the app's `models/`:

```
<model_dir>/
  ZIMAGE              # empty marker file; makes the app list it as a Z-Image model
  tokenizer.json      # Qwen2Tokenizer (from Z-Image-Turbo/tokenizer/)
  token_emb.bin       # fp16 [vocab, 2560] token embedding table, row-major
  clip_part1.bin      # QNN context binaries: Qwen3-4B text encoder, cut into
  clip_part2.bin      #   M pieces, numbered from 1, M discovered on disk.
  ...                 #   A single clip.bin is still accepted (the M = 1 case).
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

### `clip_part1.bin` … `clip_partM.bin` — Qwen3-4B text encoder

Split for the same reason the DiT is — 3.5 B parameters cannot be quantized in
one piece — but far cheaper: the chain runs **once per prompt** and the result
is prompt-cached, so the extra hand-offs are amortised over a whole image rather
than paid on every step.

**Part 1**

| | name | shape |
|---|---|---|
| in | `input_embedding` | `[1, S, D]` |
| in | `attention_mask` | `[1, S]` (1.0 = real token, 0.0 = pad) |
| out | `hidden` | `[1, S, D]` (or `context`, if it is also the last part) |

**Every later part**

| | name | shape |
|---|---|---|
| in | `hidden_in` | `[1, S, D]` |
| in | `attention_mask` | `[1, S]` |
| out | `hidden` | `[1, S, D]`, or `context` on the terminal part |

`attention_mask` reaches every part because each one rebuilds the causal +
padding mask from it internally. `hidden_in` rather than `hidden` on the input
for the same ONNX naming reason as the DiT.

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

`T` is the unified sequence length: `(H/2)·(W/2)` **image tokens first**,
then `S` caption slots — 4096 + 512 = 4608 at 1024x1024. The order is image
then caption; diffusers' basic mode builds `[x, cap]`, and getting this
backwards is silent.

**Part 1**

| | name | shape |
|---|---|---|
| in | `sample` | `[1, C, H, W]` |
| in | `timestep` | `[1]` — this is `sigma * 1000`, not sigma |
| in | `context` | `[1, S, D]` |
| in | `pos_ids` | `[1, T, 3]` int32 — 3D RoPE coordinates `(t, h, w)` |
| in | `attn_mask` | `[1, T]` — 1 up to `cap_len`, 0 beyond |
| in | `cap_pad_mask` | `[1, S]` — 1 where a caption row becomes the pad token |
| out | `hidden` | `[1, T, 3840]` — the fused image+caption token stream |
| out | `emb` | `[1, 256]` — the timestep adaLN vector (see below) |

**Every later part**

| | name | shape |
|---|---|---|
| in | `hidden_in` | `[1, T, 3840]` |
| in | `emb` | `[1, 256]` |
| in | `pos_ids` | `[1, T, 3]` int32 |
| in | `attn_mask` | `[1, T]` |
| out | `hidden` | `[1, T, 3840]` (non-terminal parts) |
| out | `out_sample` | `[1, C, H, W]` (terminal part only) |

**The two masks are the whole ballgame.** A static graph fixes the caption at
`S = 512` slots; the reference feeds a variable-length caption padded only to a
multiple of 32. They agree **bit-exactly** anyway — but only if the graph
applies `attn_mask` in *both* of the places the reference gets away without one:

1. the caption refiner (`context_refiner` self-attention), and
2. the main transformer blocks.

Apply it only in (2) and a 12-token prompt lands ~3.6 % off; apply it in
neither and it is ~16 % off (§9). The reference never needs a mask because at
batch 1 every sequence is the same length, so `_prepare_sequence` and
`_build_unified_sequence` both return `attn_mask = None` — it is an omission
you must not copy.

`cap_len = round-up(true_len, 32)`. Slots between `true_len` and `cap_len` are
*real* in the reference — they hold a learned pad token — so they stay inside
`attn_mask`; `cap_pad_mask` is what tells the graph to substitute the pad token
there. Slots past `cap_len` do not exist in the reference and must be masked
out.

`pos_ids` is likewise an input, not a constant: image tokens are positioned at
`cap_len + 1`, so their `t` coordinate moves with the prompt. The RoPE
frequency tables themselves are fixed and should be baked in as initializers,
gathered by `pos_ids`.

Because this is exact, **there is no prompt-length limit beyond `S`** — 512
Qwen tokens, with no quality penalty for short prompts.

Later parts take no `timestep`: `emb` already is the timestep's adaLN vector,
computed once by part 1 and reused by every block.

`emb` is **256 wide, not `dim`**. `TimestepEmbedder` runs 256 -> 1024 -> 256
while the residual stream is 3840, and every block's `adaLN_modulation` consumes
the 256. It is easy to assume the two match — a small test config where
`dim <= 256` makes them coincide. The runner reads the handoff width from the
graph rather than assuming, so an export that gets this wrong fails loudly at
the size check rather than corrupting memory.

Rules the runner enforces:

- A part is **terminal** iff it exposes an output named `out_sample`. There is
  no separate flag, and no special case for `N = 1` — a single-context export
  where part 1 is terminal works as-is.
- Non-terminal parts must emit `hidden`. They may re-emit `emb`, but do not have
  to: `emb` is constant across the chain and the host re-supplies the copy part
  1 produced.
- `pos_ids` and `attn_mask` are re-supplied to every part rather than threaded
  through the handoff, since they are small and prompt-dependent. Build the
  RoPE cos/sin inside each part by gathering the baked tables with `pos_ids`;
  the Anima split found that passing large precomputed adaLN/RoPE tensors as
  flat graph inputs forces the residual stream into a slow HTP layout.
- Because the stream is single-stream, `context` is **not** an input past part 1
  — the caption tokens are already inside `hidden`.
- Handoff shapes are checked against the graph's declared tensor sizes at run
  time; a mismatch is reported rather than memcpy'd.

The runner sizes the `hidden`/`emb` handoff from the graph's own declared
tensor shapes, so the exact hidden layout is yours to choose; it does assume
`T = S + (H/2)·(W/2)` when building `pos_ids` and `attn_mask`.

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

### The short version

```bash
git clone https://github.com/martinobettucci/local-dream && cd local-dream
./tools/zimage/setup_build_env.sh --build
```

That script is the executable form of everything below. On a clean Ubuntu 24.04
box it pulls ~12 GB, occupies ~25 GB, and takes about 45 minutes.

### What it does, and why each step is there

- **Android SDK** (`compileSdk 37`), JDK 21, Gradle 9.3.1 (via the wrapper).
  **The package is `platforms;android-37.0`, not `platforms;android-37`** —
  platform packages are minor-versioned, and sdkmanager's error for the wrong
  name is just `Failed to find package`. `--channel=1` will happily *list*
  packages it then refuses to *install*; if in doubt, read the real package
  names out of `https://dl.google.com/android/repository/repository2-3.xml`.
- **Rust 1.85.0** for `tokenizers-cpp`, pinned in `rust-toolchain.toml`. The
  window is narrow at both ends: newer rustc makes `implicit autoref creates a
  reference to the dereference of a raw pointer` a hard error, which its
  `tokenizers-c` crate trips in two places and `RUSTFLAGS=-A ...` does not
  suppress; older than 1.82 fails on `is_none_or` with E0658. Do not patch the
  vendored source — it would not survive a submodule update.
  `rustup target add aarch64-linux-android` is also required, or the build dies
  on `can't find crate for 'core'`.
- **Android NDK r28**, symlinked to `/data/android-ndk-r28` because
  `app/src/main/cpp/CMakePresets.json` hardcodes that path (`ANDROID_NDK_ROOT`
  in its `environment` block overrides the ambient variable, so exporting one
  is not enough).
- **Qualcomm AI Runtime 2.39.0.250926** at `/data/qairt/2.39.0.250926`, also
  hardcoded, as `QNN_SDK_ROOT` in `app/src/main/cpp/CMakeLists.txt`. The
  *Community* edition needs no Qualcomm account — see section 9. Without it
  CMake fails immediately at the `file(COPY ${QNN_SDK_ROOT}/...)` calls.
- **`ninja` and `ccache`**, which are not a nicety: the native tree is MNN plus
  xtensor plus the QNN SampleApp.
- **All ten submodules.** A missing one surfaces as a not-found header rather
  than as a submodule error.

Steps, if running them by hand:

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

---

## 9. Conversion notes (verified)

Findings from actually setting the toolchain up, rather than from reasoning
about it. Several correct earlier assumptions in this file's history.

**The SDK is publicly downloadable.** The *Qualcomm AI Runtime Community*
edition needs no Qualcomm account:

```
https://softwarecenter.qualcomm.com/api/download/software/sdks/\
Qualcomm_AI_Runtime_Community/All/2.39.0.250926/v2.39.0.250926.zip
```

1.35 GB, unpacks to 3.1 GB at `qairt/2.39.0.250926/` — exactly the layout
`CMakeLists.txt` expects. It contains the complete conversion toolchain
(`qairt-converter`, `qnn-onnx-converter`, `qnn-context-binary-generator`,
`qnn-model-lib-generator`, `qnn-net-run`), the `examples/QNN/SampleApp`
sources the app builds against, `lib/aarch64-android/libQnn*`, and Hexagon
v66–v81. Nothing about conversion needs a GPU: the converter and the context
binary generator are host CPU compilers, and w4a16 calibration runs the ONNX
graph through onnxruntime on CPU.

**RoPE must be de-complexified before export.** `RopeEmbedder` builds its
frequencies with `torch.polar(...) -> complex64` and the attention processor
applies them via `view_as_complex` / `view_as_real`. ONNX has no complex tensor
type, so export dies with `ScalarType ComplexFloat is an unexpected tensor
scalar type` *after* tracing the entire graph — which reads like an obscure
serialization bug rather than what it is.

`tools/zimage/rope_real.py` swaps in the algebraically identical real form:

```
(x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x0*sin + x1*cos)
```

`freqs_cis` then carries a trailing dim of 2 holding `(cos, sin)` instead of
being complex; all other shapes are unchanged. `tools/zimage/verify_rope.py`
checks it on a tiny random-weight model in a few seconds — measured agreement
is `8.3e-07`, i.e. float32 noise. Run it after any diffusers bump.

**The forward is list-based and still needs a static wrapper.** With RoPE
fixed, export gets one step further and stops at
`aten::pad_sequence`. `ZImageTransformer2DModel.forward` takes
`list[Tensor]` for both latents and captions and packs variable-length
sequences, which cannot become a static graph. Since this runner always
generates at batch 1, a fixed 512-token caption and a fixed 1024x1024 canvas,
every length is a compile-time constant, so `patchify_and_embed` and `forward`
can be reimplemented for fixed shapes — that wrapper is what section 4's IO
contract describes, and it is the remaining piece of export work.

Two shape facts worth knowing before writing it: latents are `(C, F, H, W)`
with `F = 1` for stills (the three RoPE axes are t/h/w), and the caption is
padded to a multiple of `SEQ_MULTI_OF` with position ids starting at t=1, the
image tokens then starting at `cap_len + 1`.

**Disk.** The released transformer is fp32, not bf16: 24.6 GB across three
shards, plus 8.05 GB for the Qwen3-4B text encoder and 0.17 GB for the VAE —
33 GB before any ONNX intermediate. Converting on a 30 GB volume means
downloading shard by shard, casting to fp16, writing out per-part weights and
deleting each shard as you go.

**A fixed 512-slot caption is exact, if you mask correctly.**
`tools/zimage/verify_static_equivalence.py` checks this on a tiny random-weight
model in seconds. Masking the caption refiner *and* the main blocks past
`cap_len`:

| prompt | mean relative difference vs. reference |
|---|---|
| 12 tok | 0.0000 % (max abs 0.0) |
| 40 tok | 0.0000 % (max abs 0.0) |
| 100 tok | 0.0000 % (max abs 0.0) |
| 300 tok | 0.0000 % (max abs 7.2e-07) |

Getting there took two wrong turns worth recording, since both look correct:

*Masking neither* (the obvious reading of the reference, which passes
`attn_mask = None` at batch 1) leaves the pad slots as full participants in
attention: 15.6 % mean relative error at 12 tokens, 4.3 % at 300 — the error
scales with the number of pad slots. Note `pad_len = (-ori_len) % 32`, so the
reference never has more than **31** pad tokens; a 512-slot graph would hand
the model ~480 of them, far outside anything it was trained on.

*Masking only the main blocks* still leaves the caption refiner self-attending
over all 512 slots before the unified sequence is built, which contaminates the
real caption rows: 3.6 % at 12 tokens.

**Caption length changes the image, so positions must be inputs.** The
reference keeps only unmasked caption rows (`prompt_embeds[i][prompt_masks[i]]`),
pads that to a multiple of `SEQ_MULTI_OF` (32), and starts the image tokens at
`cap_len + 1`. A static graph has to fix the caption at 512 slots, which would
pin every image token's `t` coordinate at 513 regardless of the prompt.

Measured on the toy model, holding caption content and attention mask identical
and varying only the padded length:

| caption padded to | mean relative difference vs. reference |
|---|---|
| 32 (what the reference does for a 12-token prompt) | — |
| 64 | 6.4 % |
| 128 | 10.4 % |
| 512 | 16.0 % |

Random weights, so the percentages say nothing about perceptual damage — but
the mechanism is real and would apply to every prompt shorter than 512 tokens,
i.e. essentially all of them. Hence `pos_ids` as a graph input, computed host
side in `PipelineZImage::buildPositions`.

One thing to watch when writing the static wrapper: in diffusers 0.39.0
`patchify_and_embed` passes the *already padded* caption length as
`pos_grid_size` to `_pad_with_ids`, which then appends `pad_len` more
coordinates on top — so for any caption that is not already a multiple of 32,
`pos_ids` comes out longer than the padded features. Do not copy that shape
arithmetic verbatim; derive the coordinates as this runner does.

**Environment: three non-obvious prerequisites.** `tools/zimage/setup_qnn_env.sh`
automates these; each one fails with an error that does not name its cause.

| Symptom | Cause |
|---|---|
| `ImportError: libc++.so.1: cannot open shared object file`, then a bogus "circular import" traceback | The SDK links LLVM's C++ runtime, which Community does not bundle and Ubuntu does not install. `apt install libc++1 libc++abi1`. |
| `ImportError: Python version mismatch: module was compiled for Python 3.10` | `libDlModelToolsPy.so` is built for **3.10 exactly**. 3.11 and 3.12 both fail. Keep this venv separate from the torch/ONNX export env. |
| `AttributeError: 'NoneType' object has no attribute 'AttributeProto'` | The converter imports `onnx.mapping`, removed in onnx 1.16. The SDK swallows the ImportError, leaves the module `None`, and dies much later. Pin **onnx < 1.16** (1.15.0 works). |

**QNN caps tensors at 5-D, and the obvious patchify is 7-D.** Transcribing
`_patchify_image` / `unpatchify` literally gives a 7-D view plus a fully
interleaved permute, which the converter rejects:

    Failed to resolve 6D tensor by merging consecutive axes for Transpose
    with permutation [6, 0, 3, 1, 4, 2, 5]

It only reduces rank by merging axes that remain adjacent, which an interleaved
permutation never permits. `static_export.py` expresses the identical
rearrangement with `pixel_unshuffle` / `pixel_shuffle` in <=4-D — these lower to
SpaceToDepth / DepthToSpace, which the HTP handles natively. Verified bit-equal
to the 7-D form in both directions. Mind the ordering: pixel_(un)shuffle is CRD
(channel-major) while a Z-Image token is `(pH, pW, C)`, so a reshape/permute
pair converts between them.

**The DiT conversion path is proven end to end** (on the real 6B weights, not a
toy): 24.6 GB fp32 -> per-part fp16 -> StaticZImageDiT -> ONNX (external-data
format, ~2.1 GB/part) -> `qairt-converter` -> `INFO_CONVERSION_SUCCESS`, a
2.18 GB float DLC for part 8. Two harmless warnings on the way: `GEMM operation
is not supported in the general case, attempting to interpret as FC`, and
`Unused Input nodes found: []`.

Remaining for a shippable model: `qairt-quantizer --weights_bitwidth 4
--act_bitwidth 16 --input_list <calib>` per part, then
`qnn-context-binary-generator` per part, then the same for Qwen3-4B and the VAE.
The quantizer needs a calibration input list — representative `sample`,
`context`, `pos_ids`, `attn_mask` and `cap_pad_mask` tensors as raw files.

**Target the OLDEST Hexagon you intend to support.** Context binaries are
compiled per Hexagon version and do not run on an older one:
8 Gen 2 (SM8550) is v73, 8 Gen 3 (SM8650) is v75, 8 Elite (SM8750) is v79. A
v75 build installs fine on an 8 Gen 2 and then fails to load. Note the app's
own `isSdxlCapableSoc()` list starts at 8 Gen 3, so an 8 Gen 2 never sees the
built-in SDXL cards either — it can still run imported custom NPU models, which
is why a device can "run SDXL" while showing none of the built-in SDXL entries.

**The full toolchain is proven, and it is fast.** Measured on the VAE decoder
(4 CPU cores, no GPU), targeting Hexagon v73:

| stage | time | output |
|---|---|---|
| `qairt-converter --preserve_io_datatype` | 9.5 s | DLC |
| `qairt-quantizer --weights_bitwidth 8 --act_bitwidth 16 --bias_bitwidth 32 --input_list <calib>` | 37.8 s | quantized DLC |
| `qnn-context-binary-generator --dlc_path <q.dlc> --backend libQnnHtp.so --config_file <ext.json>` | 11.9 s | 54 MB `.bin` |

Selecting the Hexagon version takes two nested JSON files — the generator takes
a backend-extensions wrapper, which points at the HTP config that actually
carries `dsp_arch`:

```json
// ext_v73.json  (passed as --config_file)
{"backend_extensions":{"shared_library_path":"libQnnHtpNetRunExtensions.so",
                       "config_file_path":"/abs/path/htp_v73.json"}}
// htp_v73.json
{"devices":[{"dsp_arch":"v73","cores":[{"core_id":0,"perf_profile":"burst",
                                        "rpc_control_latency":100}]}]}
```

The calibration `--input_list` is a text file with one line per sample, each
line `input_name:=/abs/path/sample.raw`, the raw being a bare fp32 dump of the
input tensor.

The generator prints a DDR bandwidth summary including `spill_bytes`. That is
**total spill traffic during compilation, not a buffer size** — do not feed it
to `LOCALDREAM_ZIMAGE_SPILL_FILL_BYTES`. The same VAE decoder reports 436 MB at
256px and 19 GB at 1024px, which makes the scaling obvious once you see both.
The buffer size still has to come from the runtime `querySpillFillSize()` or
from the "smaller than required spill-fill size N" message on a failed context
creation.

**Memory is the binding constraint, not time.** `qairt-quantizer` holds
activations for every calibration sample, and at 1024x1024 that is enough to
OOM a 15 GB box on the *smallest* graph in the model:

    Out of memory: Killed process (python) anon-rss:15740160kB

Dropping `--input_list` from 4 samples to 1 fixed it. Measured on the 1024px VAE
decoder, w8a16, Hexagon v73: convert 13 s, quantize 3m39s, context binary
18m25s, output 198 MB. Note the context-binary step dominates and scales with
graph size, so budget accordingly for the DiT parts. Adding swap is worthwhile
insurance: 6 GB was enough here to stop the OOM recurring.

**Validation.** None of this can be checked without a Snapdragon device. A
converted model that loads and produces an image still needs comparing against
the reference pipeline before it is worth publishing.

**The split count is a property of the converting machine, not the phone.**
`qairt-quantizer` holds a whole part plus one calibration sample's activations,
and was OOM-killed at 19 GB (15 GB RAM + 4 GB swap) on a 4-block part. Nothing
about the weight width changes that — w2, w4 and w8 all calibrate identically.
The only lever is how many blocks a part contains, so the split is chosen from
measured peak RSS rather than from what looks tidy:

```bash
export HF_TOKEN=...
PY=.zimage-env/exportvenv/bin/python QNN=.zimage-env/qnnvenv/bin/python \
  ./tools/zimage/convert_all.sh /scratch 15 4 v73
```

`convert_part.sh` records peak RSS and wall clock per stage to
`<work>/stats/partN.tsv`, and `convert_all.sh` prints the worst case across
every part it built. Pick the smallest part count whose quantize stage fits with
headroom; more parts is not free on device, since each one is a context switch
and a full residual-stream copy across the graph boundary on every step.

`zimage_max_dit_parts` (Config.hpp) caps this at 32, which admits one block per
part — the finest split the 30-block model allows.

**Weights are read over HTTP ranges, not downloaded.** The transformer is
24.6 GB of fp32 across three shards, and the old flow downloaded each shard,
sliced it into per-part fp16 files and deleted it: ~12 GB of intermediate for a
model whose converted form is 3 GB. safetensors is trivially range-readable —
8-byte little-endian header length, JSON header of `{name: {dtype, shape,
data_offsets}}`, then the data — so `tools/zimage/remote_safetensors.py` fetches
exactly the tensors a part needs and nothing else. Reading one DiT block costs
~360 MB of transfer and zero disk. Verified bit-exact against a downloaded copy
on all 244 VAE tensors, including the bf16 path (no numpy bf16, so it is read as
uint16 and bit-cast).

**Every part is uploaded the moment it is built.** `convert_all.sh` pushes each
context binary to the Hub under `partial/` and deletes it locally, then resumes
from the first gap in the remote listing. Two problems, one answer: peak disk
becomes one part's intermediates instead of the whole model, and a conversion
box that is reclaimed mid-run — which is the normal outcome on ephemeral
infrastructure — costs one part rather than the run. Per-part stats go up
alongside as `partial/stats/partN.tsv`.

**Watch out for `delattr` on a reduced model.** A middle part traces neither the
embedders nor the final layer, so `export_dit.py` deletes those submodules
before loading the state dict — otherwise a 2-block part carries over a GB of
randomly-initialised fp32 through ONNX export. Any validation that touches them
unconditionally then fails with `'ZImageTransformer2DModel' object has no
attribute 'all_x_embedder'`; the checks in `StaticZImageDiT.__init__` are gated
on `first` / `last` for exactly this reason.

**`hidden_states[-2]` is the output of layer N-1, so the last layer is dead.**
The reference reads `text_encoder(..., output_hidden_states=True).hidden_states[-2]`.
It is worth knowing exactly which tensor that is, because it decides how much of
Qwen3-4B has to be converted at all. Measured on a 4-layer random Qwen3 by
replaying every stage by hand and matching:

| stage | index in `hidden_states` |
|---|---|
| embeddings | `-5` |
| output of layer 1 | `-4` |
| output of layer 2 | `-3` |
| **output of layer 3** | **`-2`** |
| output of layer 4 | *not present* |
| `norm(output of layer 4)` | `-1` |

The tuple has `num_layers + 1` entries and the last decoder layer's raw output
never appears — it is only there normed. So `[-2]` is the output of layer
**N-1**, and for the real 36-layer encoder **layer 36 and the final RMSNorm are
never evaluated**. The graph needs 35 layers, ~3.5 B parameters rather than
~3.6 B. Do not export `Qwen3Model` and slice afterwards; truncate `model.layers`
to 35 and return the last one's output directly.

Note this is transformers 5.x behaviour, where hidden states are collected by
output-recorder hooks rather than appended in the forward loop. Re-run the check
after a transformers bump rather than trusting the table.

**The text encoder needs splitting too, for the same reason the DiT does.**
35 layers x ~101 M parameters is ~3.5 B, i.e. 14 GB of fp32 — it cannot even be
held for export on a 15 GB box, let alone quantized. Unlike the DiT this costs
almost nothing on device: the text encoder runs once per generation and its
result is prompt-cached, so the extra context switches are amortised over the
whole image rather than paid on every step. It does mean `clip.bin` has to
become a chain in the same way `unet_partN.bin` is.

**`torch.onnx.export` renames colliding IO, and the app binds by name.** The
natural contract for a middle DiT part is `hidden` in, `hidden` out. ONNX cannot
express that — two graph values cannot share a name — and the exporter does not
complain; it silently renames the *input* to `hidden.1`. That survives
`qairt-converter`, survives quantization, survives the context binary, and
finally shows up on device as a missing tensor. Hence `hidden_in` for the input
and `hidden` for the output (`kZImageStateInNames` / `kZImageStateOutNames` in
`QnnModel.hpp`), and hence `check_io_names()` in `export_dit.py`, which reloads
each exported graph and fails the run if any IO name is not what was asked for.

`input_names` and `output_names` are requests, not guarantees. Verify them.

**Why the quantizer needs so much memory: it retains every intermediate.**
`qairt-quantizer` collects per-tensor min/max by *executing the graph on the
QNN_CPU backend*, and to observe a tensor it has to keep it — so peak memory is
roughly the sum of every intermediate tensor in the part, not the working set a
normal inference would need. The failure lands inside graph execution:

```
[  INFO ] [QNN_CPU] QnnGraph finalize end
[  INFO ] [QNN_CPU] QnnGraph execute start
    <killed>
```

At the DiT's 1024x1024 shape that sum is dominated by one tensor per block. The
unified sequence is `T = 4096 + 512 = 4608`, and attention is 30 heads, so a
single score matrix is

    30 x 4608 x 4608 x 4 B = 2.55 GB

and softmax's output is another. Call it ~6 GB per block, against ~0.7 GB of
fp32 weights per block — **activations dominate by an order of magnitude, and
the block count is the only thing that scales them.**

Measured, one part at a time, w4a16, on 16 GB RAM + 6 GB swap:

| blocks/part | parts | export peak | convert peak | quantize peak | outcome |
|---|---|---|---|---|---|
| 4 | 8 | — | — | >19 GB | OOM-killed |
| 2 | 15 | 5.3 GB | 3.4 GB | 15.4 GB RSS + ~5 GB swap | OOM-killed (rc 137, 202 s) |
| 1 | 30 | — | — | — | see below |

Note the export and convert stages are nowhere near the limit — export peaks at
5.3 GB for a 2-block part and the converter at 3.4 GB. **Only quantization is
memory-bound**, so there is no point splitting finer than quantization requires.

**The HTP has no `IsNan`, and `F.scaled_dot_product_attention` emits one.**
This is the first failure that appears only at the *last* stage. The DLC
converts cleanly, quantizes cleanly, and then
`qnn-context-binary-generator` refuses it:

```
[ ERROR ] Input[0] has incorrect Datatype 0x416.
[ ERROR ] validateNativeOps master op validator
          /layers.0/attention/IsNaN:qti.aisw:IsNan failed 3110
[ ERROR ] Failed to validate op /layers.0/attention/IsNaN with error 0xc26
```

SDPA with a **boolean** `attn_mask` decomposes, on export, into a form that
guards rows where every key is masked — softmax of an all-`-inf` row is NaN — so
torch inserts `IsNaN` plus a `Where`. The HTP has neither, and the bool input
trips a datatype check on top.

`rope_real.py` therefore writes attention out as MatMul / Add / Softmax /
MatMul, with an **additive** mask: `0` where attending is allowed and `-1e4`
where it is not. Two reasons for `-1e4` rather than `-inf`: activations are
quantized to 16 bits and need a finite range, and no NaN guard is required at
all here because no row is ever fully masked — the runner's sequence puts the
image tokens first and always keeps them. Measured against SDPA on random
inputs: `3.6e-07` max abs difference with a mask, against `3.0e-07` for the
unmasked control, i.e. summation-order noise rather than a behaviour change.

Worth internalising: **a graph that quantizes is not a graph that compiles.**
Op-support failures surface only at context-binary generation, which is also the
slowest stage. Build one part end to end before starting the other 29.

### Text encoder export, measured

`tools/zimage/export_clip.py` builds the chain; `verify_clip_chunk.py` checks it
on a tiny random Qwen3 in seconds and is worth re-running after any transformers
bump. It reports **0.0 max abs difference** against
`hidden_states[-2]` at 1, 2, 3 and 5 parts — with a deliberate control that runs
all N layers instead of N-1 and differs by 7.1e-02, because without that control
the check would pass just as happily with an off-by-one in `usable_layers()`.

First real export, 3 of the 35 layers: **1.21 GB** of ONNX, inputs
`input_embedding` / `attention_mask`, output `hidden` — names intact, no
rename. Op scan comes back clean:

```
Constant 146, Mul 51, Cast 30, Add 27, MatMul 27, Reshape 19, Transpose 15,
Pow 12, ReduceMean 12, Sqrt 12, Div 12, Slice 12, Where 7, Neg 6, Concat 6,
Unsqueeze 6, ConstantOfShape 6, Equal 6, Expand 6, Softmax 3, Sigmoid 3,
Greater 1, And 1
```

No `IsNaN`, unlike the DiT's first attempt. The difference is the mask: the
chunk builds an **additive float** mask from `attention_mask` inside the graph
(0 / -1e4), and SDPA given a float mask has no fully-masked-row case to guard,
so nothing needs an `IsNaN`. Rotary tables are baked in as constants — the app
always right-pads to 512, so positions are compile-time known and only the mask
varies with the prompt.

The graph IO contract mirrors the DiT's, including `hidden_in` on later parts
for the same ONNX naming reason:

| part | in | out |
|---|---|---|
| 1 | `input_embedding` `[1,512,2560]`, `attention_mask` `[1,512]` | `hidden` |
| middle | `hidden_in`, `attention_mask` | `hidden` |
| last | `hidden_in`, `attention_mask` | `context` `[1,512,2560]` |

**Calibration data has to look like runtime data, or the model runs and
produces garbage.** The quantizer derives every activation's min/max by
executing the graph on the calibration inputs, so an input whose calibration
distribution does not match runtime yields an encoding that clips or wastes its
whole range. The first version of this filled every float input with `N(0,1)`
and every int input with zeros, which is wrong for four of the DiT's six:

| input | N(0,1) / zeros gives | reality |
|---|---|---|
| `timestep` | ~±3 | `sigma * 1000`, so ~1000 down to ~3 over 8 steps |
| `pos_ids` | all zeros | real 3D RoPE `(t, h, w)`; at position 0 the rotation is the identity, so no downstream tensor sees its true range |
| `attn_mask` | noise around 0 | a 0/1 indicator |
| `cap_pad_mask` | noise around 0 | a 0/1 indicator |

Only `sample`, `hidden_in`, `emb` and `context` are legitimately near-Gaussian.
`tools/zimage/make_calib.py` builds the rest properly, taking the positions from
`static_export.build_positions()` — the same function the export uses and the
C++ mirrors, rather than a numpy reimplementation, because that particular piece
of arithmetic diverging is what once measured 15.6 % error.

`timestep` gets the **top** of the schedule (1000). One calibration sample can
only pin one value per input, and the quantizer's range spans what it observed;
calibrating at the small end would clip every early step, which is where the
image is actually decided.

This is the worst kind of bug in this pipeline: nothing fails. The part exports,
converts, quantizes and compiles, and only the images are wrong — after all 30
parts have been built.

## 10. Weight bit width, measured

Section 2 said 2-bit weights are supported because `qairt-quantizer --help`
lists them. That is true and it is also not the useful question. Measured on a
real 1-block DiT part (181 M parameters), same float DLC quantized four ways:

| build | tensor datatype | encoding `bitwidth` | DLC bytes | actual bits/weight |
|---|---|---|---|---|
| float | fp32 | — | 724,077,604 | 32 |
| `--weights_bitwidth 8` | `uFxp_8` | 8 | 181,246,372 | 8 |
| `--weights_bitwidth 4` | `uFxp_8` | 4 | 181,246,412 | **8** |
| `--weights_bitwidth 2` | `uFxp_8` | 2 | 181,246,412 | **8** |
| `--weights_bitwidth 4 --pack_4_bit_weights` | `uFxp_4` | 4 | **90,806,732** | **4** |

**w2 and w4 produce byte-identical DLCs.** `--weights_bitwidth` alone sets a
metadata field on the encoding, not the storage. `QnnTypes.h` says so directly:

> data quantized to a lower precision will still occupy the full extent of bits
> allotted to the tensor as per its data type in unpacked form

So `--weights_bitwidth 2` throws away three quarters of the quantization levels
and stores the result in exactly as many bytes. It is a pure accuracy loss.

**`--pack_4_bit_weights` halves the DLC and then the HTP refuses to run it.**
The flag is hidden (`help=argparse.SUPPRESS`) and it does what it says: the
weight tensor becomes `QNN_DATATYPE_UFIXED_POINT_4`, tightly packed two per
byte, 90,806,732 bytes. `qnn-context-binary-generator` for v73 then rejects the
graph:

```
Unsupported input/output datatypes requested for the HTP Op 'FullyConnected'
  in[0]:QNN_DATATYPE_UFIXED_POINT_16
  in[1]:QNN_DATATYPE_UFIXED_POINT_4      <- the packed weights
  in[2]:QNN_DATATYPE_SFIXED_POINT_32 (optional)
  out[0]:QNN_DATATYPE_UFIXED_POINT_16
```

and helpfully lists every combination it *does* accept. Collected across all
three configurations:

| activation config | weight datatypes accepted for `FullyConnected` |
|---|---|
| FP16 | `FLOAT_16`, `FLOAT_32`, `SFIXED_POINT_8` |
| INT16 | `SFIXED_POINT_8`, `SFIXED_POINT_16`, `UFIXED_POINT_8`, `UFIXED_POINT_16` |
| INT8 | `SFIXED_POINT_8`, `UFIXED_POINT_8` |

**No 4-bit weight datatype appears in any configuration.** The narrowest weight
tensor Hexagon v73 will accept is 8 bits, whatever the encoding says.

So the runnable form of "4-bit" is exactly what `--weights_bitwidth 4` alone
produces: 4-bit *values* in an 8-bit container. That is also what the HTP's own
int4 guidelines page means — it promises power and latency benefits "solely from
the lower transfer of data to/fro VTCM", i.e. from the narrower value range, not
from a smaller tensor type on disk.

**Every runnable weight width is the same size.** w8a16, w4a16 and w2a16 all
produce ~181 MB for this part. There is no size lever here at all; the only ones
are the activation width and the model itself.

Is w2 *rejected* by the hardware? No — quantizing at 2 and running
`qnn-context-binary-generator` for Hexagon v73 passes op validation and proceeds
to compile (5 minutes with no error, against the 1-second rejection an
unsupported op like `IsNan` produces). It is not incompatible. It is pointless.

The HTP backend documents INT4 explicitly
(`docs/QNN/HTP/htp-network-design-recommendations/htp_guidelines_int4_weights.html`),
listing where the power and latency benefits apply — Conv2D 1x1, FullyConnected
and MatMul with `out_channels > 32`, which is essentially every weight in the
DiT. There is no INT2 equivalent page.
