#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>

inline int sample_width = 64;
inline int sample_height = 64;
// CLIP hidden sizes are fixed: 768 for SD1.5 / SDXL encoder 1 (CLIP-L),
// 1280 for SDXL encoder 2 (CLIP-G).
inline constexpr int text_embedding_size = 768;
inline constexpr int text_embedding_size_2 = 1280;
inline int output_width = 512;
inline int output_height = 512;

// ---- Anima (DiT + Qwen) constants ------------------------------------------
// Anima latents are 16-channel (Wan 2.1 / Qwen-Image VAE) instead of SD's 4.
inline constexpr int anima_latent_channels = 16;
// Qwen3-0.6B text encoder: 512-token input, 1024-dim hidden states (no pooled
// output, no learned positional table — RoPE is internal to the model). The
// merged text encoder (Qwen + LLM adapter) re-grids onto the T5 token sequence,
// so its OUTPUT context — what the UNet's encoder_hidden_states consumes — is
// also 512 long. Two distinct lengths (both 512, kept separate so the Qwen and
// T5 sides can diverge if a future export changes one):
//   anima_qwen_seq_len : qwen input_embedding + qwen_mask
//   anima_text_seq_len : t5_ids/t5_mask, clip context output,
//                        Conditioning hidden, UNet encoder_hidden_states
inline constexpr int anima_qwen_seq_len = 512;
inline constexpr int anima_text_seq_len = 512;
inline constexpr int anima_text_embedding_size = 1024;
// Wan 2.1 per-channel latent normalization (16 channels). The diffusion model
// works in the normalized space; the VAE consumes/produces the de-normalized
// latent: vae_latent = model_latent * std + mean  (process_out), and the
// inverse for encoding: model_latent = (vae_latent - mean) / std.
inline constexpr float anima_latent_mean[16] = {
    -0.7571f, -0.7089f, -0.9113f, 0.1075f,  -0.1745f, 0.9653f,
    -0.1517f, 1.5508f,  0.4134f,  -0.0715f, 0.5517f,  -0.3632f,
    -0.1922f, -0.9497f, 0.2503f,  -0.2921f};
inline constexpr float anima_latent_std[16] = {
    2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
    3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

// ---- Z-Image (S3-DiT + Qwen3-4B) constants ---------------------------------
// From Tongyi-MAI/Z-Image-Turbo's shipped configs:
//   transformer/config.json : ZImageTransformer2DModel, dim 3840, n_layers 30,
//       n_heads 30, all_patch_size [2], in_channels 16, cap_feat_dim 2560,
//       axes_dims [32,48,48], rope_theta 256.0, t_scale 1000.0
//   vae/config.json         : AutoencoderKL "flux-dev", latent_channels 16,
//       scaling_factor 0.3611, shift_factor 0.1159 (8x spatial downsample)
//   scheduler/config.json   : FlowMatchEulerDiscreteScheduler, shift 3.0,
//       num_train_timesteps 1000, use_dynamic_shifting false
//   model_index.json        : text_encoder Qwen3Model, tokenizer Qwen2Tokenizer
inline constexpr int zimage_latent_channels = 16;
// Fixed context length of the exported graphs. Z-Image's reference pipeline
// tokenizes with max_sequence_length=512 and pads to it, then drops padded rows
// via the attention mask; a static QNN graph keeps all 512 rows and masks them
// instead, so the padded length IS the graph's context length.
inline constexpr int zimage_text_seq_len = 512;
// Qwen3-4B hidden size == the DiT's cap_feat_dim.
inline constexpr int zimage_text_embedding_size = 2560;
// Flux VAE uses a single scalar shift/scale pair, unlike Wan's per-channel
// table: vae_latent = model_latent / scale + shift (and the inverse to encode).
inline constexpr float zimage_vae_scaling_factor = 0.3611f;
inline constexpr float zimage_vae_shift_factor = 0.1159f;
// Rectified-flow schedule. The DiT's t_scale is 1000, i.e. the timestep it
// consumes is sigma * 1000, not sigma.
inline constexpr float zimage_flow_shift = 3.0f;
inline constexpr float zimage_timestep_scale = 1000.0f;
// S3-DiT patchification: 2x2 latent patches become one token, so a 128x128
// latent yields 64x64 = 4096 image tokens. Captions are padded to a multiple of
// SEQ_MULTI_OF before being positioned.
inline constexpr int zimage_patch_size = 2;
inline constexpr int zimage_seq_multiple = 32;
// 3D RoPE over (t, h, w) — axes_dims [32,48,48], axes_lens [1536,512,512].
inline constexpr int zimage_rope_axes = 3;
inline constexpr int zimage_rope_axis_len_t = 1536;

// Upper bound on how many pieces the DiT may be exported into
// (unet_part1.bin .. unet_partN.bin). 6B parameters do not fit one HTP context
// at any supported weight width, so the split count is a property of the
// converted model, discovered on disk rather than fixed here.
//
// 32 rather than something tighter because the split count is set by the
// *converting* machine's memory, not the phone's: qairt-quantizer holds a whole
// part plus a calibration sample's activations, and was OOM-killed at 19 GB on
// a 4-block part. One block per part is the finest useful split (30 blocks),
// and this has to admit it.
// Discovery stops at the first gap, so this is only a runaway guard. The
// published build has 32 parts plus the caption branch; the headroom is
// deliberate, since a limit that exactly equals the current count turns any
// future re-split into a truncated chain.
inline constexpr int zimage_max_dit_parts = 64;

#endif  // CONFIG_HPP