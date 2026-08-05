#ifndef PIPELINEZIMAGE_HPP
#define PIPELINEZIMAGE_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Config.hpp"
#include "FlowMatchScheduler.hpp"
#include "PipelineQnn.hpp"

// Z-Image (Tongyi-MAI): a 6B Scalable Single-Stream DiT text-to-image stack on
// QNN/HTP at a fixed 1024x1024. Everything runs as QNN context binaries:
//
//   * text encoder (clip.bin): Qwen3-4B.
//       input_embedding[1,512,2560] + attention_mask[1,512] -> context
//       [1,512,2560]. The exporter must tap hidden_states[-2]: Z-Image reads
//       the second-to-last layer, not the final one. The token-embedding
//       lookup stays on the CPU (token_emb.bin), as it does for Anima.
//   * DiT (unet_part1.bin .. unet_partN.bin): 30 blocks, dim 3840, patch 2.
//       6B parameters do not fit one HTP context at any supported weight
//       width, so the model is cut between blocks into N pieces. N is a
//       property of the conversion, discovered on disk — see
//       QnnModel::executeZImageDitFirst for the per-part IO contract.
//   * VAE (vae_decoder.bin / optional vae_encoder.bin): Flux AutoencoderKL,
//       16 latent channels, 8x downsample.
//
// Differences from the Anima DiT that shares this base class:
//   * single-stream. Image and caption tokens occupy ONE residual sequence,
//     in that order, so the caption is folded into `hidden` by part 1 and
//     never re-supplied.
//     What every part does take is the per-token RoPE coordinates and the
//     attention mask: the reference drops padded caption rows, so the caption
//     length — and with it every image token's t coordinate — varies per
//     prompt and cannot be baked into the graph.
//   * Flux VAE scaling is a scalar shift/scale pair, not Wan's per-channel
//     mean/std table.
//   * the DiT's t_scale is 1000, so the value handed to the graph is
//     sigma * 1000 (see FlowMatchScheduler's timestep_scale).
//   * Turbo is distilled to 8 steps and trained guidance-free, so the shared
//     generate() loop's cfg == 1.0 fast path (one DiT chain per step instead
//     of two) is the normal case rather than an optimization.
//
// Memory behaves as it does for Anima: `lowram` loads and releases each stage
// around its use, keeping the DiT parts resident together for the whole
// denoising loop under one shared HTP spill-fill buffer. `seq_dit` goes
// further and holds a single part at a time, reloading each one every step —
// far slower, but peak memory becomes one part instead of the whole DiT, which
// is what makes a 6B model reachable at all on a 12-16GB device.
//
// Ultrafix is not supported (tiling a 6B DiT is hopeless); img2img and inpaint
// work whenever vae_encoder.bin is present.
// Breadcrumbs in this file log at ERROR on purpose. The backend now runs with
// the QNN log level at "error", because at INFO the DSP layer emits thousands
// of lines per context into a pipe the app has to drain -- and a pipe nobody
// drains fast enough blocks the WRITER, which is the backend, mid-load, alive,
// making no progress and reporting nothing. These lines are progress rather
// than failures, but they have to outrank the flood to survive it.
class PipelineZImage : public PipelineQnn {
 public:
  PipelineZImage(TextEncoder &text_encoder, const std::string &model_dir,
                 std::vector<std::string> clip_part_paths,
                 std::vector<std::string> dit_part_paths,
                 std::string cap_part_path, std::string vae_decoder_path,
                 std::string vae_encoder_path, bool lowram, bool seq_dit)
      : PipelineQnn(text_encoder, model_dir, /*sdxl=*/false,
                    /*use_v_pred=*/false),
        clip_part_paths_(std::move(clip_part_paths)),
        dit_part_paths_(std::move(dit_part_paths)),
        cap_part_path_(std::move(cap_part_path)),
        vae_decoder_path_(std::move(vae_decoder_path)),
        vae_encoder_path_(std::move(vae_encoder_path)),
        lowram_(lowram),
        // Reloading a part per step only makes sense when we are already
        // releasing models per stage; ignore it in resident mode.
        seq_dit_(lowram && seq_dit) {
    if (dit_part_paths_.empty())
      throw std::runtime_error("zimage: no DiT part binaries given");
    if (cap_part_path_.empty())
      throw std::runtime_error("zimage: no DiT caption branch binary given");
    if (clip_part_paths_.empty())
      throw std::runtime_error("zimage: no text encoder binaries given");
    dit_parts_.resize(dit_part_paths_.size());
    clip_parts_.resize(clip_part_paths_.size());
  }

  bool initialize() override {
    if (lowram_) {
      QNN_INFO(
          "[lowram] Z-Image low-RAM mode: skipping pre-load of text "
          "encoder / DiT / VAE (%zu DiT parts, seq_dit=%d)",
          dit_part_paths_.size(), seq_dit_ ? 1 : 0);
      return true;
    }

    vae_decoder_ = qnn_runtime::createModel(vae_decoder_path_, "vae_decoder");
    if (!vae_decoder_) {
      QNN_ERROR("Failed to create Z-Image VAE decoder.");
      return false;
    }
    for (size_t i = 0; i < clip_part_paths_.size(); ++i) {
      clip_parts_[i] =
          qnn_runtime::createModel(clip_part_paths_[i], clipTag(i).c_str());
      if (!clip_parts_[i]) {
        QNN_ERROR("Failed to create Z-Image text encoder part %zu.", i + 1);
        return false;
      }
    }
    for (size_t i = 0; i < dit_part_paths_.size(); ++i) {
      dit_parts_[i] =
          qnn_runtime::createModel(dit_part_paths_[i], partTag(i).c_str());
      if (!dit_parts_[i]) {
        QNN_ERROR("Failed to create Z-Image DiT part %zu.", i + 1);
        return false;
      }
    }
    cap_part_ = qnn_runtime::createModel(cap_part_path_, "unet_cap");
    if (!cap_part_) {
      QNN_ERROR("Failed to create the Z-Image DiT caption branch.");
      return false;
    }
    if (!vae_encoder_path_.empty()) {
      vae_encoder_ = qnn_runtime::createModel(vae_encoder_path_, "vae_encoder");
      if (!vae_encoder_) QNN_WARN("Failed create Z-Image QNN VAE encoder.");
    } else {
      QNN_INFO("img2img disabled: Z-Image VAE encoder not loaded");
    }

    // Every resident context executes strictly in sequence (text encoder ->
    // part 1..N -> VAE) and never concurrently, so one HTP spill-fill scratch
    // buffer serves them all instead of one multi-GB allocation each. The
    // group HEAD owns the buffer and must outlive every reference to it.
    // vae_decoder_ is a PipelineQnn member, so it is destroyed after all of
    // this class's members and after vae_encoder_ (declared later in the base,
    // destroyed first) — which makes it the only safe head here.
    const uint64_t sf_bytes = spillFillGroupBytes();
    Qnn_ContextHandle_t head = nullptr;
    if (sf_bytes)
      QNN_INFO("[spill-fill] Z-Image context group sharing enabled: %llu bytes",
               (unsigned long long)sf_bytes);

    vae_decoder_->setSpillFillGroup(sf_bytes, nullptr);
    if (qnn_runtime::initializeApp("VAEDecoder", vae_decoder_) != EXIT_SUCCESS)
      return false;
    if (sf_bytes) head = vae_decoder_->getContextHandle();

    for (size_t i = 0; i < clip_parts_.size(); ++i) {
      clip_parts_[i]->setSpillFillGroup(sf_bytes, head);
      if (qnn_runtime::initializeApp(clipTag(i).c_str(), clip_parts_[i]) !=
          EXIT_SUCCESS)
        return false;
    }

    for (size_t i = 0; i < dit_parts_.size(); ++i) {
      dit_parts_[i]->setSpillFillGroup(sf_bytes, head);
      if (qnn_runtime::initializeApp(partTag(i).c_str(), dit_parts_[i]) !=
          EXIT_SUCCESS)
        return false;
    }
    cap_part_->setSpillFillGroup(sf_bytes, head);
    if (qnn_runtime::initializeApp("unet_cap", cap_part_) != EXIT_SUCCESS)
      return false;
    if (vae_encoder_) {
      vae_encoder_->setSpillFillGroup(sf_bytes, head);
      if (qnn_runtime::initializeApp("VAEEncoder", vae_encoder_) !=
          EXIT_SUCCESS)
        return false;
    }

    QNN_INFO("Z-Image QNN pipeline initialized (text encoder + %zu DiT parts "
             "+ VAE).",
             dit_parts_.size());
    return true;
  }

  bool supportsImg2Img() const override {
    return lowram_ ? !vae_encoder_path_.empty() : vae_encoder_ != nullptr;
  }
  bool isZImage() const override { return true; }

 protected:
  // A per-step preview would cost a VAE decoder load/release per step under
  // lowram; disable it there, as SDXL and Anima do.
  bool previewSupported() const override { return !lowram_; }
  bool vaeTilingSupported() const override { return false; }
  int vaeTilePixelSize() const override { return 1024; }

  // --- generalization hooks ---
  int latentChannels() const override { return zimage_latent_channels; }
  int textSeqLen() const override { return zimage_text_seq_len; }
  int textHiddenDim() const override { return zimage_text_embedding_size; }
  // Not a pooled embedding: Z-Image has none. The slot carries the 512-entry
  // caption attention mask, which the DiT needs at every step and which the
  // prompt cache therefore has to persist next to the hidden states.
  int textPooledDim() const override { return zimage_text_seq_len; }
  bool promptCacheSupported() const override { return true; }

  std::unique_ptr<Scheduler> makeScheduler(const GenerationRequest &req,
                                           const char *) override {
    // Z-Image ships FlowMatchEulerDiscreteScheduler (deterministic, shift 3.0);
    // "euler_a" opts into the ancestral variant instead.
    const bool ancestral = (req.scheduler_type == "euler_a" ||
                            req.scheduler_type == "eulera" ||
                            req.scheduler_type == "euler_ancestral");
    return std::make_unique<FlowMatchScheduler>(
        zimage_flow_shift, /*multiplier=*/1.0f, /*eta=*/ancestral ? 1.0f : 0.0f,
        /*s_noise=*/1.0f, /*timestep_scale=*/zimage_timestep_scale,
        FlowMatchScheduler::SigmaSchedule::kDiffusersLinear);
  }

  // model latent -> VAE latent (Flux AutoencoderKL): x / scale + shift.
  void latentsToVae(xt::xarray<float> &latents) const override {
    latents = xt::eval(latents * (1.0f / zimage_vae_scaling_factor) +
                       zimage_vae_shift_factor);
  }
  // VAE latent -> model latent (img2img): (x - shift) * scale.
  void vaeToLatents(xt::xarray<float> &latents) const override {
    latents = xt::eval((latents - zimage_vae_shift_factor) *
                       zimage_vae_scaling_factor);
  }

  void encodeText(const ProcessedPromptPair &prompts, bool need_negative,
                  bool need_positive, Conditioning &cond) override {
    if (lowram_ && !seqClip()) loadClipIfNeeded();
    if (!seqClip() && (clip_parts_.empty() || !clip_parts_.front()))
      throw std::runtime_error("Z-Image text encoder not initialized!");
    if (need_negative)
      runTextEncoder(prompts.negative_embeddings, prompts.negative_qwen_mask,
                     cond.negHidden(), cond.negPooled());
    if (need_positive)
      runTextEncoder(prompts.positive_embeddings, prompts.positive_qwen_mask,
                     cond.posHidden(), cond.posPooled());
    if (lowram_) releaseClip();
  }

  void vaeEncode(const GenerationRequest &, const float *image, float *mean,
                 float *std_dev) override {
    if (lowram_) loadVaeEncoderIfNeeded();
    if (!vae_encoder_) throw std::runtime_error("Z-Image VAE encoder missing");
    if (StatusCode::SUCCESS !=
        vae_encoder_->executeZImageVaeEncoder(image, mean, std_dev))
      throw std::runtime_error("Z-Image VAE encode failed");
  }

  void beginDenoise(const GenerationRequest &) override {
    if (!lowram_) return;
    // The encode stage is over once the DiT is needed; never hold both.
    releaseVaeEncoder();
    // seq_dit loads and releases each part inside every step, so there is
    // nothing to pre-load.
    if (!seq_dit_) loadDitPartsIfNeeded();
  }

  void runUnetStep(const GenerationRequest &, const float *latents_batch2,
                   float timestep, bool skip_uncond, Conditioning &cond,
                   float *out_batch2) override {
    const int single =
        zimage_latent_channels * sample_width * sample_height;

    if (!skip_uncond)
      runDitChain(latents_batch2, timestep, cond.negHidden(), cond.negPooled(),
                  out_batch2);
    runDitChain(latents_batch2 + single, timestep, cond.posHidden(),
                cond.posPooled(), out_batch2 + single);
  }

  void endDenoise() override {
    if (!lowram_) return;
    releaseDitParts();
  }

  void vaeDecode(const GenerationRequest &, const float *latents,
                 float *pixels) override {
    reportSub("Decoding image");
    if (lowram_ && !vae_decoder_) {
      vae_decoder_ =
          qnn_runtime::createAndInitModel(vae_decoder_path_, "vae_decoder");
      QNN_INFO("[lowram] Z-Image VAE decoder loaded");
    }
    if (!vae_decoder_) throw std::runtime_error("Z-Image VAE decoder missing");
    if (StatusCode::SUCCESS !=
        vae_decoder_->executeZImageVaeDecoder(latents, pixels))
      throw std::runtime_error("Z-Image VAE decode failed");
    // Stays loaded for the rest of the decode stage; released by
    // releaseTransientModels() when generate() exits.
  }

  // Catch-all for lowram: release whatever stage model is still loaded when
  // generate() exits (normal return or exception).
  void releaseTransientModels() override {
    if (!lowram_) return;
    releaseClip();
    releaseDitParts();
    if (vae_decoder_) {
      vae_decoder_.reset();
      QNN_INFO("[lowram] Z-Image VAE decoder released");
    }
    releaseVaeEncoder();
  }

 private:

  // Number of image tokens: the DiT folds each 2x2 latch of latent into one
  // token, so a 128x128 latent becomes a 64x64 token grid.
  int imageTokens() const {
    return (sample_width / zimage_patch_size) *
           (sample_height / zimage_patch_size);
  }
  int totalTokens() const { return zimage_text_seq_len + imageTokens(); }

  // Builds the 3D RoPE coordinates and the two masks for one prompt side.
  //
  // Verified bit-exact against the reference pipeline (see docs/zimage.md §9).
  // Three things have to line up, and all three are easy to get wrong:
  //
  //  * ORDER. Basic mode concatenates [image, caption] — image tokens first.
  //  * cap_len = round-up(true_len, 32) is where the reference stops. Slots
  //    beyond it never exist there, so they must be masked OUT of attention.
  //    Slots between true_len and cap_len DO exist there (as a learned pad
  //    token) and must stay IN.
  //  * image tokens sit at t = cap_len + 1, so their RoPE coordinate moves
  //    with the prompt length. That is why none of this can be baked in.
  void buildPositions(const float *mask, std::vector<int32_t> &pos_ids,
                      std::vector<float> &attn_mask,
                      std::vector<float> &cap_pad_mask) const {
    int true_len = 0;
    for (int i = 0; i < zimage_text_seq_len; ++i)
      if (mask[i] > 0.5f) ++true_len;
    int cap_len = ((true_len + zimage_seq_multiple - 1) / zimage_seq_multiple) *
                  zimage_seq_multiple;
    if (cap_len > zimage_text_seq_len) cap_len = zimage_text_seq_len;

    const int img = imageTokens();
    const int tokens = totalTokens();
    pos_ids.assign((size_t)tokens * zimage_rope_axes, 0);
    attn_mask.assign(tokens, 0.0f);
    cap_pad_mask.assign(zimage_text_seq_len, 0.0f);

    // Image block first: one t plane at cap_len + 1, indexed by (h, w).
    const int grid_h = sample_height / zimage_patch_size;
    const int grid_w = sample_width / zimage_patch_size;
    const int t_img = cap_len + 1;
    if (t_img >= zimage_rope_axis_len_t)
      QNN_WARN("zimage: image t coordinate %d exceeds the RoPE table (%d)",
               t_img, zimage_rope_axis_len_t);
    for (int r = 0; r < grid_h; ++r) {
      for (int c = 0; c < grid_w; ++c) {
        const int tok = r * grid_w + c;
        const size_t o = (size_t)tok * zimage_rope_axes;
        pos_ids[o + 0] = t_img;
        pos_ids[o + 1] = r;
        pos_ids[o + 2] = c;
        attn_mask[tok] = 1.0f;
      }
    }

    // Caption block: t = 1..cap_len over the slots the reference would have
    // created, (0,0,0) and masked off beyond.
    for (int i = 0; i < zimage_text_seq_len; ++i) {
      const int tok = img + i;
      if (i < cap_len) {
        pos_ids[(size_t)tok * zimage_rope_axes] = i + 1;
        attn_mask[tok] = 1.0f;
      }
      // Rows past the real prompt are replaced by the DiT's learned pad token.
      cap_pad_mask[i] = (i >= true_len) ? 1.0f : 0.0f;
    }
  }

  // Refines the caption tokens, unless the last run already did it for this
  // prompt. Nothing in this graph depends on the timestep, so at cfg 1.0 --
  // the setting Turbo is distilled for -- it runs once for the whole
  // generation instead of once per step.
  //
  // The cache key is the graph's own inputs, compared byte for byte, rather
  // than a "has the prompt changed" flag. The pipeline object outlives a
  // generation, and a flag that is right for every path through generate()
  // today is one edit away from silently reusing another prompt's caption.
  // cap_pad_mask determines the true prompt length, which is what selects the
  // caption slice of pos_ids and attn_mask, so those need no separate key.
  // The comparison is ~5 MB against a step measured in seconds.
  void ensureCaption(const float *context) {
    const size_t ctx_n =
        (size_t)zimage_text_seq_len * zimage_text_embedding_size;
    const size_t pad_n = (size_t)zimage_text_seq_len;
    const bool hit =
        !cap_.empty() && cap_key_.size() == ctx_n + pad_n &&
        memcmp(cap_key_.data(), context, ctx_n * sizeof(float)) == 0 &&
        memcmp(cap_key_.data() + ctx_n, cap_pad_mask_.data(),
               pad_n * sizeof(float)) == 0;
    if (hit) return;

    reportSub("Caption branch");
    if (seq_dit_) loadCapPartAlone();
    if (!cap_part_)
      throw std::runtime_error("Z-Image DiT caption branch not loaded");
    const StatusCode st = cap_part_->executeZImageDitCaption(
        context, pos_ids_.data(), attn_mask_.data(), cap_pad_mask_.data(),
        (size_t)totalTokens(), cap_);
    if (seq_dit_) cap_part_.reset();
    if (st != StatusCode::SUCCESS) {
      // Leave no half-valid cache behind: cap_ may hold the previous prompt's
      // result, and the key must never outlive the value it describes.
      cap_key_.clear();
      throw std::runtime_error("Z-Image DiT caption branch failed");
    }

    cap_key_.resize(ctx_n + pad_n);
    std::copy(context, context + ctx_n, cap_key_.begin());
    std::copy(cap_pad_mask_.begin(), cap_pad_mask_.end(),
              cap_key_.begin() + ctx_n);
  }

  // One full pass of the DiT for a single CFG branch: the caption branch and
  // part 1 fold the prompt and the noised latent into the residual stream,
  // each later part advances it, and the terminal part emits the flow velocity.
  void runDitChain(const float *sample, float timestep, const float *context,
                   const float *mask, float *out) {
    if (!mask) throw std::runtime_error("Z-Image conditioning has no mask");
    buildPositions(mask, pos_ids_, attn_mask_, cap_pad_mask_);
    ensureCaption(context);
    const size_t tokens = (size_t)totalTokens();

    const size_t n = dit_part_paths_.size();
    for (size_t i = 0; i < n; ++i) {
      // Once per graph, ~33 times per step. This is the one that ticks
      // steadily and tells the user the device has not hung.
      reportSub("DiT", (int)i, (int)n);
      if (seq_dit_) loadDitPartAlone(i);
      QnnModel *part = dit_parts_[i].get();
      if (!part)
        throw std::runtime_error("Z-Image DiT part " + std::to_string(i + 1) +
                                 " not loaded");

      // Only the part that actually declares `out_sample` may write the
      // velocity; asking a mid-chain part for it would silently truncate the
      // stream, so the buffer is handed over solely to the terminal part.
      const bool terminal = part->zimageGraphIsTerminal();
      if (terminal && i + 1 != n)
        QNN_WARN("zimage: DiT part %zu is terminal but %zu parts were found; "
                 "the remaining parts will not run",
                 i + 1, n);

      StatusCode st =
          (i == 0) ? part->executeZImageDitFirst(
                         sample, timestep, cap_, pos_ids_.data(),
                         attn_mask_.data(), tokens, dit_state_,
                         terminal ? out : nullptr)
                   : part->executeZImageDitNext(dit_state_, cap_,
                                                pos_ids_.data(),
                                                attn_mask_.data(), tokens,
                                                terminal ? out : nullptr);
      if (seq_dit_) releaseDitPart(i);
      if (st != StatusCode::SUCCESS)
        throw std::runtime_error("Z-Image DiT part " + std::to_string(i + 1) +
                                 " failed");
      if (terminal) return;
    }
    throw std::runtime_error(
        "Z-Image DiT chain ended without a terminal part (no graph exposes an "
        "'out_sample' output)");
  }

  void runTextEncoder(const std::vector<float> &input_embedding,
                      const std::vector<float> &mask, float *out_hidden,
                      float *out_mask) {
    if (!out_mask)
      throw std::runtime_error("Z-Image conditioning has no mask slot");
    if ((int)mask.size() != zimage_text_seq_len)
      throw std::runtime_error("Z-Image attention mask has the wrong length");
    // Qwen3-4B is split across contexts for the same reason the DiT is: 3.5 B
    // parameters cannot be quantized in one piece on any machine this converts
    // on. Unlike the DiT this costs almost nothing at runtime -- the chain runs
    // once per prompt and the result is prompt-cached, rather than once per
    // step. Part 1 takes the embeddings, every later part takes the previous
    // part's hidden state, and the last emits `context`.
    const size_t elems =
        (size_t)zimage_text_seq_len * zimage_text_embedding_size;
    if (clip_state_.size() != elems) clip_state_.assign(elems, 0.0f);
    for (size_t i = 0; i < clip_parts_.size(); ++i) {
      reportSub("Text encoder", (int)i, (int)clip_parts_.size());
      // Loading one encoder context at a time halves the peak, and on paper it
      // is the obvious answer to six 620 MB contexts co-resident. On the
      // device it made things strictly worse: the backend died the instant
      // generation started, where holding them all had at least reached part 3
      // before being killed. So it is OFF by default and kept behind a switch
      // rather than deleted -- the code is right, something about
      // create/release churn on this HTP is not, and that is worth being able
      // to re-test without a rebuild.
      if (seqClip()) loadClipPartAlone(i);
      auto &part = clip_parts_[i];
      if (!part)
        throw std::runtime_error("Z-Image text encoder part " +
                                 std::to_string(i + 1) + " not loaded");
      const bool last = (i + 1 == clip_parts_.size());
      // The chain has to END on the part that emits `context`. If the model
      // directory is missing trailing parts -- an interrupted download, a
      // hand-assembled directory -- discovery stops at the first gap and the
      // chain looks complete, but its last graph emits `hidden`: a mid-stack
      // activation that would be handed to the DiT as if it were the encoder's
      // output. Nothing downstream can tell the difference, so check here.
      if (last && !part->zimageClipGraphIsTerminal())
        throw std::runtime_error(
            "Z-Image text encoder chain ends at part " + std::to_string(i + 1) +
            " of " + std::to_string(clip_parts_.size()) +
            ", which emits 'hidden' rather than 'context' — the model "
            "directory is missing later clip_part*.bin files");
      // The last part writes straight into the caller's buffer; the rest hand
      // off through clip_state_.
      float *dst = last ? out_hidden : clip_state_.data();
      const StatusCode st =
          (i == 0) ? part->executeZImageTextEncoder(input_embedding.data(),
                                                    mask.data(), dst)
                   : part->executeZImageClipNext(clip_state_.data(),
                                                 mask.data(), dst);
      // Released before the failure check, so a part that fails does not stay
      // resident while the exception unwinds past the rest of the chain.
      if (seqClip()) {
        clip_parts_[i].reset();
        QNN_ERROR("[clip-seq] part %zu released", i + 1);
      }
      if (st != StatusCode::SUCCESS)
        throw std::runtime_error("Z-Image text encoder part " +
                                 std::to_string(i + 1) + " failed");
    }
    // The DiT consumes the same mask at every step, and the prompt cache
    // persists it from this slot, so it has to be written whether or not the
    // encoder ran.
    std::copy(mask.begin(), mask.end(), out_mask);
  }

  std::string partTag(size_t i) const {
    return "unet_part" + std::to_string(i + 1);
  }

  std::string clipTag(size_t i) const {
    return "clip_part" + std::to_string(i + 1);
  }

  // ---- lowram stage (un)loading --------------------------------------------
  // The whole chain is resident together rather than one part at a time. In
  // lowram this runs with the DiT and VAE already released, so the encoder has
  // the device to itself, and it executes once per prompt -- reloading a
  // context per part would pay the load cost for no memory that is needed
  // elsewhere at that moment.
  void loadClipIfNeeded() {
    if (!clip_parts_.empty() && clip_parts_.front()) return;
    // NO spill-fill group for the encoder, and that is a memory decision, not
    // an omission. The group size (601 MB) was measured for a 4608-token DiT
    // graph; these are 512-token encoder graphs whose natural scratch is a
    // couple of orders of magnitude smaller. Requesting the group reserves the
    // full 601 MB up front -- and if group registration fails quietly, six
    // contexts reserve it EACH, which is 3.6 GB of scratch for graphs that
    // need none of it, on top of their 3.7 GB of weights. That is the
    // difference between an encoder stage that fits a 16 GB phone and the
    // swap-thrash freeze at "Loading text encoder 3/6".
    const uint64_t sf_bytes = 0;
    Qnn_ContextHandle_t head = nullptr;
    for (size_t i = 0; i < clip_part_paths_.size(); ++i) {
      // 3.6 GB of context binaries, and until this loop finishes nothing has
      // reported progress even once. Say which one is being mapped -- and
      // leave a breadcrumb in the log, because this loop is where the device
      // wedged twice (at the 4th context, ~1.9 GB mapped: a per-process DSP
      // mapping ceiling, not swap pressure).
      QNN_ERROR("[clip] context %zu/%zu (co-resident)", i + 1,
                clip_part_paths_.size());
      reportSub("Loading text encoder", (int)i, (int)clip_part_paths_.size());
      clip_parts_[i] =
          qnn_runtime::createModel(clip_part_paths_[i], clipTag(i).c_str());
      if (!clip_parts_[i])
        throw std::runtime_error("[lowram] Failed to create Z-Image text "
                                 "encoder part " + std::to_string(i + 1));
      clip_parts_[i]->setSpillFillGroup(sf_bytes, head);
      if (qnn_runtime::initializeApp(clipTag(i).c_str(), clip_parts_[i]) !=
          EXIT_SUCCESS)
        throw std::runtime_error("[lowram] Failed to init Z-Image text encoder "
                                 "part " + std::to_string(i + 1));
      if (i == 0 && sf_bytes) head = clip_parts_[0]->getContextHandle();
    }
    QNN_INFO("[lowram] Z-Image text encoder loaded (%zu part(s))",
             clip_parts_.size());
  }
  void releaseClip() {
    if (clip_parts_.empty() || !clip_parts_.front()) return;
    // Reverse order: every group reference dies before its head (part 1).
    for (size_t i = clip_parts_.size(); i-- > 0;) clip_parts_[i].reset();
    clip_state_.clear();
    clip_state_.shrink_to_fit();
    if (lowram_) QNN_INFO("[lowram] Z-Image text encoder released");
  }

  // The parts run strictly in sequence (1 -> N) every step and stay resident
  // for the whole denoising loop, so they share one spill-fill buffer: part 1
  // is the group head, the rest reference it. It is created first and released
  // last.
  void loadDitPartsIfNeeded() {
    if (dit_parts_[0]) return;
    const uint64_t sf_bytes = spillFillGroupBytes();
    if (sf_bytes)
      QNN_INFO("[lowram] Z-Image DiT part group sharing enabled: %llu bytes",
               (unsigned long long)sf_bytes);

    Qnn_ContextHandle_t head = nullptr;
    // The single longest silent stretch in a generation: 12.9 GB of context
    // binaries, all mapped before the first denoising step can start.
    const int n_load = (int)dit_parts_.size() + 1;      // + the caption branch
    for (size_t i = 0; i < dit_parts_.size(); ++i) {
      reportSub("Loading DiT", (int)i, n_load);
      dit_parts_[i] =
          qnn_runtime::createModel(dit_part_paths_[i], partTag(i).c_str());
      if (!dit_parts_[i])
        throw std::runtime_error("[lowram] Failed to create Z-Image DiT part " +
                                 std::to_string(i + 1));
      dit_parts_[i]->setSpillFillGroup(sf_bytes, head);
      if (qnn_runtime::initializeApp(partTag(i).c_str(), dit_parts_[i]) !=
          EXIT_SUCCESS)
        throw std::runtime_error("[lowram] Failed to init Z-Image DiT part " +
                                 std::to_string(i + 1));
      if (i == 0 && sf_bytes) head = dit_parts_[0]->getContextHandle();
    }
    // The caption branch joins the same group. It executes before part 1 and
    // never alongside it, so it shares the scratch buffer like everything else;
    // it is created after the head and released before it.
    reportSub("Loading DiT", n_load - 1, n_load);
    cap_part_ = qnn_runtime::createModel(cap_part_path_, "unet_cap");
    if (!cap_part_)
      throw std::runtime_error(
          "[lowram] Failed to create the Z-Image DiT caption branch");
    cap_part_->setSpillFillGroup(sf_bytes, head);
    if (qnn_runtime::initializeApp("unet_cap", cap_part_) != EXIT_SUCCESS)
      throw std::runtime_error(
          "[lowram] Failed to init the Z-Image DiT caption branch");
    QNN_INFO("[lowram] Z-Image DiT parts loaded (%zu + caption)",
             dit_parts_.size());
  }
  void releaseDitParts() {
    bool any = cap_part_ != nullptr;
    cap_part_.reset();
    // Reverse order: every group reference dies before its head (part 1).
    for (size_t i = dit_parts_.size(); i-- > 0;) {
      if (!dit_parts_[i]) continue;
      dit_parts_[i].reset();
      any = true;
    }
    // The refined caption outlives no DiT context: dropping it here is what
    // makes the next generation recompute it against whatever graphs are
    // loaded then, rather than trust a buffer from a released one.
    cap_.clear();
    cap_.shrink_to_fit();
    cap_key_.clear();
    cap_key_.shrink_to_fit();
    if (any && lowram_) QNN_INFO("[lowram] Z-Image DiT parts released");
  }

  // seq_dit: one part resident at a time, each with its own spill-fill buffer
  // (no group sharing — the parts are never co-resident).
  void loadDitPartAlone(size_t i) {
    if (dit_parts_[i]) return;
    dit_parts_[i] =
        qnn_runtime::createAndInitModel(dit_part_paths_[i], partTag(i).c_str());
    if (!dit_parts_[i])
      throw std::runtime_error("[seq_dit] Failed to load Z-Image DiT part " +
                               std::to_string(i + 1));
  }
  void releaseDitPart(size_t i) {
    if (!dit_parts_[i]) return;
    dit_parts_[i].reset();
  }
  // Opt-in: LOCALDREAM_ZIMAGE_SEQ_CLIP=1. Read once, because it is consulted
  // per encoder part.
  static bool seqClip() {
    static const bool on = [] {
      const char *e = getenv("LOCALDREAM_ZIMAGE_SEQ_CLIP");
      return e && *e == '1';
    }();
    return on;
  }

  // seq mode: one encoder context resident at a time, with its own spill-fill
  // buffer (no group sharing -- the parts are never co-resident).
  void loadClipPartAlone(size_t i) {
    if (clip_parts_[i]) return;
    // Breadcrumbs on both sides: if the process dies or wedges in here, the
    // error tail's last line names the exact context and phase instead of
    // whatever QNN happened to print last.
    QNN_ERROR("[clip-seq] part %zu/%zu: creating context", i + 1,
              clip_part_paths_.size());
    clip_parts_[i] =
        qnn_runtime::createAndInitModel(clip_part_paths_[i], clipTag(i).c_str());
    if (!clip_parts_[i])
      throw std::runtime_error("[seq] Failed to load Z-Image text encoder part " +
                               std::to_string(i + 1));
    QNN_ERROR("[clip-seq] part %zu ready", i + 1);
  }
  void loadCapPartAlone() {
    if (cap_part_) return;
    cap_part_ = qnn_runtime::createAndInitModel(cap_part_path_, "unet_cap");
    if (!cap_part_)
      throw std::runtime_error(
          "[seq_dit] Failed to load the Z-Image DiT caption branch");
  }

  void loadVaeEncoderIfNeeded() {
    if (vae_encoder_) return;
    if (vae_encoder_path_.empty())
      throw std::runtime_error("[lowram] Z-Image VAE encoder path missing");
    vae_encoder_ =
        qnn_runtime::createAndInitModel(vae_encoder_path_, "vae_encoder");
    QNN_INFO("[lowram] Z-Image VAE encoder loaded");
  }
  void releaseVaeEncoder() {
    if (!vae_encoder_) return;
    vae_encoder_.reset();
    if (lowram_) QNN_INFO("[lowram] Z-Image VAE encoder released");
  }

  // Shared spill-fill buffer size (bytes) for the Z-Image context group. A DiT
  // part's requirement depends on where the conversion cut the model, so the
  // default is a starting point: override with
  // LOCALDREAM_ZIMAGE_SPILL_FILL_BYTES (context creation failures log the real
  // requirement as "...smaller than required spill-fill size N"; take the max
  // across parts). 0 disables sharing.
  static uint64_t spillFillGroupBytes() {
    const char *e = getenv("LOCALDREAM_ZIMAGE_SPILL_FILL_BYTES");
    if (e && *e) return strtoull(e, nullptr, 10);
    return 601096192ULL;
  }

  const std::vector<std::string> clip_part_paths_;
  const std::vector<std::string> dit_part_paths_;
  const std::string cap_part_path_;
  const std::string vae_decoder_path_;
  const std::string vae_encoder_path_;
  const bool lowram_;
  const bool seq_dit_;

  std::vector<std::unique_ptr<QnnModel>> clip_parts_;
  // Residual stream handed between text encoder parts, [1, S, D]. Held here
  // rather than on the stack: it is 5 MB and encodeText runs per prompt.
  std::vector<float> clip_state_;
  std::vector<std::unique_ptr<QnnModel>> dit_parts_;
  // The caption branch and its result, [1, S, dim]. Held across steps: nothing
  // in that graph depends on the timestep. cap_key_ is the concatenation of the
  // inputs it was computed from (context ++ cap_pad_mask); see ensureCaption.
  std::unique_ptr<QnnModel> cap_part_;
  std::vector<float> cap_;
  std::vector<float> cap_key_;
  // {hidden, emb} handed from one part to the next, reused across steps.
  std::vector<std::vector<float>> dit_state_;
  // Rebuilt per CFG branch (the two sides can have different prompt lengths),
  // reused across steps and parts.
  std::vector<int32_t> pos_ids_;
  std::vector<float> attn_mask_;
  // Only part 1 embeds the caption, so only it needs the pad-token mask.
  std::vector<float> cap_pad_mask_;
};

#endif  // PIPELINEZIMAGE_HPP
