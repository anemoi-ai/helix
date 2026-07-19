#include "mmproj.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "image_decode.hpp"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cassert>
#include <memory>
#include <stdexcept>

namespace helix {

MmProj::MmProj(const char* mmproj_path,
               const llama_model* text_model,
               bool use_gpu,
               int n_threads) {

    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu      = use_gpu;
    p.print_timings = false;
    p.n_threads    = n_threads;

    ctx_ = mtmd_init_from_file(mmproj_path, text_model, p);
    if (!ctx_) {
        throw Error(HELIX_E_MODEL_LOAD_FAILED,
                    std::string("failed to load mmproj: ") + mmproj_path,
                    "model_load_failed", "mmproj_path", "helix_e_model_load_failed");
    }

    supports_vision_ = mtmd_support_vision(ctx_);
    supports_audio_  = mtmd_support_audio(ctx_);

    log_info(std::string("mmproj loaded: vision=") +
             (supports_vision_ ? "1" : "0") +
             " audio=" + (supports_audio_ ? "1" : "0"));
}

MmProj::~MmProj() {
    if (ctx_) {
        mtmd_free(ctx_);
        ctx_ = nullptr;
    }
}

llama_pos MmProj::eval_media(
    llama_context* lctx,
    const char*    prompt_with_markers,
    bool           add_special,
    const std::vector<std::vector<uint8_t>>& media_bytes,
    llama_pos      n_past,
    int            n_batch) {

    for (const auto& raw : media_bytes) {
        if (raw.empty()) {
            throw Error(HELIX_E_VALIDATION,
                        "media payload is empty",
                        "invalid_request_error", "messages", "helix_e_validation");
        }
        if (raw.size() > kMaxMediaRawBytes) {
            throw Error(HELIX_E_VALIDATION,
                        "media payload exceeds " + std::to_string(kMaxMediaRawBytes / (1024 * 1024)) + " MiB limit",
                        "invalid_request_error", "messages", "helix_e_validation");
        }
    }

    auto bitmap_deleter = [](mtmd_bitmap* b) { mtmd_bitmap_free(b); };
    using BitmapPtr = std::unique_ptr<mtmd_bitmap, decltype(bitmap_deleter)>;

    std::vector<BitmapPtr> bitmaps;
    bitmaps.reserve(media_bytes.size());
    for (const auto& raw : media_bytes) {
        /* Upstream returns a wrapper carrying an optional video decode context.
         * Helix passes placeholder=false (it needs real decoded media) and frees
         * any video_ctx — it is only populated for video inputs, which Helix does
         * not expose (and MTMD_VIDEO is built OFF), but handle it defensively. */
        auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_, raw.data(), raw.size(), false);
        if (wrapper.video_ctx) {
            mtmd_helper_video_free(wrapper.video_ctx);
            wrapper.video_ctx = nullptr;
        }

        BitmapPtr bmp(wrapper.bitmap, bitmap_deleter);
        if (!bmp) {
            throw Error(HELIX_E_VALIDATION,
                        "failed to decode media payload (unsupported format or corrupt data)",
                        "invalid_request_error", "messages", "helix_e_validation");
        }
        bitmaps.push_back(std::move(bmp));
    }

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    auto chunks_guard = std::unique_ptr<mtmd_input_chunks,
                                         decltype(&mtmd_input_chunks_free)>(
        chunks, &mtmd_input_chunks_free);

    mtmd_input_text text_input;
    text_input.text          = prompt_with_markers;
    text_input.add_special   = add_special;
    text_input.parse_special = true;

    std::vector<const mtmd_bitmap*> bitmaps_c;
    bitmaps_c.reserve(bitmaps.size());
    for (const auto& b : bitmaps) bitmaps_c.push_back(b.get());

    const int32_t tok_res = mtmd_tokenize(
        ctx_,
        chunks,
        &text_input,
        bitmaps_c.data(),
        bitmaps_c.size());

    bitmaps.clear();

    if (tok_res != 0) {
        throw Error(HELIX_E_VALIDATION,
                    "mtmd_tokenize failed (code " + std::to_string(tok_res) +
                    "); check that the number of media parts matches markers in the prompt",
                    "invalid_request_error", "messages", "helix_e_validation");
    }

    /* Eval all chunks into the llama context. */
    llama_pos new_n_past = 0;
    const int32_t eval_res = mtmd_helper_eval_chunks(
        ctx_,
        lctx,
        chunks,
        n_past,
        0 /* seq_id */,
        n_batch,
        true /* logits_last */,
        &new_n_past);

    chunks_guard.reset();

    if (eval_res != 0) {
        throw Error(HELIX_E_BACKEND,
                    "mtmd_helper_eval_chunks failed (code " + std::to_string(eval_res) + ")",
                    "backend_error", "", "helix_e_backend");
    }

    return new_n_past;
}

} // namespace helix
