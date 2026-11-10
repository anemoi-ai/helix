#include "mmproj.hpp"

#include <cstring>
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
        throw_model_load_failed(std::string("failed to load mmproj: ") + mmproj_path,
                                "mmproj_path");
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
            throw_validation("media payload is empty", "messages");
        }
        if (raw.size() > kMaxMediaRawBytes) {
            throw_validation("media payload exceeds "
                             + std::to_string(kMaxMediaRawBytes / (1024 * 1024)) + " MiB limit",
                             "messages");
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
        auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_, raw.data(), raw.size(), false,
                                                        mtmd_helper_init_opt_default());
        if (wrapper.video_ctx) {
            mtmd_helper_video_free(wrapper.video_ctx);
            wrapper.video_ctx = nullptr;
        }

        BitmapPtr bmp(wrapper.bitmap, bitmap_deleter);
        if (!bmp) {
            throw_validation("failed to decode media payload (unsupported format or corrupt data)",
                             "messages");
        }
        bitmaps.push_back(std::move(bmp));
    }

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    auto chunks_guard = std::unique_ptr<mtmd_input_chunks,
                                         decltype(&mtmd_input_chunks_free)>(
        chunks, &mtmd_input_chunks_free);

    /* Value-initialised, and every field set: llama.cpp b12033 added
     * text_len, and mtmd reads exactly that many bytes (no strlen). Left
     * uninitialised it is stack garbage: a huge value throws bad_alloc
     * inside mtmd_tokenize, a wrong one reads past the prompt. */
    mtmd_input_text text_input{};
    text_input.text          = prompt_with_markers;
    text_input.text_len      = std::strlen(prompt_with_markers);
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
        throw_validation("mtmd_tokenize failed (code " + std::to_string(tok_res) +
                         "); check that the number of media parts matches markers in the prompt",
                         "messages");
    }

    /* A media chunk decoded with non-causal attention must fit in one ubatch;
     * past that, llama.cpp's decode is a GGML_ASSERT ("non-causal attention
     * requires n_ubatch >= n_tokens"), which aborts the process. It used to be
     * unreachable for Gemma 4 (images came to <= 268 tokens); llama.cpp b12033
     * stopped letterbox-padding its images (#30922) and a 1344x1008 image now
     * costs 588 tokens. Refuse it here, as a request error, instead. */
    {
        const uint32_t n_ubatch = llama_n_ubatch(lctx);
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        for (size_t i = 0; i < n_chunks; ++i) {
            const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
            if (!mtmd_decode_use_non_causal(ctx_, chunk)) continue;
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk);
            if (n_tok > n_ubatch) {
                throw_validation("a media part needs " + std::to_string(n_tok) +
                                 " tokens in one batch, more than this session's n_ubatch (" +
                                 std::to_string(n_ubatch) + "); send a smaller image or "
                                 "create the session with a larger n_ubatch",
                                 "messages");
            }
        }
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
        throw_backend("mtmd_helper_eval_chunks failed (code " + std::to_string(eval_res) + ")");
    }

    return new_n_past;
}

} // namespace helix
