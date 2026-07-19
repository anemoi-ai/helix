#pragma once
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct mtmd_context;
struct mtmd_input_chunks;
typedef int32_t llama_pos;

namespace helix {

/* RAII wrapper around mtmd_context.
 *
 * Threading: mtmd_tokenize is thread-safe (shared ctx).
 * mtmd_helper_eval_chunks is NOT thread-safe — callers must hold mu() before
 * calling eval_media. */
class MmProj {
public:
    MmProj(const char* mmproj_path,
           const llama_model* text_model,
           bool use_gpu,
           int n_threads = 0);
    ~MmProj();

    /* Capability queries — cheap, no lock needed. */
    bool supports_vision() const { return supports_vision_; }
    bool supports_audio()  const { return supports_audio_;  }

    /* Tokenize the prompt (which contains media markers) together with the
     * given bitmaps, then eval all chunks into the llama context.
     *
     * Must be called with mu() held (or within a single-threaded context).
     *
     * Returns the new n_past value (= total token positions consumed by
     * the prefill, including image-embedding positions).
     * Throws helix::Error on failure. */
    llama_pos eval_media(
        llama_context* lctx,
        const char*    prompt_with_markers,
        bool           add_special,
        const std::vector<std::vector<uint8_t>>& media_bytes,
        llama_pos      n_past,
        int            n_batch);

    std::mutex& mu() { return mu_; }

    mtmd_context* ctx() const { return ctx_; }

    MmProj(const MmProj&) = delete;
    MmProj& operator=(const MmProj&) = delete;

private:
    mtmd_context* ctx_ = nullptr;
    bool          supports_vision_ = false;
    bool          supports_audio_  = false;
    std::mutex    mu_;
};

} // namespace helix
