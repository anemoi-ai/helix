/* state.cpp — session KV-state persistence (1.5).
 *
 * File format (all integers little-endian, fixed-size header written
 * verbatim; bump kFormatVersion on any layout change):
 *
 *   StateHeader
 *   llama_token tokens[n_tokens]     token history for prefix validation
 *   uint8_t     state[state_size]    llama_state_get_data blob
 *
 * SECURITY: the state blob is decoded directly into engine buffers by
 * llama_state_set_data — files are trusted input (documented in helix.h).
 * The header checks (magic, version, model fingerprint, n_ctx) catch
 * mistakes, not malice.
 */

#include "session.hpp"
#include "../model/model.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"

#include "llama.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace helix {

namespace {

constexpr uint32_t kFormatVersion = 1;

struct StateHeader {
    char     magic[4];          /* "HLXS" */
    uint32_t format_version;    /* kFormatVersion */
    uint32_t abi_version;       /* HELIX_ABI_VERSION at save time (info) */
    uint32_t n_ctx;
    uint64_t model_fingerprint; /* Model::state_fingerprint() */
    uint64_t n_tokens;
    uint64_t state_size;
};
static_assert(sizeof(StateHeader) == 40, "StateHeader layout is the file format");

[[noreturn]] void throw_state_mismatch(const std::string& msg) {
    throw Error(HELIX_E_STATE_MISMATCH, msg,
                "state_mismatch", "path", "helix_e_state_mismatch");
}

} // namespace

/* Shared gate: state persistence needs the prefix cache — restored KV is
 * only reachable through a prefix match on the next request. The message
 * names the actual reason the session cannot persist. */
void Session::require_persistable(const char* what) const {
    if (opts_.embedding) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    std::string(what) + " is not supported for embedding "
                    "sessions (KV is cleared every flush; there is no state "
                    "to persist)",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }
    if (opts_.speculative.type != SpeculativeType::None) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    std::string(what) + " is not supported for speculative "
                    "(MTP) sessions in this release",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }
    if (!opts_.prefix_cache) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    std::string(what) + " requires the prefix cache "
                    "(session option \"prefix_cache\": true): restored KV "
                    "state is only used through prefix matching",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }
    if (!active_loras_.empty()) {
        throw Error(HELIX_E_UNSUPPORTED_FEATURE,
                    std::string(what) + " is not supported for sessions with "
                    "active lora adapters in this release (the state file "
                    "does not record the adapter set, so a restore could "
                    "silently mix KV computed under different adapters)",
                    "unsupported_feature_error", "session",
                    "helix_e_unsupported_feature");
    }
}

helix_status_t Session::save_state(const char* path) {
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    std::lock_guard<std::mutex> cuda_guard(model_.cuda_mu());
#endif
    /* Waits for any in-flight request (one inference at a time), then holds
     * the session so the KV cannot change under the serialization. */
    std::lock_guard<std::mutex> guard(mu_);

    require_persistable("helix_session_save");

    const size_t state_size = llama_state_get_size(ctx_);
    std::vector<uint8_t> blob(state_size);
    const size_t copied = llama_state_get_data(ctx_, blob.data(), blob.size());
    if (copied == 0 || copied > state_size) {
        throw Error(HELIX_E_INTERNAL,
                    "llama_state_get_data failed",
                    "internal_error", "", "helix_e_internal");
    }

    StateHeader hdr{};
    std::memcpy(hdr.magic, "HLXS", 4);
    hdr.format_version    = kFormatVersion;
    hdr.abi_version       = HELIX_ABI_VERSION;
    hdr.n_ctx             = llama_n_ctx(ctx_);
    hdr.model_fingerprint = model_.state_fingerprint();
    hdr.n_tokens          = last_tokens_.size();
    hdr.state_size        = copied;

    /* Write to a sibling temp file, then rename into place so a crash or
     * ENOSPC mid-write never leaves a truncated file at `path`. */
    const std::string tmp_path = std::string(path) + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f) {
            throw Error(HELIX_E_INVALID_ARG,
                        "cannot open for writing: " + tmp_path,
                        "invalid_request_error", "path", "helix_e_invalid_arg");
        }
        f.write(reinterpret_cast<const char*>(&hdr), sizeof hdr);
        if (!last_tokens_.empty()) {
            f.write(reinterpret_cast<const char*>(last_tokens_.data()),
                    static_cast<std::streamsize>(
                        last_tokens_.size() * sizeof(llama_token)));
        }
        f.write(reinterpret_cast<const char*>(blob.data()),
                static_cast<std::streamsize>(copied));
        f.flush();
        if (!f) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            throw Error(HELIX_E_INTERNAL,
                        "I/O error writing session state to " + tmp_path,
                        "internal_error", "path", "helix_e_internal");
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::error_code ec2;
        std::filesystem::remove(tmp_path, ec2);
        throw Error(HELIX_E_INTERNAL,
                    "cannot rename " + tmp_path + " into place: " + ec.message(),
                    "internal_error", "path", "helix_e_internal");
    }

    log_info("session state saved: " + std::string(path) + " (" +
             std::to_string(hdr.n_tokens) + " tokens, " +
             std::to_string(copied) + " state bytes)");
    return HELIX_OK;
}

helix_status_t Session::restore_state(const char* path) {
#if HELIX_CUDA_REQUIRES_GLOBAL_LOCK
    std::lock_guard<std::mutex> cuda_guard(model_.cuda_mu());
#endif
    std::lock_guard<std::mutex> guard(mu_);

    require_persistable("helix_session_restore");

    /* All file reading and header validation happens BEFORE the context is
     * touched, so a rejected file leaves existing session state intact. */
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error(HELIX_E_INVALID_ARG,
                    "cannot open session state file: " + std::string(path),
                    "invalid_request_error", "path", "helix_e_invalid_arg");
    }

    StateHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof hdr);
    if (!f || std::memcmp(hdr.magic, "HLXS", 4) != 0) {
        throw_state_mismatch("not a helix session state file: " +
                             std::string(path));
    }
    if (hdr.format_version != kFormatVersion) {
        throw_state_mismatch(
            "session state file format v" + std::to_string(hdr.format_version) +
            " is not supported by this library (expected v" +
            std::to_string(kFormatVersion) + ")");
    }
    if (hdr.model_fingerprint != model_.state_fingerprint()) {
        throw_state_mismatch(
            "session state file was saved with a different model");
    }
    if (hdr.n_ctx != llama_n_ctx(ctx_)) {
        throw_state_mismatch(
            "session state file was saved with n_ctx=" +
            std::to_string(hdr.n_ctx) + " but this session has n_ctx=" +
            std::to_string(llama_n_ctx(ctx_)));
    }
    if (hdr.n_tokens > hdr.n_ctx) {
        throw_state_mismatch("session state file is corrupt "
                             "(token history exceeds n_ctx)");
    }

    std::vector<llama_token> tokens(static_cast<size_t>(hdr.n_tokens));
    if (!tokens.empty()) {
        f.read(reinterpret_cast<char*>(tokens.data()),
               static_cast<std::streamsize>(tokens.size() * sizeof(llama_token)));
    }
    std::vector<uint8_t> blob(static_cast<size_t>(hdr.state_size));
    f.read(reinterpret_cast<char*>(blob.data()),
           static_cast<std::streamsize>(blob.size()));
    if (!f || f.gcount() != static_cast<std::streamsize>(blob.size())) {
        throw_state_mismatch("session state file is truncated: " +
                             std::string(path));
    }

    /* Run the warmup BEFORE loading state: run_warmup clears KV seq 0, so
     * letting the first real request trigger it would silently wipe the
     * restored cells while last_tokens_ still claims they exist (breaking
     * the prefix-cache invariant). */
    maybe_run_warmup();

    /* From here on the context is being overwritten: any failure must leave
     * the session EMPTY and usable, never half-restored. */
    try {
        const size_t read = llama_state_set_data(ctx_, blob.data(), blob.size());
        if (read != blob.size()) {
            throw_state_mismatch(
                "session state blob was rejected by the engine (read " +
                std::to_string(read) + " of " + std::to_string(blob.size()) +
                " bytes)");
        }
    } catch (const Error&) {
        llama_memory_clear(llama_get_memory(ctx_), /*data=*/true);
        last_tokens_.clear();
        throw;
    } catch (const std::exception& e) {
        llama_memory_clear(llama_get_memory(ctx_), /*data=*/true);
        last_tokens_.clear();
        throw_state_mismatch(std::string("engine rejected session state: ") +
                             e.what());
    }

    last_tokens_ = std::move(tokens);

    log_info("session state restored: " + std::string(path) + " (" +
             std::to_string(last_tokens_.size()) + " tokens)");
    return HELIX_OK;
}

} // namespace helix
