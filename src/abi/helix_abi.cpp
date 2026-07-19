/* helix_abi.cpp — Every extern "C" function exported by libhelix.
 *
 * Each function follows the same pattern:
 *   1. clear_last_error()
 *   2. validate non-null pointer args
 *   3. call into the C++ implementation
 *   4. catch all exceptions → translate to helix_status_t + set_last_error
 */

#include "helix.h"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "../internal/version.hpp"
#include "../runtime/runtime.hpp"
#include "../model/model.hpp"
#include "../session/session.hpp"

#include <cstdlib>
#include <cstring>
#include <new>

/* The helix_status_t size invariant (sizeof == 4) is asserted in helix.h,
 * which this TU includes, so it is verified at every build without a duplicate
 * assertion here. */

/* ------------------------------------------------------------------ */
/*  Exception catch wrapper                                            */
/* ------------------------------------------------------------------ */

#define HELIX_TRY           \
    helix::clear_last_error(); \
    try {

#define HELIX_CATCH                                                         \
    } catch (const helix::Error& e) {                                       \
        helix::set_last_error(e);                                            \
        return e.status;                                                     \
    } catch (const std::bad_alloc&) {                                       \
        helix::set_last_error(HELIX_E_OOM, "out of memory",                 \
            "memory_error", "", "helix_e_oom");                              \
        return HELIX_E_OOM;                                                  \
    } catch (const std::exception& e) {                                     \
        helix::set_last_error(HELIX_E_INTERNAL, e.what(),                   \
            "internal_error", "", "helix_e_internal");                       \
        return HELIX_E_INTERNAL;                                             \
    } catch (...) {                                                          \
        helix::set_last_error(HELIX_E_INTERNAL, "unknown exception",        \
            "internal_error", "", "helix_e_internal");                       \
        return HELIX_E_INTERNAL;                                             \
    }

/* ------------------------------------------------------------------ */
/*  Versioning                                                         */
/* ------------------------------------------------------------------ */

extern "C" uint32_t helix_abi_version(void) {
    return HELIX_ABI_VERSION;
}

extern "C" const char* helix_version_string(void) {
    static const char* ver =
        HELIX_VERSION_STRING "+llama.cpp-" HELIX_LLAMACPP_SHA;
    return ver;
}

/* ------------------------------------------------------------------ */
/*  Error                                                              */
/* ------------------------------------------------------------------ */

extern "C" const char* helix_last_error_json(void) {
    return helix::last_error_json_ptr();
}

/* ------------------------------------------------------------------ */
/*  Runtime                                                            */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_runtime_create(const char* options_json,
                                                helix_runtime_t** out_runtime) {
    HELIX_TRY
    if (!out_runtime) helix::throw_invalid_arg("out_runtime must not be NULL");
    *out_runtime = nullptr;

    helix::RuntimeOptions opts = helix::parse_runtime_options(options_json);
    helix::Logger::instance().min_level.store(opts.log_level, std::memory_order_relaxed);

    *out_runtime = reinterpret_cast<helix_runtime_t*>(
        new helix::Runtime(opts));
    return HELIX_OK;
    HELIX_CATCH
}

extern "C" void helix_runtime_destroy(helix_runtime_t* runtime) {
    if (!runtime) return;
    delete reinterpret_cast<helix::Runtime*>(runtime);
}

/* WARNING: helix_runtime_describe() returns a borrowed pointer to an internal
 * std::string.  The pointer is valid only while the Runtime object is alive.
 * Callers that need the string to outlive the Runtime must copy it immediately.
 * For a safer alternative, use helix_runtime_describe_copy() which returns a
 * malloc'd copy that the caller must free with helix_free(). */
extern "C" const char* helix_runtime_describe(helix_runtime_t* runtime) {
    if (!runtime) return "{}";
    return reinterpret_cast<helix::Runtime*>(runtime)->describe().c_str();
}

static char* describe_copy_impl(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (p) std::memcpy(p, s.data(), s.size() + 1);
    return p;
}

extern "C" char* helix_runtime_describe_copy(helix_runtime_t* runtime) {
    if (!runtime) return describe_copy_impl("{}");
    return describe_copy_impl(
        reinterpret_cast<helix::Runtime*>(runtime)->describe());
}

/* ------------------------------------------------------------------ */
/*  Models                                                             */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_model_load(helix_runtime_t* runtime,
                                            const char* model_json,
                                            helix_model_t** out_model) {
    HELIX_TRY
    if (!runtime)    helix::throw_invalid_arg("runtime must not be NULL");
    if (!model_json) helix::throw_invalid_arg("model_json must not be NULL");
    if (!out_model)  helix::throw_invalid_arg("out_model must not be NULL");
    *out_model = nullptr;

    auto& rt  = *reinterpret_cast<helix::Runtime*>(runtime);
    auto  opts = helix::parse_model_options(model_json);

    *out_model = reinterpret_cast<helix_model_t*>(
        new helix::Model(rt, opts));
    return HELIX_OK;
    HELIX_CATCH
}

extern "C" helix_status_t helix_model_load_ex(helix_runtime_t* runtime,
                                              const char* model_json,
                                              helix_load_progress_cb cb,
                                              void* user_data,
                                              helix_model_t** out_model) {
    HELIX_TRY
    if (!runtime)    helix::throw_invalid_arg("runtime must not be NULL");
    if (!model_json) helix::throw_invalid_arg("model_json must not be NULL");
    if (!out_model)  helix::throw_invalid_arg("out_model must not be NULL");
    *out_model = nullptr;

    auto& rt  = *reinterpret_cast<helix::Runtime*>(runtime);
    auto  opts = helix::parse_model_options(model_json);

    /* cb may be NULL — then this is exactly helix_model_load. */
    *out_model = reinterpret_cast<helix_model_t*>(
        new helix::Model(rt, opts, cb, user_data));
    return HELIX_OK;
    HELIX_CATCH
}

extern "C" void helix_model_release(helix_model_t* model) {
    if (!model) return;
    delete reinterpret_cast<helix::Model*>(model);
}

/* WARNING: helix_model_describe() returns a borrowed pointer to an internal
 * std::string.  The pointer is valid only while the Model object is alive.
 * Callers that need the string to outlive the Model must copy it immediately.
 * For a safer alternative, use helix_model_describe_copy() which returns a
 * malloc'd copy that the caller must free with helix_free(). */
extern "C" const char* helix_model_describe(helix_model_t* model) {
    if (!model) return "{}";
    return reinterpret_cast<helix::Model*>(model)->describe().c_str();
}

extern "C" char* helix_model_describe_copy(helix_model_t* model) {
    if (!model) return describe_copy_impl("{}");
    return describe_copy_impl(
        reinterpret_cast<helix::Model*>(model)->describe());
}

/* ------------------------------------------------------------------ */
/*  Sessions                                                           */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_session_create(helix_model_t* model,
                                                const char* session_json,
                                                helix_session_t** out_session) {
    HELIX_TRY
    if (!model)      helix::throw_invalid_arg("model must not be NULL");
    if (!out_session) helix::throw_invalid_arg("out_session must not be NULL");
    *out_session = nullptr;

    auto& m   = *reinterpret_cast<helix::Model*>(model);
    auto  opts = helix::parse_session_options(session_json);

    *out_session = reinterpret_cast<helix_session_t*>(
        new helix::Session(m, opts));
    return HELIX_OK;
    HELIX_CATCH
}

extern "C" void helix_session_destroy(helix_session_t* session) {
    if (!session) return;
    delete reinterpret_cast<helix::Session*>(session);
}

extern "C" char* helix_session_describe(helix_session_t* session) {
    if (!session) return describe_copy_impl("{}");
    return describe_copy_impl(
        reinterpret_cast<helix::Session*>(session)->describe());
}

/* ------------------------------------------------------------------ */
/*  Chat completions                                                   */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_chat_completions(helix_session_t* session,
                                                   const char* request_json,
                                                   char** out_response_json) {
    HELIX_TRY
    if (!session)          helix::throw_invalid_arg("session must not be NULL");
    if (!request_json)     helix::throw_invalid_arg("request_json must not be NULL");
    if (!out_response_json) helix::throw_invalid_arg("out_response_json must not be NULL");
    *out_response_json = nullptr;

    return reinterpret_cast<helix::Session*>(session)
        ->chat_completions(request_json, out_response_json);
    HELIX_CATCH
}

extern "C" helix_status_t helix_chat_completions_stream(helix_session_t* session,
                                                          const char* request_json,
                                                          helix_stream_cb on_chunk,
                                                          void* user_data) {
    HELIX_TRY
    if (!session)      helix::throw_invalid_arg("session must not be NULL");
    if (!request_json) helix::throw_invalid_arg("request_json must not be NULL");
    if (!on_chunk)     helix::throw_invalid_arg("on_chunk must not be NULL");

    return reinterpret_cast<helix::Session*>(session)
        ->chat_completions_stream(request_json, on_chunk, user_data);
    HELIX_CATCH
}

extern "C" void helix_session_cancel(helix_session_t* session) {
    if (!session) return;
    reinterpret_cast<helix::Session*>(session)->cancel();
}

/* ------------------------------------------------------------------ */
/*  Hidden-state extraction (Phase 3) — stubs                         */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_session_encode(
        helix_session_t* session,
        const char* messages_json,
        float* out_hidden_state) {
    HELIX_TRY
    if (!session)       helix::throw_invalid_arg("session must not be NULL");
    if (!messages_json) helix::throw_invalid_arg("messages_json must not be NULL");
    if (!out_hidden_state) helix::throw_invalid_arg("out_hidden_state must not be NULL");

    helix::throw_unsupported("session_encode is not yet implemented");
    HELIX_CATCH
}

extern "C" uint32_t helix_model_hidden_dim(helix_model_t* model) {
    if (!model) return 0;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Embeddings (HELIX_1.1)                                             */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_embeddings(helix_session_t* session,
                                           const char* request_json,
                                           char** out_response_json) {
    HELIX_TRY
    if (!session)           helix::throw_invalid_arg("session must not be NULL");
    if (!request_json)      helix::throw_invalid_arg("request_json must not be NULL");
    if (!out_response_json) helix::throw_invalid_arg("out_response_json must not be NULL");
    *out_response_json = nullptr;

    return reinterpret_cast<helix::Session*>(session)
        ->embeddings(request_json, out_response_json);
    HELIX_CATCH
}

/* Pure query: no error state, no thread-local writes, safe to call
 * concurrently. NULL-safe like the destroy functions. */
extern "C" uint32_t helix_model_embedding_dim(helix_model_t* model) {
    if (!model) return 0;
    return reinterpret_cast<helix::Model*>(model)->embedding_dim();
}

/* ------------------------------------------------------------------ */
/*  Session state persistence + rerank (HELIX_1.5)                     */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_session_save(helix_session_t* session,
                                             const char* path) {
    HELIX_TRY
    if (!session) helix::throw_invalid_arg("session must not be NULL");
    if (!path)    helix::throw_invalid_arg("path must not be NULL");
    if (!path[0]) helix::throw_invalid_arg("path must not be empty");

    return reinterpret_cast<helix::Session*>(session)->save_state(path);
    HELIX_CATCH
}

extern "C" helix_status_t helix_session_restore(helix_session_t* session,
                                                const char* path) {
    HELIX_TRY
    if (!session) helix::throw_invalid_arg("session must not be NULL");
    if (!path)    helix::throw_invalid_arg("path must not be NULL");
    if (!path[0]) helix::throw_invalid_arg("path must not be empty");

    return reinterpret_cast<helix::Session*>(session)->restore_state(path);
    HELIX_CATCH
}

extern "C" helix_status_t helix_rerank(helix_session_t* session,
                                       const char* request_json,
                                       char** out_response_json) {
    HELIX_TRY
    if (!session)           helix::throw_invalid_arg("session must not be NULL");
    if (!request_json)      helix::throw_invalid_arg("request_json must not be NULL");
    if (!out_response_json) helix::throw_invalid_arg("out_response_json must not be NULL");
    *out_response_json = nullptr;

    return reinterpret_cast<helix::Session*>(session)
        ->rerank(request_json, out_response_json);
    HELIX_CATCH
}

/* ------------------------------------------------------------------ */
/*  Tokenizer utilities (HELIX_1.4)                                    */
/* ------------------------------------------------------------------ */

extern "C" helix_status_t helix_count_tokens(helix_session_t* session,
                                             const char* request_json,
                                             uint32_t* out_token_count) {
    HELIX_TRY
    if (!session)         helix::throw_invalid_arg("session must not be NULL");
    if (!request_json)    helix::throw_invalid_arg("request_json must not be NULL");
    if (!out_token_count) helix::throw_invalid_arg("out_token_count must not be NULL");
    *out_token_count = 0;

    return reinterpret_cast<helix::Session*>(session)
        ->count_tokens(request_json, out_token_count);
    HELIX_CATCH
}

extern "C" helix_status_t helix_tokenize(helix_model_t* model,
                                         const char* text,
                                         int add_special,
                                         int parse_special,
                                         char** out_tokens_json) {
    HELIX_TRY
    if (!model)           helix::throw_invalid_arg("model must not be NULL");
    if (!text)            helix::throw_invalid_arg("text must not be NULL");
    if (!out_tokens_json) helix::throw_invalid_arg("out_tokens_json must not be NULL");
    *out_tokens_json = nullptr;

    const std::string json = reinterpret_cast<helix::Model*>(model)
        ->tokenize_json(text, add_special != 0, parse_special != 0);

    /* Allocated with C malloc() so the caller frees it via helix_free()
     * (which calls free()). Keep these two paired — see helix_free(). */
    char* p = static_cast<char*>(std::malloc(json.size() + 1));
    if (!p) throw std::bad_alloc();
    std::memcpy(p, json.c_str(), json.size() + 1);
    *out_tokens_json = p;
    return HELIX_OK;
    HELIX_CATCH
}

/* Must stay paired with the allocation in run_chat_completions, which uses
 * the C library's malloc(). If that ever changes to new[]/new, this must change
 * to the matching delete[]/delete in lockstep — crossing the malloc/new heaps
 * (especially across a mismatched Windows CRT) corrupts the heap. */
extern "C" void helix_free(char* ptr) {
    std::free(ptr);
}

/* ------------------------------------------------------------------ */
/*  Logging                                                            */
/* ------------------------------------------------------------------ */

extern "C" void helix_set_log_callback(helix_log_cb cb,
                                        void* user_data,
                                        helix_log_level_t min_level) {
    auto& logger = helix::Logger::instance();
    std::lock_guard<std::mutex> lk(logger.mu);
    logger.cb        = cb;
    logger.userdata  = user_data;
    logger.min_level.store(min_level, std::memory_order_relaxed);
}
