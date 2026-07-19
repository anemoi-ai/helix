#include "../session/session.hpp"
#include "../model/model.hpp"
#include "../json/request.hpp"
#include "../json/response.hpp"
#include "../chat/template.hpp"
#include "../chat/reasoning.hpp"
#include "../multimodal/mmproj.hpp"
#include "../sampling/chain.hpp"
#include "../sampling/logit_bias.hpp"
#include "../grammar/response_format.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "../internal/version.hpp"
#include "decode_loop.hpp"
#include "event_sink.hpp"
#include "logprobs.hpp"
#include "utf8.hpp"
#include "stop_strings.hpp"
#include "lcp.hpp"

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "peg-parser.h"
#include "speculative.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace helix {

/* ------------------------------------------------------------------ */
/*  Completion ID / fingerprint helpers                                */
/* ------------------------------------------------------------------ */

// Single per-thread RNG.  std::random_device may throw on some platforms
// (e.g. MinGW without a working entropy source), so we fall back to a
// time + thread-id seed rather than letting the exception propagate.
static std::mt19937_64& thread_rng() {
    static thread_local std::mt19937_64 rng = []() {
        try {
            return std::mt19937_64(std::random_device{}());
        } catch (...) {
            log_warn("std::random_device unavailable; seeding completion-ID RNG "
                     "from clock + thread id (IDs are non-cryptographic regardless)");
            auto t = static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            auto id = static_cast<uint64_t>(
                std::hash<std::thread::id>{}(std::this_thread::get_id()));
            return std::mt19937_64(t ^ (id * 0x9e3779b97f4a7c15ULL));
        }
    }();
    return rng;
}

static std::string make_completion_id() {
    char buf[40];
    snprintf(buf, sizeof(buf), "chatcmpl-helix-%016llx",
             static_cast<unsigned long long>(thread_rng()()));
    return buf;
}

/* Generate "call_" + 24 random base62 characters, matching OpenAI's format. */
static std::string make_tool_call_id() {
    static constexpr char kBase62Chars[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    static_assert(sizeof(kBase62Chars) == 63, "base62 table must be 62 chars + NUL");
    std::uniform_int_distribution<int> dist(0, 61);
    char buf[30];
    for (int i = 0; i < 24; ++i)
        buf[i] = kBase62Chars[dist(thread_rng())];
    buf[24] = '\0';
    return std::string("call_") + buf;
}

static std::string make_fingerprint() {
    return "helix-" HELIX_VERSION_STRING "+llama.cpp-" HELIX_LLAMACPP_SHA;
}

static int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/* ------------------------------------------------------------------ */
/*  LCP (longest common prefix) of two token sequences                */
/*  (see lcp.hpp for the template implementation)                      */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Single-completion decode                                           */
/* ------------------------------------------------------------------ */

struct LogprobsConfig {
    bool  enabled     = false;
    int   top_n       = 0;
    float temperature = 1.0f;
};

struct DecodeResult {
    std::string                   content;
    std::string                   finish_reason;
    int                           n_predicted    = 0;
    int                           reasoning_tokens = 0; /* phase 7: tokens inside <think> */
    bool                          cancelled      = false;
    std::vector<llama_token>      tokens;       /* all generated tokens (for prefix cache) */
    std::vector<TokenLogprobEntry> logprobs;    /* phase 6: per-token log-prob info */

    /* 1.6 context_shift: tokens evicted from the KV during generation, and
     * the post-shift token sequence that exactly mirrors KV seq 0 (replaces
     * prompt+tokens for the prefix cache when evicted_tokens > 0). */
    int                      evicted_tokens = 0;
    std::vector<llama_token> cache_tokens;
};

/* Generation-time context shift (1.6). When enabled and the next KV cell
 * would exceed n_ctx, evict a block of the oldest cells after n_keep
 * (the system-prompt prefix) and rope-shift the rest down — the same
 * discard-half strategy llama-server uses. */
struct ShiftConfig {
    bool enabled = false;
    int  n_keep  = 0;
    int  n_ctx   = 0;
};

/* Route tool-call diffs from the PEG parser to the event sink.
 * Returns the number of newly-started tool calls. */
static void route_diffs(const std::vector<common_chat_msg_diff>& diffs,
                        EventSink& sink,
                        int choice_index,
                        int& tool_calls_started,
                        std::string* reasoning_text_out = nullptr) {
    for (const auto& diff : diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            sink.on_reasoning_delta(choice_index, diff.reasoning_content_delta);
            if (reasoning_text_out)
                *reasoning_text_out += diff.reasoning_content_delta;
        }
        if (!diff.content_delta.empty()) {
            sink.on_content(choice_index, diff.content_delta);
        }

        if (diff.tool_call_index == std::string::npos) continue;

        const int ti = static_cast<int>(diff.tool_call_index);
        if (ti >= tool_calls_started) {
            /* New tool call. Close the previous one if any. */
            if (tool_calls_started > 0) {
                sink.on_tool_call_end(choice_index, tool_calls_started - 1);
            }
            /* Prefer model-supplied id; generate one if absent or if it lacks "call_" prefix. */
            std::string tc_id = diff.tool_call_delta.id;
            if (tc_id.empty() || tc_id.rfind("call_", 0) != 0) {
                tc_id = make_tool_call_id();
            }
            sink.on_tool_call_start(choice_index, ti, tc_id, diff.tool_call_delta.name);
            tool_calls_started++;
            if (!diff.tool_call_delta.arguments.empty()) {
                sink.on_tool_call_arg_delta(choice_index, ti, diff.tool_call_delta.arguments);
            }
        } else {
            /* Existing tool call: argument delta (name/id deltas are internal). */
            if (!diff.tool_call_delta.arguments.empty()) {
                sink.on_tool_call_arg_delta(choice_index, ti, diff.tool_call_delta.arguments);
            }
        }
    }
}

static DecodeResult decode_one(Session& sess,
                                const std::vector<llama_token>& prompt_tokens,
                                int max_new_tokens,
                                const std::vector<std::string>& stops,
                                llama_sampler* sampler,
                                std::atomic<bool>& cancel_flag,
                                EventSink& sink,
                                int choice_index,
                                size_t lcp,
                                int stream_coalesce_ms,
                                const RenderResult& rendered,
                                const LogprobsConfig& lp_cfg,
                                bool skip_text_prefill = false,
                                llama_pos n_past_mm = -1,
                                const ShiftConfig& shift = {}) {
    llama_context* ctx = sess.ctx();
    const int n_prompt = static_cast<int>(prompt_tokens.size());
    const int n_batch  = llama_n_batch(ctx);

    if (!skip_text_prefill) {
        /* ---- KV management: keep [0, lcp), remove [lcp, ∞) ---- */
        {
            llama_memory_t mem = llama_get_memory(ctx);
            if (!llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(lcp), -1)) {
                llama_memory_seq_rm(mem, 0, 0, -1);
                lcp = 0;
            }
        }

        /* ---- Cancel check before heavy work ---- */
        if (cancel_flag.load(std::memory_order_relaxed)) {
            return {"", "stop", 0, 0, true, {}};
        }

        /* ---- Prefill ---- */
        for (int i = static_cast<int>(lcp); i < n_prompt; i += n_batch) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                return {"", "stop", 0, 0, true, {}};
            }
            int chunk_size = std::min(n_batch, n_prompt - i);
            llama_batch batch = llama_batch_get_one(
                const_cast<llama_token*>(prompt_tokens.data()) + i, chunk_size);
            if (llama_decode(ctx, batch) != 0) {
                throw Error(HELIX_E_BACKEND, "llama_decode failed during prefill",
                            "backend_error", "", "helix_e_backend");
            }
        }
    } else {
        /* Multimodal path: prefill was done externally by eval_media. */
        if (cancel_flag.load(std::memory_order_relaxed)) {
            return {"", "stop", 0, 0, true, {}};
        }
    }

    sink.on_role(choice_index);
    if (cancel_flag.load(std::memory_order_relaxed)) {
        return {"", "stop", 0, 0, true, {}};
    }

    /* ---- Set up PEG parser when tools are active ---- */
    bool has_parser = rendered.has_tools && !rendered.parser_data.empty();
    common_peg_arena peg_arena;
    common_chat_parser_params pp;
    common_chat_msg prev_parsed;
    if (has_parser) {
        try {
            peg_arena.load(rendered.parser_data);
            pp.format             = static_cast<common_chat_format>(rendered.chat_format);
            pp.generation_prompt  = rendered.generation_prompt;
            if (rendered.extract_reasoning)
                pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        } catch (const std::exception& e) {
            /* pp is only partially initialized here; clear the flag so no
             * code path can pair has_parser==true with an unusable parser. */
            has_parser = false;
            log_warn("PEG arena load failed, falling back to content-only: " +
                     std::string(e.what()));
        }
    }

    /* ---- Decode loop ---- */
    std::string output;
    std::string finish_reason = "stop";
    int n_predicted = 0;
    std::string reasoning_text; /* accumulated reasoning content (re-tokenized for an
                                 * accurate reasoning_tokens count at end of decode) */
    bool cancelled = false;
    std::vector<llama_token> gen_tokens;
    std::vector<TokenLogprobEntry> logprob_entries;
    int tool_calls_started = 0;

    /* Whether this call is operating on a multimodal-prefilled context.
     * n_past_mm == -1 is the sentinel for "text-only"; any non-negative value
     * is the starting KV position after multimodal prefill. */
    const bool has_mm = (n_past_mm >= 0);

    /* For multimodal: the ISWA KV cache's seq_pos_max() delegates to the SWA
     * sub-cache, which may mis-report max position after M-RoPE image embedding.
     * Track n_past explicitly (as mtmd-cli does) and pass it to each decode call
     * so llama.cpp never needs to infer the next position from the KV.
     *
     * BatchGuard ensures llama_batch_free() runs on all exits (return + throw).
     * llama_batch_free({}) is safe — it null-checks every pointer. */
    struct BatchGuard {
        llama_batch b;
        ~BatchGuard() { llama_batch_free(b); }
    } mm_guard{has_mm ? llama_batch_init(1, 0, 1) : llama_batch{}};
    llama_batch& mm_batch  = mm_guard.b;
    llama_pos    mm_n_past = n_past_mm;

    /* Reasoning extractor: splits <think>...</think> from regular content.
     * thinking_forced_open: the prompt already opened the think block, so
     * generation starts mid-thought (only the closing tag will appear). */
    ReasoningExtractor reasoning_ex(rendered.extract_reasoning,
                                    rendered.thinking_forced_open);

    // For logprob capture we need vocab + n_vocab.
    const llama_vocab* lp_vocab =
        lp_cfg.enabled ? llama_model_get_vocab(sess.model().llama_model_ptr()) : nullptr;
    const int lp_n_vocab = lp_vocab ? llama_vocab_n_tokens(lp_vocab) : 0;

    size_t emitted_up_to = 0; /* content-only path: bytes sent to sink */
    size_t stop_scan     = 0; /* incremental scan offset for check_stops */
    auto last_emit = std::chrono::steady_clock::now();
    const llama_vocab* vocab = llama_model_get_vocab(sess.model().llama_model_ptr());

    /* 1.6 context-shift state. n_past mirrors the KV cell count of seq 0
     * (prompt + fed-back tokens); cache_tokens mirrors the cell CONTENT so
     * the prefix-cache invariant (last_tokens_ == KV seq 0) survives
     * evictions. Multimodal requests never reach here with shift enabled
     * (rejected in run_chat_completions). */
    int n_past        = n_prompt;
    int evicted_total = 0;
    std::vector<llama_token> cache_tokens;
    if (shift.enabled) cache_tokens = prompt_tokens;

    for (;;) {
        if (cancel_flag.load(std::memory_order_relaxed)) {
            cancelled = true;
            finish_reason = "stop";
            break;
        }

        /* Capture the logits pointer for logprob computation. Sampling builds
         * its own candidate array from these values without modifying them,
         * and the buffer stays valid until the next llama_decode, so no copy
         * is needed. */
        const float* lp_raw = lp_cfg.enabled ? llama_get_logits_ith(ctx, -1)
                                             : nullptr;

        /* llama_sampler_sample applies all samplers AND calls accept internally.
         * Do NOT call llama_sampler_accept again — that would double-advance the
         * grammar state machine and cause "empty grammar stack" errors. */
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            finish_reason = "stop";
            break;
        }

        std::string piece = common_token_to_piece(ctx, new_token, /*special=*/false);
        output += piece;
        gen_tokens.push_back(new_token);
        ++n_predicted;

        // Compute logprob entry for this token (phase 6).
        TokenLogprobEntry lp_entry;
        if (lp_cfg.enabled && lp_raw) {
            lp_entry = compute_logprob_entry(lp_raw, lp_n_vocab, new_token, ctx,
                                              lp_cfg.temperature, lp_cfg.top_n);
            logprob_entries.push_back(lp_entry);
        }

        /* Stop-string check (content-only path: against raw output).
         * For tool-aware paths, stop strings are checked against raw output too —
         * the grammar constraint keeps well-formed output so template stops fire
         * at the right place (EOS / end-of-turn tokens). */
        size_t safe = valid_utf8_prefix_len(output.data(), output.size());
        size_t stop_pos = check_stops(output, safe, stops, &stop_scan);
        if (stop_pos != std::string::npos) {
            output.resize(stop_pos);
            finish_reason = "stop";
            break;
        }

        if (n_predicted >= max_new_tokens) {
            finish_reason = "length";
            break;
        }

        /* Context shift (1.6): the next decode writes KV cell n_past; when
         * that would exceed the context, evict the oldest block after the
         * system-prompt prefix and rope-shift the remaining cells down so
         * decoding continues instead of dying at the wall. */
        if (shift.enabled && n_past >= shift.n_ctx) {
            const int n_left    = n_past - shift.n_keep;
            const int n_discard = std::max(1, n_left / 2);
            llama_memory_t mem = llama_get_memory(ctx);
            llama_memory_seq_rm (mem, 0, shift.n_keep, shift.n_keep + n_discard);
            llama_memory_seq_add(mem, 0, shift.n_keep + n_discard, -1, -n_discard);
            cache_tokens.erase(cache_tokens.begin() + shift.n_keep,
                               cache_tokens.begin() + shift.n_keep + n_discard);
            n_past        -= n_discard;
            evicted_total += n_discard;
            log_debug("context shift: evicted " + std::to_string(n_discard) +
                      " tokens (n_keep=" + std::to_string(shift.n_keep) +
                      ", n_past now " + std::to_string(n_past) + ")");
        }

        /* Feed token back for next sampling step. */
        llama_batch next_batch;
        if (has_mm) {
            common_batch_clear(mm_batch);
            common_batch_add(mm_batch, new_token, mm_n_past++, {0}, true);
            next_batch = mm_batch;
        } else {
            next_batch = llama_batch_get_one(&new_token, 1);
        }
        if (llama_decode(ctx, next_batch) != 0) {
            throw Error(HELIX_E_BACKEND, "llama_decode failed during generation",
                        "backend_error", "", "helix_e_backend");
        }
        ++n_past;
        if (shift.enabled) cache_tokens.push_back(new_token);

        /* Content emission: per-token with logprobs, or coalesced without.
         * Both paths hold back any output suffix that is a prefix of a stop
         * string — a stop match can span token boundaries, and once bytes
         * reach the sink they cannot be retracted. Held-back bytes that never
         * complete a stop are released on a later token or the final flush. */
        if (lp_cfg.enabled && !has_parser && lp_raw) {
            /* Per-token emission path (logprobs active on content-only path).
             * Route the newly emittable bytes through the reasoning extractor
             * and attach this token's logprob entry to whatever content the
             * extractor releases this step. */
            size_t new_safe = valid_utf8_prefix_len(output.data(), output.size());
            size_t emit_len = new_safe - partial_stop_len(output, new_safe, stops);
            if (emit_len > emitted_up_to) {
                auto delta = std::string_view(output.data() + emitted_up_to,
                                              emit_len - emitted_up_to);
                auto rout = reasoning_ex.push(delta);
                if (!rout.reasoning.empty()) {
                    sink.on_reasoning_delta(choice_index, rout.reasoning);
                    reasoning_text += rout.reasoning;
                }
                if (!rout.content.empty()) {
                    sink.on_token_with_logprob(choice_index, rout.content, lp_entry);
                }
                emitted_up_to = emit_len;
            }
            last_emit = std::chrono::steady_clock::now();
            if (cancel_flag.load(std::memory_order_relaxed)) {
                cancelled = true;
                finish_reason = "stop";
                break;
            }
        } else {
            /* Coalesced content emission. */
            size_t new_safe = valid_utf8_prefix_len(output.data(), output.size());
            size_t emit_len = new_safe - partial_stop_len(output, new_safe, stops);
            if (emit_len > (has_parser ? 0 : emitted_up_to)) {
                bool should_emit = (stream_coalesce_ms <= 0);
                if (!should_emit) {
                    auto now_tp = std::chrono::steady_clock::now();
                    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now_tp - last_emit).count();
                    should_emit = (ms >= stream_coalesce_ms);
                }
                if (should_emit) {
                    if (has_parser && !peg_arena.empty()) {
                        try {
                            common_chat_msg current = common_chat_peg_parse(
                                peg_arena, output.substr(0, emit_len),
                                /*is_partial=*/true, pp);
                            auto diffs = common_chat_msg_diff::compute_diffs(prev_parsed, current);
                            route_diffs(diffs, sink, choice_index, tool_calls_started,
                                        &reasoning_text);
                            prev_parsed = std::move(current);
                        } catch (const std::exception& e) {
                            log_warn("PEG parse error during streaming: " + std::string(e.what()));
                        }
                    } else {
                        auto delta = std::string_view(output.data() + emitted_up_to,
                                                      emit_len - emitted_up_to);
                        auto rout = reasoning_ex.push(delta);
                        if (!rout.reasoning.empty()) {
                            sink.on_reasoning_delta(choice_index, rout.reasoning);
                            reasoning_text += rout.reasoning;
                        }
                        if (!rout.content.empty()) {
                            sink.on_content(choice_index, rout.content);
                        }
                        emitted_up_to = emit_len;
                    }
                    last_emit = std::chrono::steady_clock::now();
                    if (cancel_flag.load(std::memory_order_relaxed)) {
                        cancelled = true;
                        finish_reason = "stop";
                        break;
                    }
                }
            }
        }
    }

    /* ---- Final flush ---- */
    if (has_parser && !peg_arena.empty() && !cancelled) {
        /* Final parse (non-partial): extracts complete tool calls and remaining content. */
        try {
            common_chat_msg final_msg = common_chat_peg_parse(
                peg_arena, output, /*is_partial=*/false, pp);
            auto diffs = common_chat_msg_diff::compute_diffs(prev_parsed, final_msg);
            route_diffs(diffs, sink, choice_index, tool_calls_started, &reasoning_text);

            if (tool_calls_started > 0) {
                sink.on_tool_call_end(choice_index, tool_calls_started - 1);
                finish_reason = "tool_calls";
            }
        } catch (const std::exception& e) {
            log_warn("PEG final parse error: " + std::string(e.what()));
            /* Fall through with whatever content the sink already received. */
        }
    } else if (!has_parser) {
        /* Content-only: push remaining safe bytes through reasoning extractor, then flush. */
        size_t final_safe = valid_utf8_prefix_len(output.data(), output.size());
        if (final_safe > emitted_up_to) {
            auto delta = std::string_view(output.data() + emitted_up_to,
                                           final_safe - emitted_up_to);
            auto rout = reasoning_ex.push(delta);
            if (!rout.reasoning.empty()) {
                sink.on_reasoning_delta(choice_index, rout.reasoning);
                reasoning_text += rout.reasoning;
            }
            if (!rout.content.empty()) sink.on_content(choice_index, rout.content);
        }
        /* Flush the lookahead buffer (partial tag text). */
        auto flush_rout = reasoning_ex.flush();
        if (!flush_rout.reasoning.empty()) {
            sink.on_reasoning_delta(choice_index, flush_rout.reasoning);
            reasoning_text += flush_rout.reasoning;
        }
        if (!flush_rout.content.empty()) sink.on_content(choice_index, flush_rout.content);

        if (!reasoning_text.empty()) sink.on_reasoning_end(choice_index);
    }
    /* Note: if cancelled and has_parser, we skip the final parse — any pending
     * tool call (started but not ended) is dropped, and finish_reason stays "stop". */

    /* Reasoning token count: re-tokenize the extracted reasoning text with the
     * model's own tokenizer so the count is BPE-accurate (matching how usage
     * is reported by OpenAI), rather than a chars/4 heuristic. add_special and
     * parse_special are false: the reasoning text carries no BOS and the
     * <think>/</think> markers have already been stripped by the extractor. */
    int reasoning_tokens = 0;
    if (!reasoning_text.empty()) {
        reasoning_tokens = static_cast<int>(
            common_tokenize(ctx, reasoning_text, /*add_special=*/false,
                            /*parse_special=*/false).size());
    }

    return {std::move(output), finish_reason, n_predicted, reasoning_tokens, cancelled,
            std::move(gen_tokens), std::move(logprob_entries),
            evicted_total, std::move(cache_tokens)};
}

/* ------------------------------------------------------------------ */
/*  Speculative (MTP) single-completion decode                         */
/* ------------------------------------------------------------------ */

static DecodeResult decode_one_speculative(Session& sess,
                                            const std::vector<llama_token>& prompt_tokens,
                                            int max_new_tokens,
                                            const std::vector<std::string>& stops,
                                            llama_sampler* sampler,
                                            std::atomic<bool>& cancel_flag,
                                            EventSink& sink,
                                            int choice_index,
                                            int stream_coalesce_ms,
                                            const RenderResult& rendered,
                                            const LogprobsConfig& lp_cfg) {
    llama_context* ctx = sess.ctx();
    common_speculative* spec = sess.spec_mtp();
    const int n_prompt = static_cast<int>(prompt_tokens.size());
    const int n_batch  = llama_n_batch(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(sess.model().llama_model_ptr());

    /* ---- Prefill (text-only; the speculative path is never used for
     * multimodal — run_chat_completions falls back to decode_one). ---- */
    {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, 0, 0, -1);
    }
    if (cancel_flag.load(std::memory_order_relaxed))
        return {"", "stop", 0, 0, true, {}};

    for (int i = 0; i < n_prompt; i += n_batch) {
        if (cancel_flag.load(std::memory_order_relaxed))
            return {"", "stop", 0, 0, true, {}};
        int chunk_size = std::min(n_batch, n_prompt - i);
        /* Use common_batch_add (sets seq_id) rather than llama_batch_get_one,
         * because common_speculative_process reads seq_id arrays per token. */
        llama_batch pbatch = llama_batch_init(chunk_size, 0, 1);
        for (int j = 0; j < chunk_size; ++j) {
            /* Only the final prompt token needs logits (for sampling the
             * first output token). common_batch_add requires explicit logits
             * flags, unlike llama_batch_get_one which auto-enables the last. */
            bool is_last = (i + j == n_prompt - 1);
            common_batch_add(pbatch, prompt_tokens[i + j],
                             static_cast<llama_pos>(i + j), {0}, /*logits=*/is_last);
        }
        if (llama_decode(ctx, pbatch) != 0) {
            llama_batch_free(pbatch);
            throw Error(HELIX_E_BACKEND, "llama_decode failed during prefill",
                        "backend_error", "", "helix_e_backend");
        }
        /* Mirror the prefill chunk into the MTP draft context so its KV
         * state matches the target before drafting begins. */
        if (!common_speculative_process(spec, pbatch)) {
            llama_batch_free(pbatch);
            throw Error(HELIX_E_BACKEND,
                        "common_speculative_process failed during prefill",
                        "backend_error", "", "helix_e_backend");
        }
        llama_batch_free(pbatch);
    }

    sink.on_role(choice_index);
    if (cancel_flag.load(std::memory_order_relaxed))
        return {"", "stop", 0, 0, true, {}};

    /* ---- PEG parser setup (same as decode_one) ---- */
    bool has_parser = rendered.has_tools && !rendered.parser_data.empty();
    common_peg_arena peg_arena;
    common_chat_parser_params pp;
    common_chat_msg prev_parsed;
    if (has_parser) {
        try {
            peg_arena.load(rendered.parser_data);
            pp.format            = static_cast<common_chat_format>(rendered.chat_format);
            pp.generation_prompt = rendered.generation_prompt;
            if (rendered.extract_reasoning)
                pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        } catch (const std::exception& e) {
            has_parser = false;
            log_warn("PEG arena load failed, falling back to content-only: " +
                     std::string(e.what()));
        }
    }

    /* ---- Decode state (mirrors decode_one) ---- */
    std::string output;
    std::string finish_reason = "stop";
    int n_predicted = 0;
    std::string reasoning_text;
    bool cancelled = false;
    std::vector<llama_token> gen_tokens;
    std::vector<TokenLogprobEntry> logprob_entries;
    int tool_calls_started = 0;

    ReasoningExtractor reasoning_ex(rendered.extract_reasoning,
                                    rendered.thinking_forced_open);

    const llama_vocab* lp_vocab =
        lp_cfg.enabled ? llama_model_get_vocab(sess.model().llama_model_ptr()) : nullptr;
    const int lp_n_vocab = lp_vocab ? llama_vocab_n_tokens(lp_vocab) : 0;

    size_t emitted_up_to = 0;
    size_t stop_scan     = 0; /* incremental scan offset for check_stops */
    auto last_emit = std::chrono::steady_clock::now();

    /* ---- Shared per-token emission helper ----
     * Processes one token through EOG/stop/length/emission, appending to
     * output and driving the sink. Returns true if generation should stop. */
    auto emit_one = [&](llama_token tk, const float* tk_logits) -> bool {
        if (llama_vocab_is_eog(vocab, tk)) {
            finish_reason = "stop";
            return true;
        }
        std::string piece = common_token_to_piece(ctx, tk, /*special=*/false);
        output += piece;
        gen_tokens.push_back(tk);
        ++n_predicted;

        TokenLogprobEntry lp_entry;
        if (lp_cfg.enabled && tk_logits) {
            lp_entry = compute_logprob_entry(tk_logits, lp_n_vocab, tk, ctx,
                                              lp_cfg.temperature, lp_cfg.top_n);
            logprob_entries.push_back(lp_entry);
        }

        size_t safe = valid_utf8_prefix_len(output.data(), output.size());
        size_t stop_pos = check_stops(output, safe, stops, &stop_scan);
        if (stop_pos != std::string::npos) {
            output.resize(stop_pos);
            finish_reason = "stop";
            return true;
        }
        if (n_predicted >= max_new_tokens) {
            finish_reason = "length";
            return true;
        }

        /* Content emission: per-token with logprobs, or coalesced. Both paths
         * hold back any output suffix that is a prefix of a stop string —
         * see decode_one for the rationale. */
        if (lp_cfg.enabled && !has_parser && tk_logits) {
            size_t new_safe = valid_utf8_prefix_len(output.data(), output.size());
            size_t emit_len = new_safe - partial_stop_len(output, new_safe, stops);
            if (emit_len > emitted_up_to) {
                auto delta = std::string_view(output.data() + emitted_up_to,
                                              emit_len - emitted_up_to);
                auto rout = reasoning_ex.push(delta);
                if (!rout.reasoning.empty()) {
                    sink.on_reasoning_delta(choice_index, rout.reasoning);
                    reasoning_text += rout.reasoning;
                }
                if (!rout.content.empty())
                    sink.on_token_with_logprob(choice_index, rout.content, lp_entry);
                emitted_up_to = emit_len;
            }
            last_emit = std::chrono::steady_clock::now();
            if (cancel_flag.load(std::memory_order_relaxed)) {
                cancelled = true;
                finish_reason = "stop";
                return true;
            }
        } else {
            size_t new_safe = valid_utf8_prefix_len(output.data(), output.size());
            size_t emit_len = new_safe - partial_stop_len(output, new_safe, stops);
            if (emit_len > (has_parser ? 0 : emitted_up_to)) {
                bool should_emit = (stream_coalesce_ms <= 0);
                if (!should_emit) {
                    auto now_tp = std::chrono::steady_clock::now();
                    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now_tp - last_emit).count();
                    should_emit = (ms >= stream_coalesce_ms);
                }
                if (should_emit) {
                    if (has_parser && !peg_arena.empty()) {
                        try {
                            common_chat_msg current = common_chat_peg_parse(
                                peg_arena, output.substr(0, emit_len),
                                /*is_partial=*/true, pp);
                            auto diffs = common_chat_msg_diff::compute_diffs(prev_parsed, current);
                            route_diffs(diffs, sink, choice_index, tool_calls_started,
                                        &reasoning_text);
                            prev_parsed = std::move(current);
                        } catch (const std::exception& e) {
                            log_warn("PEG parse error during streaming: " + std::string(e.what()));
                        }
                    } else {
                        auto delta = std::string_view(output.data() + emitted_up_to,
                                                       emit_len - emitted_up_to);
                        auto rout = reasoning_ex.push(delta);
                        if (!rout.reasoning.empty()) {
                            sink.on_reasoning_delta(choice_index, rout.reasoning);
                            reasoning_text += rout.reasoning;
                        }
                        if (!rout.content.empty())
                            sink.on_content(choice_index, rout.content);
                        emitted_up_to = emit_len;
                    }
                    last_emit = std::chrono::steady_clock::now();
                    if (cancel_flag.load(std::memory_order_relaxed)) {
                        cancelled = true;
                        finish_reason = "stop";
                        return true;
                    }
                }
            }
        }
        return false;
    };

    /* ---- common_speculative_begin: notify the manager that prefill is done */
    common_speculative_begin(spec, /*seq_id=*/0, prompt_tokens);

    /* ---- Sample the first token from the prefill logits (no draft for it) */
    const float* lp_raw_first = lp_cfg.enabled ? llama_get_logits_ith(ctx, -1) : nullptr;
    llama_token id_last = llama_sampler_sample(sampler, ctx, -1);
    llama_pos   n_past  = static_cast<llama_pos>(n_prompt);

    if (emit_one(id_last, lp_raw_first)) goto spec_done;

    /* ---- Speculative decode loop ---- */
    {
        llama_batch sbatch = llama_batch_init(n_batch, /*embd=*/ 0, /*n_seq_max=*/ 1);
        struct SBatchGuard { llama_batch b; ~SBatchGuard(){ llama_batch_free(b); } } sguard{sbatch};

        for (;;) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                cancelled = true;
                finish_reason = "stop";
                break;
            }

            /* 1. Request a draft. */
            std::vector<llama_token> draft;
            common_speculative_get_draft_params(spec, /*seq_id=*/0) = {
                /*drafting=*/true,
                /*n_max=*/   -1,                 /* impl clamps to remaining ctx */
                /*n_past=*/  n_past,
                /*id_last=*/ id_last,
                /*prompt=*/  &prompt_tokens,
                /*result=*/  &draft,
            };
            common_speculative_draft(spec);

            /* If no draft (e.g. context exhausted), fall back to a plain
             * single-token decode to make progress. Build the batch with
             * common_batch_add — common_speculative_process reads per-token
             * seq_id arrays, which llama_batch_get_one leaves null. */
            if (draft.empty()) {
                common_batch_clear(sbatch);
                common_batch_add(sbatch, id_last, n_past, {0}, /*logits=*/true);
                if (llama_decode(ctx, sbatch) != 0)
                    throw Error(HELIX_E_BACKEND, "llama_decode failed during MTP fallback",
                                "backend_error", "", "helix_e_backend");
                if (!common_speculative_process(spec, sbatch))
                    throw Error(HELIX_E_BACKEND,
                                "common_speculative_process failed during MTP fallback",
                                "backend_error", "", "helix_e_backend");
                llama_token next = llama_sampler_sample(sampler, ctx, -1);
                llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + 1, -1);
                common_speculative_accept(spec, 0, 0);
                if (emit_one(next, lp_cfg.enabled ? llama_get_logits_ith(ctx, 0) : nullptr))
                    break;
                n_past += 1;
                id_last = next;
                continue;
            }

            /* 2. Build target batch: [id_last, draft0, ..., draftN-1]. */
            std::vector<int> spec_i_batch;
            spec_i_batch.reserve(draft.size() + 1);
            common_batch_clear(sbatch);
            spec_i_batch.push_back(sbatch.n_tokens);
            common_batch_add(sbatch, id_last, n_past, {0}, /*logits=*/true);
            for (size_t i = 0; i < draft.size(); ++i) {
                spec_i_batch.push_back(sbatch.n_tokens);
                common_batch_add(sbatch, draft[i],
                                 n_past + static_cast<llama_pos>(i) + 1, {0}, /*logits=*/true);
            }

            /* 3. Target decode. */
            if (llama_decode(ctx, sbatch) != 0)
                throw Error(HELIX_E_BACKEND, "llama_decode failed during MTP generation",
                            "backend_error", "", "helix_e_backend");

            /* 4. Mirror target hidden states into the draft context. */
            if (!common_speculative_process(spec, sbatch))
                throw Error(HELIX_E_BACKEND,
                            "common_speculative_process failed",
                            "backend_error", "", "helix_e_backend");

            /* 5. Verify draft against the target via the raw sampler chain. */
            auto accepted = helix_sample_and_accept_n(sampler, ctx, spec_i_batch, draft);

            /* 6. KV cleanup: remove rejected draft positions. The last accepted
             *    token is only sampled (not decoded) so it stays out of the KV
             *    and becomes id_last for the next iteration. */
            llama_memory_seq_rm(llama_get_memory(ctx), 0,
                                n_past + static_cast<llama_pos>(accepted.size()), -1);

            /* 7. Inform the manager how many drafts were accepted. */
            common_speculative_accept(spec, /*seq_id=*/0,
                                      static_cast<uint16_t>(accepted.size() - 1));

            /* 8. Emit each accepted token through the shared emission helper. */
            bool loop_broke = false;
            for (size_t i = 0; i < accepted.size(); ++i) {
                const float* tk_logits = lp_cfg.enabled
                    ? llama_get_logits_ith(ctx, spec_i_batch[i]) : nullptr;
                if (emit_one(accepted[i], tk_logits)) {
                    loop_broke = true;
                    break;
                }
            }
            if (loop_broke) break;

            /* 9. Advance. */
            n_past  += static_cast<llama_pos>(accepted.size());
            id_last  = accepted.back();
        }
    }

spec_done:
    /* ---- Final flush (same as decode_one) ---- */
    if (has_parser && !peg_arena.empty() && !cancelled) {
        try {
            common_chat_msg final_msg = common_chat_peg_parse(
                peg_arena, output, /*is_partial=*/false, pp);
            auto diffs = common_chat_msg_diff::compute_diffs(prev_parsed, final_msg);
            route_diffs(diffs, sink, choice_index, tool_calls_started, &reasoning_text);
            if (tool_calls_started > 0) {
                sink.on_tool_call_end(choice_index, tool_calls_started - 1);
                finish_reason = "tool_calls";
            }
        } catch (const std::exception& e) {
            log_warn("PEG final parse error: " + std::string(e.what()));
        }
    } else if (!has_parser) {
        size_t final_safe = valid_utf8_prefix_len(output.data(), output.size());
        if (final_safe > emitted_up_to) {
            auto delta = std::string_view(output.data() + emitted_up_to,
                                           final_safe - emitted_up_to);
            auto rout = reasoning_ex.push(delta);
            if (!rout.reasoning.empty()) {
                sink.on_reasoning_delta(choice_index, rout.reasoning);
                reasoning_text += rout.reasoning;
            }
            if (!rout.content.empty()) sink.on_content(choice_index, rout.content);
        }
        auto flush_rout = reasoning_ex.flush();
        if (!flush_rout.reasoning.empty()) {
            sink.on_reasoning_delta(choice_index, flush_rout.reasoning);
            reasoning_text += flush_rout.reasoning;
        }
        if (!flush_rout.content.empty()) sink.on_content(choice_index, flush_rout.content);
        if (!reasoning_text.empty()) sink.on_reasoning_end(choice_index);
    }

    int reasoning_tokens = 0;
    if (!reasoning_text.empty()) {
        reasoning_tokens = static_cast<int>(
            common_tokenize(ctx, reasoning_text, /*add_special=*/false,
                            /*parse_special=*/false).size());
    }

    return {std::move(output), finish_reason, n_predicted, reasoning_tokens, cancelled,
            std::move(gen_tokens), std::move(logprob_entries)};
}
/* ------------------------------------------------------------------ */

helix_status_t run_chat_completions(Session& sess,
                                     const char* request_json,
                                     char** out_response_json,
                                     helix_stream_cb on_chunk,
                                     void* user_data,
                                     bool streaming) {
    /* 1. Parse and validate. */
    ChatRequest req = ChatRequest::from_json(request_json ? request_json : "");
    req.validate();

    /* 2. Route streaming vs non-streaming. */
    if (streaming && !req.stream) {
        throw_unsupported("streaming requires stream:true in the request body", "stream");
    }
    if (!streaming && req.stream) {
        throw_unsupported("use helix_chat_completions_stream for streaming requests", "stream");
    }

    /* 3. Model alias check. */
    if (req.model != sess.model().alias()) {
        throw Error(HELIX_E_MODEL_NOT_FOUND,
                    "model '" + req.model + "' not loaded in this session",
                    "model_not_found", "model", "helix_e_model_not_found");
    }

    /* 4. Multimodal capability check: reject vision/audio if model has no mmproj. */
    bool has_multimodal_content = false;
    for (const auto& msg : req.messages) {
        if (msg.content_parts.empty()) continue;
        for (const auto& part : msg.content_parts) {
            if (part.type == "image_url" || part.type == "input_audio") {
                has_multimodal_content = true;
                if (!sess.model().mmproj()) {
                    throw_unsupported(
                        "image/audio content parts require a model loaded with mmproj_path",
                        "messages");
                }
            }
        }
    }

    /* Multimodal prefill embeds the image positions once into the KV cache and
     * cannot be cheaply replayed per choice, so only n=1 is meaningful. Reject
     * n>1 explicitly rather than silently returning degraded completions that
     * reuse the first choice's image-embedding positions. */
    if (has_multimodal_content && req.n.value_or(1) > 1) {
        throw_unsupported("n > 1 is not supported with image/audio content", "n");
    }

    /* context_shift limits (1.6). Multimodal: image-embedding KV cells have
     * no token representation, so they can be neither re-rendered (prefill
     * truncation) nor position-shifted soundly. n > 1: choices re-use the
     * prompt KV, which eviction would have destroyed. Honest rejection
     * instead of a silent no-op, per house style. */
    const bool shift_on = sess.context_shift_enabled();
    if (shift_on && has_multimodal_content) {
        throw_unsupported(
            "context_shift sessions do not support image/audio content; "
            "create a separate session without context_shift for multimodal "
            "requests", "messages");
    }
    if (shift_on && req.n.value_or(1) > 1) {
        throw_unsupported("n > 1 is not supported with context_shift", "n");
    }

    /* 5. Render prompt (with tools if present).
     * Disable thinking for response_format requests — the grammar constrains
     * output directly, and thinking preamble would exhaust the token budget. */
    const bool has_rf = req.response_format &&
                        req.response_format->type != "text";
    const std::string& reasoning_fmt = sess.model().options().reasoning_format;
    RenderResult rendered = render_prompt(
        req.messages, req.stop, sess.model().chat_templates(),
        req.tools, req.tool_choice, req.parallel_tool_calls,
        /*enable_thinking=*/!has_rf,
        reasoning_fmt);

    /* 6. Tokenise (text-only path) or multimodal prefill. */
    llama_context* ctx = sess.ctx();
    std::vector<llama_token> prompt_tokens;
    bool multimodal_prefill_done = false;
    int n_prompt = 0;
    llama_pos mm_n_past_start = -1; /* set to n_past_after when multimodal */

    if (has_multimodal_content && sess.model().mmproj()) {
        /* Multimodal prefill: collect all media bytes from content parts in order,
         * call eval_media which tokenises the prompt + embeds images, then decode. */
        std::vector<std::vector<uint8_t>> all_media;
        for (const auto& msg : req.messages) {
            if (msg.content_parts.empty()) continue;
            for (const auto& part : msg.content_parts) {
                if (part.type == "image_url" || part.type == "input_audio") {
                    all_media.push_back(part.media_raw);
                }
            }
        }

        /* Clear KV for seq 0 before multimodal prefill — eval_media starts at
         * n_past=0 and the KV may still hold state from a prior request. */
        {
            llama_memory_t mem = llama_get_memory(ctx);
            llama_memory_seq_rm(mem, 0, 0, -1);
        }

        MmProj* mmproj = sess.model().mmproj();
        const int n_batch = llama_n_batch(ctx);

        /* Acquire mmproj mutex — eval_media is not thread-safe. */
        std::lock_guard<std::mutex> mm_lock(mmproj->mu());
        mm_n_past_start = mmproj->eval_media(
            ctx,
            rendered.prompt.c_str(),
            /*add_special=*/true,
            all_media,
            /*n_past=*/0,
            n_batch);

        n_prompt = static_cast<int>(mm_n_past_start);
        multimodal_prefill_done = true;

        /* If the caller cancelled while eval_media was running, clean up the
         * KV (which now holds image-embedding tokens) and bail out early
         * rather than wasting a generation step. */
        if (sess.cancel_flag().load(std::memory_order_relaxed)) {
            llama_memory_t mem = llama_get_memory(ctx);
            llama_memory_seq_rm(mem, 0, 0, -1);
            sess.set_last_tokens({});
            throw Error(HELIX_E_CANCELLED, "cancelled during multimodal prefill",
                        "cancel", "", "helix_e_cancelled");
        }
    } else {
        prompt_tokens = common_tokenize(
            ctx, rendered.prompt, /*add_special=*/true, /*parse_special=*/true);
        n_prompt = static_cast<int>(prompt_tokens.size());
    }

    const int n_ctx_size = static_cast<int>(llama_n_ctx(ctx));

    /* 6b. Prefill-overflow handling for context_shift sessions (1.6):
     * template-aware truncation. Drop the oldest non-system message (plus
     * any tool results paired with a dropped assistant tool call), then
     * RE-RENDER — messages are never split, and the template sees a valid
     * conversation every time. The leading system/developer block and the
     * last user message onward are never dropped. */
    int  prefill_evicted   = 0;
    bool prefill_truncated = false;
    if (shift_on) {
        /* Reserve generation headroom so the request doesn't immediately
         * fall into generation-time eviction. */
        const int reserve = std::min(256, std::max(16, n_ctx_size / 16));
        const int budget  = std::max(1, n_ctx_size - reserve);
        if (n_prompt > budget) {
            const int n_prompt_orig = n_prompt;
            std::vector<Message> msgs = req.messages;

            size_t sys_end = 0;
            while (sys_end < msgs.size() &&
                   (msgs[sys_end].role == "system" ||
                    msgs[sys_end].role == "developer")) {
                ++sys_end;
            }
            auto last_user_idx = [&msgs, sys_end]() -> size_t {
                for (size_t i = msgs.size(); i-- > sys_end; ) {
                    if (msgs[i].role == "user") return i;
                }
                return msgs.empty() ? 0 : msgs.size() - 1;
            };

            while (n_prompt > budget) {
                const size_t keep_from = last_user_idx();
                if (sys_end >= keep_from) break; /* nothing droppable left */

                size_t erase_end = sys_end + 1;
                if (msgs[sys_end].role == "assistant" &&
                    !msgs[sys_end].tool_calls.empty()) {
                    while (erase_end < keep_from &&
                           msgs[erase_end].role == "tool") {
                        ++erase_end;
                    }
                }
                msgs.erase(msgs.begin() + sys_end, msgs.begin() + erase_end);
                prefill_truncated = true;

                rendered = render_prompt(
                    msgs, req.stop, sess.model().chat_templates(),
                    req.tools, req.tool_choice, req.parallel_tool_calls,
                    /*enable_thinking=*/!has_rf, reasoning_fmt);
                prompt_tokens = common_tokenize(
                    ctx, rendered.prompt, /*add_special=*/true,
                    /*parse_special=*/true);
                n_prompt = static_cast<int>(prompt_tokens.size());
            }

            if (n_prompt >= n_ctx_size) {
                throw Error(HELIX_E_CONTEXT_FULL,
                            "context_shift could not make the request fit: "
                            "the system prompt + latest user message occupy " +
                            std::to_string(n_prompt) + " of " +
                            std::to_string(n_ctx_size) + " context tokens",
                            "context_length_exceeded", "messages",
                            "helix_e_context_full");
            }
            if (prefill_truncated) {
                prefill_evicted = n_prompt_orig - n_prompt;
                log_info("context shift: dropped oldest history before "
                         "prefill (" + std::to_string(prefill_evicted) +
                         " tokens; prompt now " + std::to_string(n_prompt) +
                         "/" + std::to_string(n_ctx_size) + ")");
            }
        }
    }

    /* With context_shift, generation can outlive the window via eviction, so
     * the request's own budget applies un-capped by the wall; an unset budget
     * is bounded at one full context of new tokens as a sanity limit. */
    int max_new;
    if (shift_on) {
        if (req.max_completion_tokens)  max_new = *req.max_completion_tokens;
        else if (req.max_tokens)        max_new = *req.max_tokens;
        else                            max_new = n_ctx_size;
    } else {
        max_new = req.effective_max_tokens(n_ctx_size, n_prompt);
    }

    const int n_choices      = req.n.value_or(1);
    const std::string comp_id = make_completion_id();
    const std::string fp      = make_fingerprint();
    const int64_t created_at  = unix_now();
    const bool include_usage  = req.stream_options &&
                                 req.stream_options->include_usage;

    /* 7. Prefix-cache: disabled for multimodal (image embeddings can't be cached). */
    size_t lcp = 0;
    if (!multimodal_prefill_done &&
        sess.prefix_cache_enabled() && !sess.last_tokens().empty()) {
        lcp = compute_lcp(prompt_tokens, sess.last_tokens());
        size_t last_cap = sess.last_tokens().size() - 1;
        if (lcp > last_cap) lcp = last_cap;
        if (n_prompt > 0 && lcp >= static_cast<size_t>(n_prompt)) {
            lcp = static_cast<size_t>(n_prompt - 1);
        }
    }

    /* 8. Build sink. */
    ChatResponse response;
    response.id                = comp_id;
    response.created           = created_at;
    response.model             = req.model;
    response.system_fingerprint = fp;

    CollectingSink collect_sink(response);
    StreamingSink  stream_sink(on_chunk, user_data,
                               comp_id, created_at, req.model, fp,
                               include_usage, sess.cancel_flag());

    EventSink& sink = streaming
        ? static_cast<EventSink&>(stream_sink)
        : static_cast<EventSink&>(collect_sink);

    /* 9. Resolve response_format grammar (phase 6).
     * When a PEG parser is present, skip response_format grammar too — the PEG parser
     * handles output without grammar constraints to avoid conflicts with tool call output
     * and <think> tokens on reasoning models (Qwen3.5, etc.). */
    const bool skip_grammar = rendered.has_tools && !rendered.parser_data.empty();
    const llama_vocab* vocab = llama_model_get_vocab(sess.model().llama_model_ptr());

    std::string rf_grammar;
    if (req.response_format && req.response_format->type != "text" && !skip_grammar) {
        ResponseFormatResolved rfs = resolve_response_format(
            req.response_format->type,
            req.response_format->json_schema,
            req.response_format->strict);
        rf_grammar = rfs.grammar_gbnf;
        for (const auto& w : rfs.warnings) log_warn(w);
    }

    // Determine which grammar to pass to the sampler chain.
    std::string sampler_grammar;
    bool sampler_grammar_lazy = false;
    std::vector<GrammarTrigger> sampler_triggers;
    if (!rf_grammar.empty()) {
        sampler_grammar = rf_grammar; /* eager — thinking disabled for rf requests */
    } else if (!skip_grammar) {
        sampler_grammar = rendered.grammar;
        sampler_grammar_lazy = rendered.grammar_lazy;
        sampler_triggers = rendered.grammar_triggers;
    }

    // Logit bias resolution (phase 6).
    std::vector<ResolvedBias> resolved_biases;
    if (!req.logit_bias.empty()) {
        resolved_biases = resolve_logit_bias(req.logit_bias, vocab);
    }

    // Logprobs config (phase 6).
    LogprobsConfig lp_cfg;
    lp_cfg.enabled     = req.logprobs;
    lp_cfg.top_n       = req.top_logprobs.value_or(0);
    lp_cfg.temperature = req.temperature.value_or(1.0f);

    /* Generation-time shift config (1.6). n_keep pins the system-prompt
     * prefix: the token-LCP between the full prompt and a system-only render
     * is exactly the shared system block, so eviction never touches it.
     * Without a system message, only a BOS (if any) is pinned. Capped at
     * n_ctx/2 so eviction always makes progress. */
    ShiftConfig shift_cfg;
    if (shift_on) {
        shift_cfg.enabled = true;
        shift_cfg.n_ctx   = n_ctx_size;
        int n_keep = llama_vocab_get_add_bos(vocab) ? 1 : 0;
        size_t sys_end = 0;
        while (sys_end < req.messages.size() &&
               (req.messages[sys_end].role == "system" ||
                req.messages[sys_end].role == "developer")) {
            ++sys_end;
        }
        if (sys_end > 0) {
            RenderResult sys_r = render_prompt(
                std::vector<Message>(req.messages.begin(),
                                     req.messages.begin() + sys_end),
                {}, sess.model().chat_templates(),
                {}, {}, true, /*enable_thinking=*/!has_rf, reasoning_fmt);
            const std::vector<llama_token> sys_toks = common_tokenize(
                sess.ctx(), sys_r.prompt, /*add_special=*/true,
                /*parse_special=*/true);
            n_keep = static_cast<int>(compute_lcp(prompt_tokens, sys_toks));
        }
        shift_cfg.n_keep = std::min(n_keep, n_ctx_size / 2);
    }


    /* 10. Run completions. */
    int  total_completion_tokens = 0;
    int  total_reasoning_tokens  = 0;
    bool any_cancelled           = false;
    bool gen_shifted             = false;
    int  gen_evicted             = 0;
    std::vector<llama_token> last_gen_tokens;
    std::vector<llama_token> shifted_cache;

    for (int run = 0; run < n_choices; ++run) {
        /* For multimodal, prefill is already done; skip text prefill in decode_one.
         * For subsequent choices (run > 0), we can't re-run multimodal prefill cheaply,
         * so multimodal requests are effectively n=1. */
        const bool skip_prefill = multimodal_prefill_done && (run == 0);
        const size_t run_lcp    = (run == 0 && !multimodal_prefill_done) ? lcp : 0;

        unsigned int run_seed = 0;
        if (req.seed) {
            run_seed = static_cast<unsigned int>(
                (*req.seed * 2654435761u) ^ static_cast<unsigned int>(run));
        }

        SamplerPtr sampler = build_sampler_chain(req, run_seed, vocab,
                                                  sampler_grammar,
                                                  sampler_grammar_lazy,
                                                  sampler_triggers,
                                                  resolved_biases,
                                                  req.reasoning_budget.value_or(-1),
                                                  rendered.thinking_start_tag,
                                                  rendered.thinking_end_tag,
                                                  rendered.thinking_forced_open);

        /* Pass mm_n_past_start so decode_one uses explicit positions for each
         * generated token; otherwise ISWA's seq_pos_max() mis-reports after
         * M-RoPE image embedding and the KV consecutive-position check fails. */
        const llama_pos n_past_mm_arg = skip_prefill ? mm_n_past_start : -1;

        /* Route to the speculative decode path when MTP is active and this is
         * a text-only request. Multimodal requests fall back to decode_one
         * (the MTP head cannot consume image/audio embedding positions). */
        DecodeResult result;
        if (sess.has_speculative() && !multimodal_prefill_done && !has_multimodal_content) {
            result = decode_one_speculative(
                sess, prompt_tokens, max_new, rendered.stop_strings,
                sampler.get(), sess.cancel_flag(),
                sink, run, sess.stream_coalesce_ms(),
                rendered, lp_cfg);
        } else {
            result = decode_one(
                sess, prompt_tokens, max_new, rendered.stop_strings,
                sampler.get(), sess.cancel_flag(),
                sink, run, run_lcp, sess.stream_coalesce_ms(),
                rendered, lp_cfg,
                skip_prefill,
                n_past_mm_arg,
                shift_cfg);
        }

        if (result.evicted_tokens > 0) {
            gen_shifted    = true;
            gen_evicted   += result.evicted_tokens;
            shifted_cache  = std::move(result.cache_tokens);
        }

        sink.on_finish(run, result.finish_reason);

        // Attach logprobs to non-streaming response (phase 6).
        if (lp_cfg.enabled && !streaming && !result.logprobs.empty()) {
            while (static_cast<int>(response.choices.size()) <= run) {
                Choice c;
                c.index = static_cast<int>(response.choices.size());
                response.choices.push_back(std::move(c));
            }
            ChoiceLogprobs clp;
            clp.content = std::move(result.logprobs);
            response.choices[run].logprobs = std::move(clp);
        }

        total_completion_tokens += result.n_predicted;
        total_reasoning_tokens  += result.reasoning_tokens;
        last_gen_tokens = std::move(result.tokens);

        if (result.cancelled) {
            any_cancelled = true;
            break;
        }
    }

    /* 11. Usage and stream finalisation. */
    Usage usage;
    usage.prompt_tokens     = n_prompt;
    usage.completion_tokens = total_completion_tokens;
    usage.total_tokens      = n_prompt + total_completion_tokens;
    if (total_reasoning_tokens > 0) {
        CompletionTokensDetails ctd;
        ctd.reasoning_tokens = total_reasoning_tokens;
        usage.completion_tokens_details = ctd;
    }

    /* Context-shift report (1.6): callers must be able to detect silent
     * history loss, whether it happened at prefill (dropped messages) or
     * during generation (evicted KV blocks). */
    if (prefill_truncated || gen_shifted) {
        sink.on_context_shift(prefill_evicted + gen_evicted);
    }

    sink.on_usage(usage);
    sink.on_stream_done();

    /* 12. Update prefix cache.
     *
     * The prefix cache is only sound when last_tokens_ exactly mirrors the
     * contents of KV seq 0, so that the next request can keep [0, lcp) and
     * trust that those KV cells correspond to the matched token prefix.
     *
     * That invariant holds ONLY after a clean, text-only completion:
     *   - On cancellation the KV holds a partial, unknown prefill/generation
     *     state that no longer matches any token sequence we can reconstruct.
     *   - After a multimodal prefill the KV holds image-embedding cells that
     *     cannot be represented as a token prefix at all (and re-using them as
     *     if they were text tokens would corrupt a subsequent request).
     *
     * In both of those cases we must INVALIDATE the cache (clear last_tokens_)
     * rather than leave it describing stale KV. The next request then rebuilds
     * the KV from scratch (text path: lcp==0 ⇒ full seq_rm; multimodal path:
     * KV is cleared before eval_media). */
    if (sess.prefix_cache_enabled() && !any_cancelled && !multimodal_prefill_done) {
        if (gen_shifted) {
            /* Generation-time eviction changed the KV contents; decode_one's
             * cache_tokens mirrors the post-shift cells exactly. */
            sess.set_last_tokens(std::move(shifted_cache));
        } else {
            std::vector<llama_token> cached = prompt_tokens;
            cached.insert(cached.end(), last_gen_tokens.begin(), last_gen_tokens.end());
            sess.set_last_tokens(std::move(cached));
        }
    } else {
        /* Cache disabled or invariant broken (cancel / multimodal KV):
         * clear rather than copy a token history nothing will read. */
        sess.set_last_tokens({});
    }

    /* 13. Serialise non-streaming response. */
    if (!streaming && out_response_json) {
        std::string json_str = response.to_json();
        /* Allocated with C malloc() so the caller frees it via helix_free()
         * (which calls free()). Keep these two paired — see helix_free(). */
        *out_response_json = static_cast<char*>(malloc(json_str.size() + 1));
        if (!*out_response_json) throw std::bad_alloc();
        std::copy_n(json_str.c_str(), json_str.size() + 1, *out_response_json);
    }

    return HELIX_OK;
}

helix_status_t run_count_tokens(Session& sess,
                                const char* request_json,
                                uint32_t* out_token_count) {
    /* Parse + validate exactly like a real request, so a body that would be
     * rejected by helix_chat_completions is rejected here with the same
     * error. Generation parameters pass validation but do not affect the
     * count. */
    ChatRequest req = ChatRequest::from_json(request_json ? request_json : "");
    req.validate();

    if (req.model != sess.model().alias()) {
        throw Error(HELIX_E_MODEL_NOT_FOUND,
                    "model '" + req.model + "' not loaded in this session",
                    "model_not_found", "model", "helix_e_model_not_found");
    }

    /* v1 limitation: media patch-token counting needs the projector; reject
     * honestly rather than report a text-only count that undercounts. */
    for (const auto& msg : req.messages) {
        for (const auto& part : msg.content_parts) {
            if (part.type == "image_url" || part.type == "input_audio") {
                throw_unsupported(
                    "helix_count_tokens does not support image/audio content parts",
                    "messages");
            }
        }
    }

    /* Mirror run_chat_completions: response_format disables thinking, and
     * the model's reasoning_format shapes the rendered prompt. */
    const bool has_rf = req.response_format &&
                        req.response_format->type != "text";
    RenderResult rendered = render_prompt(
        req.messages, req.stop, sess.model().chat_templates(),
        req.tools, req.tool_choice, req.parallel_tool_calls,
        /*enable_thinking=*/!has_rf,
        sess.model().options().reasoning_format);

    const std::vector<llama_token> toks = common_tokenize(
        sess.ctx(), rendered.prompt, /*add_special=*/true, /*parse_special=*/true);

    *out_token_count = static_cast<uint32_t>(toks.size());
    return HELIX_OK;
}

} // namespace helix
