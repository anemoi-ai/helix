#include "../session/session.hpp"
#include "../model/model.hpp"
#include "../json/request.hpp"
#include "../json/response.hpp"
#include "../chat/template.hpp"
#include "../chat/reasoning.hpp"
#include "../multimodal/image_decode.hpp"
#include "../multimodal/mmproj.hpp"
#include "../sampling/chain.hpp"
#include "../sampling/logit_bias.hpp"
#include "../grammar/response_format.hpp"
#include "../internal/caller_alloc.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "../internal/version.hpp"
#include "../runtime/runtime.hpp"
#include "decode_loop.hpp"
#include "llama_batch_util.hpp"
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
#include <cstdio>
#include <cstring>
#include <optional>
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

    /* Cancelled before any token was generated. */
    static DecodeResult cancelled_before_output() {
        DecodeResult r;
        r.finish_reason = "stop";
        r.cancelled     = true;
        return r;
    }
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

/* Per-choice output state shared by decode_one and decode_one_speculative:
 * stop/length detection, logprobs, the reasoning extractor, tool-call
 * parsing and streaming to the sink. The decoders only own KV and batch
 * management.
 *
 * Each sampled token goes through append() (EOG / stop / length and logprob
 * capture) and then emit() (release whatever became safe to stream). They
 * are separate because decode_one feeds the token back into the KV between
 * the two; push() runs both back to back for the speculative path. */
class TokenEmitter {
public:
    TokenEmitter(Session& sess, EventSink& sink, int choice_index,
                 const RenderResult& rendered, const LogprobsConfig& lp_cfg,
                 const std::vector<std::string>& stops, int max_new_tokens,
                 int stream_coalesce_ms, std::atomic<bool>& cancel_flag);

    /* Appends `tk` to the output and records its logprob from `logits`
     * (may be null). Returns true when generation stops at this token. */
    bool append(llama_token tk, const float* logits);

    /* Streams the bytes the last append() made safe to release. Returns true
     * when the request was cancelled meanwhile (e.g. from the callback). */
    bool emit();

    bool push(llama_token tk, const float* logits) { return append(tk, logits) || emit(); }

    void mark_cancelled() {
        cancelled_     = true;
        finish_reason_ = "stop";
    }

    /* Final parse / flush and the reasoning token count. */
    DecodeResult finish();

private:
    llama_context*                  ctx_;
    const llama_vocab*              vocab_;
    EventSink&                      sink_;
    const int                       choice_;
    const LogprobsConfig&           lp_cfg_;
    const std::vector<std::string>& stops_;
    const int                       max_new_tokens_;
    const int                       stream_coalesce_ms_;
    std::atomic<bool>&              cancel_flag_;

    bool                      has_parser_;
    common_peg_arena          peg_arena_;
    common_chat_parser_params pp_;
    common_chat_msg           prev_parsed_;
    int                       tool_calls_started_ = 0;

    /* Splits the template's thinking block (default <think>...</think>) from
     * regular content. thinking_forced_open: the prompt already opened the
     * think block, so generation starts mid-thought (only the closing tag
     * will appear). */
    ReasoningExtractor reasoning_ex_;
    std::string        reasoning_text_; /* re-tokenized in finish() for an
                                         * accurate reasoning_tokens count */
    int                lp_n_vocab_ = 0;

    std::string                    output_;
    std::string                    finish_reason_ = "stop";
    int                            n_predicted_   = 0;
    bool                           cancelled_     = false;
    std::vector<llama_token>       gen_tokens_;
    std::vector<TokenLogprobEntry> logprob_entries_;
    std::vector<TokenLogprobEntry> lp_pending_; /* entries awaiting content release */
    TokenLogprobEntry              last_lp_;
    bool                           last_has_logits_ = false;

    size_t emitted_up_to_ = 0; /* content-only path: bytes sent to sink */
    size_t stop_scan_     = 0; /* incremental scan offset for check_stops */
    std::chrono::steady_clock::time_point last_emit_ = std::chrono::steady_clock::now();

    bool check_cancel() {
        if (!cancel_flag_.load(std::memory_order_relaxed)) return false;
        mark_cancelled();
        return true;
    }
    void on_reasoning(const std::string& text) {
        if (text.empty()) return;
        sink_.on_reasoning_delta(choice_, text);
        reasoning_text_ += text;
    }
    /* Queued logprob entries ride on the next content release, when allowed. */
    void on_content(const std::string& content, bool with_logprobs) {
        if (content.empty()) return;
        if (with_logprobs && !lp_pending_.empty()) {
            sink_.on_token_with_logprob(choice_, content, lp_pending_);
            lp_pending_.clear();
        } else {
            sink_.on_content(choice_, content);
        }
    }
    /* Pushes output_[emitted_up_to_, upto) through the reasoning extractor. */
    void release(size_t upto, bool with_logprobs) {
        auto rout = reasoning_ex_.push(std::string_view(output_.data() + emitted_up_to_,
                                                        upto - emitted_up_to_));
        on_reasoning(rout.reasoning);
        on_content(rout.content, with_logprobs);
        emitted_up_to_ = upto;
    }
};

TokenEmitter::TokenEmitter(Session& sess, EventSink& sink, int choice_index,
                           const RenderResult& rendered, const LogprobsConfig& lp_cfg,
                           const std::vector<std::string>& stops, int max_new_tokens,
                           int stream_coalesce_ms, std::atomic<bool>& cancel_flag)
    : ctx_(sess.ctx())
    , vocab_(llama_model_get_vocab(sess.model().llama_model_ptr()))
    , sink_(sink)
    , choice_(choice_index)
    , lp_cfg_(lp_cfg)
    , stops_(stops)
    , max_new_tokens_(max_new_tokens)
    , stream_coalesce_ms_(stream_coalesce_ms)
    , cancel_flag_(cancel_flag)
    , has_parser_(rendered.has_tools && rendered.parser != nullptr)
    , reasoning_ex_(rendered.extract_reasoning,
                    rendered.thinking_forced_open,
                    resolve_thinking_start_tag(rendered.thinking_start_tag,
                                               rendered.thinking_end_tag),
                    rendered.thinking_end_tag) {
    if (has_parser_) {
        /* A copy, not a reload: the arena is a plain value since llama.cpp
         * b12033. There is no parse-time reasoning_format any more either:
         * the reasoning split is decided when the template is applied
         * (common_chat_templates_inputs::reasoning_format, set in
         * render_prompt), and the arena built for it does the splitting. */
        peg_arena_            = *rendered.parser;
        pp_.format            = static_cast<common_chat_format>(rendered.chat_format);
        pp_.generation_prompt = common_chat_input(rendered.generation_prompt);
    }
    if (lp_cfg_.enabled) lp_n_vocab_ = llama_vocab_n_tokens(vocab_);
}

bool TokenEmitter::append(llama_token tk, const float* logits) {
    if (llama_vocab_is_eog(vocab_, tk)) {
        finish_reason_ = "stop";
        return true;
    }
    output_ += common_token_to_piece(ctx_, tk, /*special=*/false);
    gen_tokens_.push_back(tk);
    ++n_predicted_;

    last_has_logits_ = lp_cfg_.enabled && logits;
    last_lp_         = TokenLogprobEntry{};
    if (last_has_logits_) {
        last_lp_ = compute_logprob_entry(logits, lp_n_vocab_, tk, ctx_,
                                         lp_cfg_.temperature, lp_cfg_.top_n);
        logprob_entries_.push_back(last_lp_);
    }

    /* Stop strings are checked against the raw output on every path; for
     * tool-aware paths the grammar keeps the output well-formed, so template
     * stops fire at the right place (EOS / end-of-turn tokens). */
    const size_t safe     = valid_utf8_prefix_len(output_.data(), output_.size());
    const size_t stop_pos = check_stops(output_, safe, stops_, &stop_scan_);
    if (stop_pos != std::string::npos) {
        output_.resize(stop_pos);
        finish_reason_ = "stop";
        return true;
    }
    if (n_predicted_ >= max_new_tokens_) {
        finish_reason_ = "length";
        return true;
    }
    return false;
}

bool TokenEmitter::emit() {
    /* Both paths hold back any output suffix that is a prefix of a stop
     * string — a stop match can span token boundaries, and once bytes reach
     * the sink they cannot be retracted. Held-back bytes that never complete
     * a stop are released on a later token or the final flush. */
    const size_t new_safe = valid_utf8_prefix_len(output_.data(), output_.size());
    const size_t emit_len = new_safe - partial_stop_len(output_, new_safe, stops_);

    if (lp_cfg_.enabled && !has_parser_ && last_has_logits_) {
        /* Per-token emission (logprobs on the content-only path). Every entry
         * accrued since the last release rides on whatever content the
         * extractor releases now; entries are queued rather than dropped
         * because the extractor's partial-tag lookahead (and stop-string
         * holdback) can withhold a token's bytes until a later step — or
         * until the final flush. */
        lp_pending_.push_back(last_lp_);
        if (emit_len > emitted_up_to_) release(emit_len, /*with_logprobs=*/true);
        last_emit_ = std::chrono::steady_clock::now();
        return check_cancel();
    }

    /* Coalesced emission. */
    if (emit_len <= (has_parser_ ? 0 : emitted_up_to_)) return false;
    if (stream_coalesce_ms_ > 0) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - last_emit_).count();
        if (ms < stream_coalesce_ms_) return false;
    }
    if (has_parser_ && !peg_arena_.empty()) {
        try {
            common_chat_msg current = common_chat_peg_parse(
                peg_arena_, common_chat_input(output_.substr(0, emit_len)),
                /*is_partial=*/true, pp_);
            auto diffs = common_chat_msg_diff::compute_diffs(prev_parsed_, current);
            route_diffs(diffs, sink_, choice_, tool_calls_started_, &reasoning_text_);
            prev_parsed_ = std::move(current);
        } catch (const std::exception& e) {
            log_warn("PEG parse error during streaming: " + std::string(e.what()));
        }
    } else {
        release(emit_len, /*with_logprobs=*/false);
    }
    last_emit_ = std::chrono::steady_clock::now();
    return check_cancel();
}

DecodeResult TokenEmitter::finish() {
    if (has_parser_ && !peg_arena_.empty() && !cancelled_) {
        /* Final parse (non-partial): extracts complete tool calls and remaining content. */
        try {
            common_chat_msg final_msg = common_chat_peg_parse(
                peg_arena_, common_chat_input(output_), /*is_partial=*/false, pp_);
            auto diffs = common_chat_msg_diff::compute_diffs(prev_parsed_, final_msg);
            route_diffs(diffs, sink_, choice_, tool_calls_started_, &reasoning_text_);

            if (tool_calls_started_ > 0) {
                sink_.on_tool_call_end(choice_, tool_calls_started_ - 1);
                finish_reason_ = "tool_calls";
            }
        } catch (const std::exception& e) {
            log_warn("PEG final parse error: " + std::string(e.what()));
            /* Fall through with whatever content the sink already received. */
        }
    } else if (!has_parser_) {
        /* Content-only: push the remaining safe bytes through the reasoning
         * extractor, then flush its lookahead buffer (partial tag text).
         * Queued logprob entries ride on the first content release so short
         * generations (fully held back by the lookahead) still deliver their
         * logprobs. */
        const size_t final_safe = valid_utf8_prefix_len(output_.data(), output_.size());
        if (final_safe > emitted_up_to_) release(final_safe, /*with_logprobs=*/true);
        auto flush_rout = reasoning_ex_.flush();
        on_reasoning(flush_rout.reasoning);
        on_content(flush_rout.content, /*with_logprobs=*/true);

        if (!reasoning_text_.empty()) sink_.on_reasoning_end(choice_);
    }
    /* Note: if cancelled with a parser, the final parse is skipped — any
     * pending tool call (started but not ended) is dropped, and finish_reason
     * stays "stop". */

    DecodeResult r;
    /* Reasoning token count: re-tokenize the extracted reasoning text with the
     * model's own tokenizer so the count is BPE-accurate (matching how usage
     * is reported by OpenAI), rather than a chars/4 heuristic. add_special and
     * parse_special are false: the reasoning text carries no BOS and the
     * thinking tags have already been stripped by the extractor. */
    if (!reasoning_text_.empty()) {
        r.reasoning_tokens = static_cast<int>(
            common_tokenize(ctx_, reasoning_text_, /*add_special=*/false,
                            /*parse_special=*/false).size());
    }
    r.content       = std::move(output_);
    r.finish_reason = finish_reason_;
    r.n_predicted   = n_predicted_;
    r.cancelled     = cancelled_;
    r.tokens        = std::move(gen_tokens_);
    r.logprobs      = std::move(logprob_entries_);
    return r;
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
            return DecodeResult::cancelled_before_output();
        }

        /* ---- Prefill ---- */
        for (int i = static_cast<int>(lcp); i < n_prompt; i += n_batch) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                return DecodeResult::cancelled_before_output();
            }
            int chunk_size = std::min(n_batch, n_prompt - i);
            llama_batch batch = llama_batch_get_one(
                const_cast<llama_token*>(prompt_tokens.data()) + i, chunk_size);
            if (llama_decode(ctx, batch) != 0) {
                throw_backend("llama_decode failed during prefill");
            }
        }
    } else {
        /* Multimodal path: prefill was done externally by eval_media. */
        if (cancel_flag.load(std::memory_order_relaxed)) {
            return DecodeResult::cancelled_before_output();
        }
    }

    sink.on_role(choice_index);
    if (cancel_flag.load(std::memory_order_relaxed)) {
        return DecodeResult::cancelled_before_output();
    }

    TokenEmitter emitter(sess, sink, choice_index, rendered, lp_cfg, stops,
                         max_new_tokens, stream_coalesce_ms, cancel_flag);

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
            emitter.mark_cancelled();
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

        if (emitter.append(new_token, lp_raw)) break;

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
            batch_clear(mm_batch);
            batch_add(mm_batch, new_token, mm_n_past++, {0}, true);
            next_batch = mm_batch;
        } else {
            next_batch = llama_batch_get_one(&new_token, 1);
        }
        if (llama_decode(ctx, next_batch) != 0) {
            throw_backend("llama_decode failed during generation");
        }
        ++n_past;
        if (shift.enabled) cache_tokens.push_back(new_token);

        if (emitter.emit()) break;
    }

    DecodeResult result  = emitter.finish();
    result.evicted_tokens = evicted_total;
    result.cache_tokens   = std::move(cache_tokens);
    return result;
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
    llama_context* ctx_mtp = sess.ctx_mtp();
    common_speculative* spec = sess.spec_mtp();
    const int n_prompt = static_cast<int>(prompt_tokens.size());
    const int n_batch  = llama_n_batch(ctx);

    /* The draft context's KV is the caller's to keep in step with the
     * target's (llama.cpp b12033). Upstream's draft-MTP trims ctx_dft itself
     * only when it chains several MTP heads; for one head on a separate KV
     * (e.g. Qwen3.5-family MTP GGUFs) the speculative positions it decodes
     * stay behind unless the caller removes them, as speculative-simple.cpp
     * and the server do. Left there, the next catch-up decode and draft seed
     * collide with them: fatal on M-RoPE models ("inconsistent sequence
     * positions"), silent draft degradation otherwise. This replaces the
     * helix-1.4 fork patch 2ee83d0c3, which did the same inside llama.cpp.
     *
     * Unguarded for shared-KV drafts (Gemma-4 assistants), as upstream's
     * callers do it: every trim below starts at or past the target's own
     * next position, so on a shared cache it removes nothing the target
     * holds. The return value is ignored, as the fork patch did: a failed
     * trim can only cost draft quality, since the target verifies every
     * token. */
    auto trim_mtp = [ctx_mtp](llama_pos from) {
        if (ctx_mtp) llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, from, -1);
    };

    /* ---- Prefill (text-only; the speculative path is never used for
     * multimodal — run_chat_completions falls back to decode_one). ---- */
    {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, 0, 0, -1);
        trim_mtp(0);
    }
    if (cancel_flag.load(std::memory_order_relaxed))
        return DecodeResult::cancelled_before_output();

    /* Every batch shown to common_speculative_process is a common_batch,
     * decoded with llama_process: the speculative state reads per-token
     * entries from the common_batch mirror (llama.cpp b12033), so the batch
     * it is given must be the very one that was decoded. */
    common_batch pbatch(ctx);
    for (int i = 0; i < n_prompt; i += n_batch) {
        if (cancel_flag.load(std::memory_order_relaxed))
            return DecodeResult::cancelled_before_output();
        int chunk_size = std::min(n_batch, n_prompt - i);
        pbatch.clear();
        for (int j = 0; j < chunk_size; ++j) {
            /* Only the final prompt token needs logits (for sampling the
             * first output token). */
            bool is_last = (i + j == n_prompt - 1);
            pbatch.add(prompt_tokens[i + j], static_cast<llama_pos>(i + j),
                       /*seq_id=*/0, /*output=*/is_last);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, pbatch.get()) != 0)
            throw_backend("llama_decode failed during prefill");
        /* Mirror the prefill chunk into the MTP draft context so its KV
         * state matches the target before drafting begins. */
        if (!common_speculative_process(spec, pbatch))
            throw_backend("common_speculative_process failed during prefill");
    }

    sink.on_role(choice_index);
    if (cancel_flag.load(std::memory_order_relaxed))
        return DecodeResult::cancelled_before_output();

    TokenEmitter emitter(sess, sink, choice_index, rendered, lp_cfg, stops,
                         max_new_tokens, stream_coalesce_ms, cancel_flag);

    /* ---- common_speculative_begin: notify the manager that prefill is done */
    common_speculative_begin(spec, /*seq_id=*/0, prompt_tokens);

    /* ---- Sample the first token from the prefill logits (no draft for it) */
    const float* lp_raw_first = lp_cfg.enabled ? llama_get_logits_ith(ctx, -1) : nullptr;
    llama_token id_last = llama_sampler_sample(sampler, ctx, -1);
    llama_pos   n_past  = static_cast<llama_pos>(n_prompt);

    if (emitter.push(id_last, lp_raw_first)) return emitter.finish();

    /* ---- Speculative decode loop ---- */
    common_batch sbatch(ctx);
    /* Draft bookkeeping, logged at the end: helix reports no acceptance rate
     * through the ABI, and it is the number that says whether MTP is worth
     * running at all. */
    size_t n_drafted = 0, n_accepted = 0, n_rounds = 0;

    for (;;) {
        if (cancel_flag.load(std::memory_order_relaxed)) {
            emitter.mark_cancelled();
            break;
        }

        /* 1. Request a draft. */
        std::vector<llama_token> draft;
        {
            /* Named fields, not a positional initialiser: the struct was
             * reordered and grew (b12033: n_past -> pos0; result_q, temp,
             * seed). result_q stays null: greedy drafting, as before. */
            auto& dp    = common_speculative_get_draft_params(spec, /*seq_id=*/0);
            dp.drafting = true;
            dp.n_max    = -1;                /* impl clamps to remaining ctx */
            dp.pos0     = n_past;
            dp.id_last  = id_last;
            dp.prompt   = &prompt_tokens;
            dp.result   = &draft;
            dp.result_q = nullptr;
        }
        common_speculative_draft(spec);

        /* Drafting decoded speculative positions [n_past, ...) into the draft
         * context; drop them before the catch-up decode below rewrites those
         * positions from the target's hidden states (see trim_mtp). */
        trim_mtp(n_past);

        /* If no draft (e.g. context exhausted), fall back to a plain
         * single-token decode to make progress. */
        if (draft.empty()) {
            sbatch.clear();
            sbatch.add(id_last, n_past, /*seq_id=*/0, /*output=*/true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, sbatch.get()) != 0)
                throw_backend("llama_decode failed during MTP fallback");
            if (!common_speculative_process(spec, sbatch))
                throw_backend("common_speculative_process failed during MTP fallback");
            llama_token next = llama_sampler_sample(sampler, ctx, -1);
            llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + 1, -1);
            trim_mtp(n_past + 1);
            common_speculative_accept(spec, 0, 0);
            if (emitter.push(next, lp_cfg.enabled ? llama_get_logits_ith(ctx, 0) : nullptr))
                break;
            n_past += 1;
            id_last = next;
            continue;
        }

        /* 2. Build target batch: [id_last, draft0, ..., draftN-1]. */
        std::vector<int> spec_i_batch;
        spec_i_batch.reserve(draft.size() + 1);
        sbatch.clear();
        spec_i_batch.push_back(sbatch.add(id_last, n_past, /*seq_id=*/0, /*output=*/true));
        for (size_t i = 0; i < draft.size(); ++i) {
            spec_i_batch.push_back(sbatch.add(draft[i], n_past + static_cast<llama_pos>(i) + 1,
                                              /*seq_id=*/0, /*output=*/true));
        }

        /* 3. Target decode. */
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, sbatch.get()) != 0)
            throw_backend("llama_decode failed during MTP generation");

        /* 4. Mirror target hidden states into the draft context. */
        if (!common_speculative_process(spec, sbatch))
            throw_backend("common_speculative_process failed");

        /* 5. Verify draft against the target via the raw sampler chain. */
        auto accepted = helix_sample_and_accept_n(sampler, ctx, spec_i_batch, draft);
        n_drafted  += draft.size();
        n_accepted += accepted.size() - 1;
        ++n_rounds;

        /* 6. KV cleanup: remove rejected draft positions. The last accepted
         *    token is only sampled (not decoded) so it stays out of the KV
         *    and becomes id_last for the next iteration. */
        llama_memory_seq_rm(llama_get_memory(ctx), 0,
                            n_past + static_cast<llama_pos>(accepted.size()), -1);
        trim_mtp(n_past + static_cast<llama_pos>(accepted.size()));

        /* 7. Inform the manager how many drafts were accepted. */
        common_speculative_accept(spec, /*seq_id=*/0,
                                  static_cast<uint16_t>(accepted.size() - 1));

        /* 8. Emit each accepted token. */
        bool stop = false;
        for (size_t i = 0; i < accepted.size() && !stop; ++i) {
            const float* tk_logits = lp_cfg.enabled
                ? llama_get_logits_ith(ctx, spec_i_batch[i]) : nullptr;
            stop = emitter.push(accepted[i], tk_logits);
        }
        if (stop) break;

        /* 9. Advance. */
        n_past  += static_cast<llama_pos>(accepted.size());
        id_last  = accepted.back();
    }

    if (n_rounds > 0) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "mtp: %zu rounds, %zu drafted, %zu accepted (%.1f%%)",
                      n_rounds, n_drafted, n_accepted,
                      n_drafted ? 100.0 * static_cast<double>(n_accepted) / n_drafted : 0.0);
        log_info(buf);
    }
    return emitter.finish();
}
/* ------------------------------------------------------------------ */

/* Renders `messages` with the request's stops, tools and tool_choice. Every
 * path that tokenizes a request goes through here: count_tokens must render
 * exactly as the chat path does. response_format disables thinking — the
 * grammar constrains output directly, and a thinking preamble would exhaust
 * the token budget. */
static RenderResult render_for_request(const Session& sess, const ChatRequest& req,
                                       const std::vector<Message>& messages) {
    const bool has_rf = req.response_format && req.response_format->type != "text";
    return render_prompt(messages, req.stop, sess.model().chat_templates(),
                         req.tools, req.tool_choice, req.parallel_tool_calls,
                         /*enable_thinking=*/!has_rf,
                         sess.model().options().reasoning_format);
}

/* Number of leading system/developer messages (the prefix context shift
 * never evicts). */
static size_t leading_system_count(const std::vector<Message>& msgs) {
    size_t n = 0;
    while (n < msgs.size() &&
           (msgs[n].role == "system" || msgs[n].role == "developer")) {
        ++n;
    }
    return n;
}

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
    sess.require_model_alias(req.model);

    /* 4. Multimodal capability check: reject vision/audio if model has no mmproj. */
    bool has_multimodal_content = false;
    for (const auto& msg : req.messages) {
        if (msg.content_parts.empty()) continue;
        for (const auto& part : msg.content_parts) {
            if (part.is_media()) {
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

    /* 5. Render prompt (with tools if present). */
    const bool has_rf = req.response_format &&
                        req.response_format->type != "text";
    RenderResult rendered = render_for_request(sess, req, req.messages);

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
                if (part.is_media()) {
                    all_media.push_back(part.media_file_uri.empty()
                                            ? part.media_raw
                                            : load_file_uri(part.media_file_uri));
                }
            }
        }

        /* Clear KV for seq 0 before multimodal prefill — eval_media starts at
         * n_past=0 and the KV may still hold state from a prior request.
         * Invalidate the prefix cache first: if eval_media throws (corrupt
         * media, marker mismatch), the KV no longer holds the prior
         * conversation and a stale last_tokens_ would make the next text
         * request reuse cells that are gone. */
        sess.set_last_tokens({});
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
            throw_cancelled("cancelled during multimodal prefill");
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

            const size_t sys_end = leading_system_count(msgs);
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

                rendered = render_for_request(sess, req, msgs);
                prompt_tokens = common_tokenize(
                    ctx, rendered.prompt, /*add_special=*/true,
                    /*parse_special=*/true);
                n_prompt = static_cast<int>(prompt_tokens.size());
            }

            if (n_prompt >= n_ctx_size) {
                throw_context_full(n_prompt, n_ctx_size,
                                   "context_shift could not make the request fit: "
                                   "the system prompt + latest user message occupy " +
                                   std::to_string(n_prompt) + " of " +
                                   std::to_string(n_ctx_size) + " context tokens",
                                   "messages");
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
     * When a PEG parser is present (tools active), the tool grammar from the
     * template — not response_format — owns the sampler: llama.cpp derives
     * both the GBNF and the PEG parser from the same parse tree, so they
     * agree by construction. The grammar is what makes tool_choice:"required"
     * an actual guarantee (eager grammar) rather than a hint; for "auto" it
     * is lazy and only engages at the tool-call trigger marker, leaving
     * regular content and <think> output unconstrained. */
    const bool has_peg = rendered.has_tools && rendered.parser != nullptr;
    const llama_vocab* vocab = llama_model_get_vocab(sess.model().llama_model_ptr());

    std::string rf_grammar;
    if (req.response_format && req.response_format->type != "text" && !has_peg) {
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
    } else {
        sampler_grammar = rendered.grammar; /* empty unless tools are active */
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
        const size_t sys_end = leading_system_count(req.messages);
        if (sys_end > 0) {
            RenderResult sys_r = render_prompt(
                std::vector<Message>(req.messages.begin(),
                                     req.messages.begin() + sys_end),
                {}, sess.model().chat_templates(),
                {}, {}, true, /*enable_thinking=*/!has_rf,
                sess.model().options().reasoning_format);
            const std::vector<llama_token> sys_toks = common_tokenize(
                sess.ctx(), sys_r.prompt, /*add_special=*/true,
                /*parse_special=*/true);
            n_keep = static_cast<int>(compute_lcp(prompt_tokens, sys_toks));
        }
        shift_cfg.n_keep = std::min(n_keep, n_ctx_size / 2);
    }


    /* 10. Run completions. From here on the KV is mutated; drop the prefix
     * cache now so an exception mid-decode (backend failure, sink throw,
     * OOM) leaves last_tokens_ empty rather than describing stale KV. Step
     * 12 re-establishes it on success. */
    sess.set_last_tokens({});

    int  total_completion_tokens = 0;
    int  total_reasoning_tokens  = 0;
    bool any_cancelled           = false;
    bool gen_shifted             = false;
    int  gen_evicted             = 0;
    std::vector<llama_token> last_gen_tokens;
    std::vector<llama_token> shifted_cache;

    /* Runtime "deterministic": requests that omit "seed" behave as seed 0. */
    std::optional<uint32_t> base_seed = req.seed;
    if (!base_seed && sess.model().runtime().options().deterministic) {
        base_seed = 0u;
    }

    for (int run = 0; run < n_choices; ++run) {
        /* For multimodal, prefill is already done; skip text prefill in decode_one.
         * For subsequent choices (run > 0), we can't re-run multimodal prefill cheaply,
         * so multimodal requests are effectively n=1. */
        const bool skip_prefill = multimodal_prefill_done && (run == 0);
        /* Later choices reuse the prompt KV the first choice left behind:
         * keep [0, n_prompt - 1) and re-decode only the last prompt token to
         * regenerate its logits. decode_one falls back to a full re-prefill
         * if the memory cannot be partially removed. */
        size_t run_lcp = 0;
        if (!multimodal_prefill_done) {
            if (run == 0)          run_lcp = lcp;
            else if (n_prompt > 0) run_lcp = static_cast<size_t>(n_prompt - 1);
        }

        std::optional<uint32_t> run_seed;
        if (base_seed) {
            run_seed = static_cast<uint32_t>(
                (*base_seed * 2654435761u) ^ static_cast<unsigned int>(run));
        }

        SamplerPtr sampler = build_sampler_chain(req, run_seed, vocab,
                                                  sampler_grammar,
                                                  sampler_grammar_lazy,
                                                  sampler_triggers,
                                                  resolved_biases,
                                                  req.reasoning_budget.value_or(-1),
                                                  rendered.thinking_start_tag,
                                                  rendered.thinking_end_tag,
                                                  rendered.thinking_forced_open,
                                                  rendered.thinking_end_tags);

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
    /* Drain in-flight device work after a cancel. cancel() only sets a flag;
     * the decode loop observes it at the next iteration boundary and breaks,
     * so the last llama_decode() batch may still be mid-submit on the device
     * queue (asynchronous on Vulkan/CUDA). Without this drain the session is
     * not actually reusable despite the "cancelled session stays reusable"
     * contract — a follow-on request would race that pending batch — and a
     * subsequent drop would tear the device down with work still in flight
     * (the SIGSEGV/BSOD in HELIX_R7_CANCEL_TEARDOWN_BUG.md). The destructor
     * also drains defensively, but draining here keeps the cancelled session
     * sound for reuse. No-op on CPU. */
    if (any_cancelled) {
        if (sess.has_speculative() && sess.ctx_mtp()) {
            llama_synchronize(sess.ctx_mtp());
        }
        llama_synchronize(sess.ctx());
    }

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
        *out_response_json = dup_for_caller(response.to_json());
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

    sess.require_model_alias(req.model);

    /* v1 limitation: media patch-token counting needs the projector; reject
     * honestly rather than report a text-only count that undercounts. */
    for (const auto& msg : req.messages) {
        for (const auto& part : msg.content_parts) {
            if (part.is_media()) {
                throw_unsupported(
                    "helix_count_tokens does not support image/audio content parts",
                    "messages");
            }
        }
    }

    RenderResult rendered = render_for_request(sess, req, req.messages);

    const std::vector<llama_token> toks = common_tokenize(
        sess.ctx(), rendered.prompt, /*add_special=*/true, /*parse_special=*/true);

    *out_token_count = static_cast<uint32_t>(toks.size());
    return HELIX_OK;
}

} // namespace helix
