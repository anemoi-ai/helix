#pragma once
#include "../json/response.hpp"
#include "helix.h"
#include <atomic>
#include <map>
#include <string>
#include <string_view>

namespace helix {

/* ------------------------------------------------------------------ */
/*  Abstract event sink                                                */
/* ------------------------------------------------------------------ */

class EventSink {
public:
    virtual ~EventSink() = default;

    /* Called once per choice after prefill completes. */
    virtual void on_role(int choice_index) = 0;

    /* Called for each batch of safely-emittable content. */
    virtual void on_content(int choice_index, std::string_view delta) = 0;

    /* Phase 6: per-token emission with logprob info (bypasses coalescing).
     * Default implementation delegates to on_content (ignoring logprob). */
    virtual void on_token_with_logprob(int choice_index, std::string_view piece,
                                        const TokenLogprobEntry& /*info*/) {
        on_content(choice_index, piece);
    }

    /* Phase 7: reasoning content (default no-ops — only populated for reasoning models). */
    virtual void on_reasoning_delta(int /*choice_index*/, std::string_view /*delta*/) {}
    virtual void on_reasoning_end(int /*choice_index*/) {}

    /* Phase 3: tool call events (default no-ops so subclasses can opt in). */
    virtual void on_tool_call_start(int /*choice_index*/, int /*tc_index*/,
                                     const std::string& /*id*/,
                                     const std::string& /*name*/) {}
    virtual void on_tool_call_arg_delta(int /*choice_index*/, int /*tc_index*/,
                                         const std::string& /*delta*/) {}
    virtual void on_tool_call_end(int /*choice_index*/, int /*tc_index*/) {}

    /* Called when a choice finishes (reason = "stop" | "length" | "tool_calls"). */
    virtual void on_finish(int choice_index, std::string_view reason) = 0;

    /* 1.6: called once (before on_usage) when context_shift evicted history
     * during this request. Default no-op. */
    virtual void on_context_shift(int /*evicted_tokens*/) {}

    /* Called once after all choices complete; carries aggregate token counts. */
    virtual void on_usage(const Usage& usage) = 0;

    /* Called once at the very end of the stream (NULL sentinel for streaming). */
    virtual void on_stream_done() = 0;
};

/* ------------------------------------------------------------------ */
/*  CollectingSink — accumulates into a ChatResponse (non-streaming)   */
/* ------------------------------------------------------------------ */

class CollectingSink : public EventSink {
public:
    explicit CollectingSink(ChatResponse& response) : response_(response) {}

    void on_role(int /*choice_index*/) override {}

    void on_content(int choice_index, std::string_view delta) override {
        ensure_choice(choice_index);
        response_.choices[choice_index].message.content += delta;
    }

    void on_reasoning_delta(int choice_index, std::string_view delta) override {
        ensure_choice(choice_index);
        auto& rc = response_.choices[choice_index].message.reasoning_content;
        if (!rc) rc = "";
        *rc += delta;
    }

    /* Phase 3: accumulate tool calls.
     * Tool calls are "pending" until on_tool_call_end; only completed calls are
     * included in the response (partial calls dropped on cancel). */
    void on_tool_call_start(int choice_index, int /*tc_index*/,
                             const std::string& id,
                             const std::string& name) override {
        ensure_choice(choice_index);
        ToolCall tc;
        tc.id            = id;
        tc.type          = "function";
        tc.function_name = name;
        pending_[choice_index] = tc;
    }

    void on_tool_call_arg_delta(int choice_index, int /*tc_index*/,
                                 const std::string& delta) override {
        auto it = pending_.find(choice_index);
        if (it != pending_.end()) it->second.function_arguments += delta;
    }

    void on_tool_call_end(int choice_index, int /*tc_index*/) override {
        auto it = pending_.find(choice_index);
        if (it != pending_.end()) {
            ensure_choice(choice_index);
            response_.choices[choice_index].message.tool_calls.push_back(it->second);
            pending_.erase(it);
        }
    }

    void on_finish(int choice_index, std::string_view reason) override {
        ensure_choice(choice_index);
        response_.choices[choice_index].finish_reason = std::string(reason);
    }

    void on_context_shift(int evicted_tokens) override {
        response_.context_shifted = true;
        response_.evicted_tokens  = evicted_tokens;
    }

    void on_usage(const Usage& usage) override {
        response_.usage = usage;
    }

    void on_stream_done() override {}

private:
    ChatResponse& response_;
    std::map<int, ToolCall> pending_; /* in-progress (started but not ended) tool calls */

    void ensure_choice(int idx) {
        while (static_cast<int>(response_.choices.size()) <= idx) {
            Choice c;
            c.index = static_cast<int>(response_.choices.size());
            response_.choices.push_back(std::move(c));
        }
    }
};

/* ------------------------------------------------------------------ */
/*  StreamingSink — emits chat.completion.chunk JSON via callback      */
/* ------------------------------------------------------------------ */

class StreamingSink : public EventSink {
public:
    StreamingSink(helix_stream_cb cb, void* user_data,
                  const std::string& id, int64_t created,
                  const std::string& model, const std::string& fingerprint,
                  bool include_usage,
                  std::atomic<bool>& cancel_flag)
        : cb_(cb), user_data_(user_data)
        , id_(id), created_(created)
        , model_(model), fingerprint_(fingerprint)
        , include_usage_(include_usage)
        , cancel_flag_(cancel_flag) {}

    void on_role(int choice_index) override {
        if (cancelled_) return;
        DeltaContent d;
        d.role = "assistant";
        emit(make_chunk_json(id_, created_, model_, fingerprint_, choice_index, d));
    }

    void on_content(int choice_index, std::string_view delta) override {
        if (cancelled_) return;
        DeltaContent d;
        d.content = std::string(delta);
        emit(make_chunk_json(id_, created_, model_, fingerprint_, choice_index, d));
    }

    void on_reasoning_delta(int choice_index, std::string_view delta) override {
        if (cancelled_ || delta.empty()) return;
        DeltaContent d;
        d.reasoning_content = std::string(delta);
        emit(make_chunk_json(id_, created_, model_, fingerprint_, choice_index, d));
    }

    void on_token_with_logprob(int choice_index, std::string_view piece,
                                const TokenLogprobEntry& info) override {
        if (cancelled_) return;
        DeltaContent d;
        d.content = std::string(piece);
        ChoiceLogprobs lp;
        lp.content.push_back(info);
        emit(make_chunk_json(id_, created_, model_, fingerprint_, choice_index, d, "", &lp));
    }

    /* Phase 3: first chunk carries id, type, name, arguments="" */
    void on_tool_call_start(int choice_index, int tc_index,
                             const std::string& id,
                             const std::string& name) override {
        if (cancelled_) return;
        ToolCall tc;
        tc.index             = tc_index;
        tc.id                = id;
        tc.type              = "function";
        tc.function_name     = name;
        tc.function_arguments = "";
        DeltaContent d;
        d.tool_calls.push_back(tc);
        emit(make_chunk_json(id_, created_, model_, fingerprint_, choice_index, d));
    }

    /* Subsequent argument-delta chunks: id empty → serializer emits args-only shape */
    void on_tool_call_arg_delta(int choice_index, int tc_index,
                                 const std::string& delta) override {
        if (cancelled_ || delta.empty()) return;
        ToolCall tc;
        tc.index              = tc_index;
        /* id/type/name left empty → response.cpp detects arg-only shape */
        tc.function_arguments = delta;
        DeltaContent d;
        d.tool_calls.push_back(tc);
        emit(make_chunk_json(id_, created_, model_, fingerprint_, choice_index, d));
    }

    void on_tool_call_end(int /*choice_index*/, int /*tc_index*/) override {
        /* No explicit chunk for tool_call_end; the closing finish chunk carries
         * finish_reason:"tool_calls" which signals completion to the client. */
    }

    /* Emit closing chunk even when cancelled — the client must see finish_reason. */
    void on_finish(int choice_index, std::string_view reason) override {
        if (!cb_) return;
        DeltaContent d;
        std::string json = make_chunk_json(id_, created_, model_, fingerprint_,
                                           choice_index, d, std::string(reason));
        cb_(user_data_, json.c_str());
    }

    void on_context_shift(int evicted_tokens) override {
        /* Always emitted (not gated on include_usage): silent history loss
         * is exactly what a streaming client needs to detect. */
        if (!cb_) return;
        std::string json = make_shift_chunk_json(id_, created_, model_,
                                                 fingerprint_, evicted_tokens);
        cb_(user_data_, json.c_str());
    }

    void on_usage(const Usage& usage) override {
        if (!include_usage_ || !cb_) return;
        std::string json = make_usage_chunk_json(id_, created_, model_, fingerprint_, usage);
        cb_(user_data_, json.c_str());
    }

    void on_stream_done() override {
        if (cb_) cb_(user_data_, nullptr);
    }

    bool is_cancelled() const { return cancelled_; }

private:
    helix_stream_cb    cb_;
    void*              user_data_;
    std::string        id_;
    int64_t            created_;
    std::string        model_;
    std::string        fingerprint_;
    bool               include_usage_;
    std::atomic<bool>& cancel_flag_;
    bool               cancelled_ = false;

    void emit(const std::string& json) {
        if (!cb_ || cancelled_) return;
        if (cb_(user_data_, json.c_str()) != 0) {
            cancel_flag_.store(true, std::memory_order_relaxed);
            cancelled_ = true;
        }
    }
};

} // namespace helix
