#include <gtest/gtest.h>
#include "engine/event_sink.hpp"
#include "json/response.hpp"
#include "nlohmann/json.hpp"

#include <atomic>
#include <string>
#include <vector>

using helix::ChatResponse;
using helix::CollectingSink;
using helix::StreamingSink;
using helix::Usage;
using json = nlohmann::json;

/* ------------------------------------------------------------------ */
/*  CollectingSink                                                     */
/* ------------------------------------------------------------------ */

TEST(CollectingSink, AccumulatesContent) {
    ChatResponse resp;
    CollectingSink sink(resp);

    sink.on_role(0);
    sink.on_content(0, "Hello");
    sink.on_content(0, ", ");
    sink.on_content(0, "world");
    sink.on_finish(0, "stop");

    ASSERT_EQ(resp.choices.size(), 1u);
    EXPECT_EQ(resp.choices[0].message.content, "Hello, world");
    EXPECT_EQ(resp.choices[0].finish_reason, "stop");
}

TEST(CollectingSink, MultipleChoices) {
    ChatResponse resp;
    CollectingSink sink(resp);

    sink.on_content(0, "alpha");
    sink.on_finish(0, "stop");
    sink.on_content(1, "beta");
    sink.on_finish(1, "length");

    ASSERT_EQ(resp.choices.size(), 2u);
    EXPECT_EQ(resp.choices[0].message.content, "alpha");
    EXPECT_EQ(resp.choices[0].finish_reason, "stop");
    EXPECT_EQ(resp.choices[1].message.content, "beta");
    EXPECT_EQ(resp.choices[1].finish_reason, "length");
}

TEST(CollectingSink, UsageStored) {
    ChatResponse resp;
    CollectingSink sink(resp);

    Usage u;
    u.prompt_tokens = 10;
    u.completion_tokens = 5;
    u.total_tokens = 15;
    sink.on_usage(u);

    EXPECT_EQ(resp.usage.prompt_tokens,     10);
    EXPECT_EQ(resp.usage.completion_tokens,  5);
    EXPECT_EQ(resp.usage.total_tokens,      15);
}

TEST(CollectingSink, EmptyContent) {
    ChatResponse resp;
    CollectingSink sink(resp);
    sink.on_finish(0, "stop");

    ASSERT_EQ(resp.choices.size(), 1u);
    EXPECT_EQ(resp.choices[0].message.content, "");
    EXPECT_EQ(resp.choices[0].finish_reason, "stop");
}

/* ------------------------------------------------------------------ */
/*  StreamingSink                                                      */
/* ------------------------------------------------------------------ */

struct Capture {
    std::vector<json> chunks;
    bool null_received = false;

    static int callback(void* ud, const char* chunk_json) {
        auto* c = static_cast<Capture*>(ud);
        if (!chunk_json) { c->null_received = true; return 0; }
        c->chunks.push_back(json::parse(chunk_json));
        return 0;
    }

    /* callback that returns non-zero on the Nth content chunk */
    int cancel_after_n = -1;
    int content_count  = 0;
    static int cancelling_callback(void* ud, const char* chunk_json) {
        auto* c = static_cast<Capture*>(ud);
        if (!chunk_json) { c->null_received = true; return 0; }
        auto j = json::parse(chunk_json);
        c->chunks.push_back(j);
        if (!j["choices"].empty() && j["choices"][0]["delta"].contains("content"))
            ++c->content_count;
        if (c->cancel_after_n >= 0 && c->content_count >= c->cancel_after_n)
            return 1;
        return 0;
    }
};

static StreamingSink make_sink(Capture& cap,
                                std::atomic<bool>& flag,
                                bool include_usage = false,
                                helix_stream_cb cb = Capture::callback) {
    return StreamingSink(cb, &cap,
                         "chatcmpl-test-id", 1000,
                         "test-model", "helix-test",
                         include_usage, flag);
}

TEST(StreamingSink, OpeningChunkHasRole) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_role(0);

    ASSERT_EQ(cap.chunks.size(), 1u);
    EXPECT_EQ(cap.chunks[0]["choices"][0]["delta"]["role"], "assistant");
    EXPECT_EQ(cap.chunks[0]["object"], "chat.completion.chunk");
}

TEST(StreamingSink, ContentChunk) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_content(0, "hello");

    ASSERT_EQ(cap.chunks.size(), 1u);
    EXPECT_EQ(cap.chunks[0]["choices"][0]["delta"]["content"], "hello");
    EXPECT_TRUE(cap.chunks[0]["choices"][0]["finish_reason"].is_null());
}

TEST(StreamingSink, ClosingChunkHasFinishReason) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_finish(0, "stop");

    ASSERT_EQ(cap.chunks.size(), 1u);
    EXPECT_EQ(cap.chunks[0]["choices"][0]["finish_reason"], "stop");
}

TEST(StreamingSink, NullCallbackOnStreamDone) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);
    sink.on_stream_done();
    EXPECT_TRUE(cap.null_received);
}

TEST(StreamingSink, UsageChunkEmittedWhenRequested) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag, /*include_usage=*/true);

    Usage u{10, 5, 15};
    sink.on_usage(u);

    ASSERT_EQ(cap.chunks.size(), 1u);
    EXPECT_TRUE(cap.chunks[0]["choices"].empty());
    EXPECT_EQ(cap.chunks[0]["usage"]["total_tokens"], 15);
}

TEST(StreamingSink, UsageChunkSuppressedByDefault) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag, /*include_usage=*/false);

    Usage u{10, 5, 15};
    sink.on_usage(u);

    EXPECT_TRUE(cap.chunks.empty());
}

TEST(StreamingSink, CancelViaCbSetsFlag) {
    Capture cap;
    cap.cancel_after_n = 1;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag, false, Capture::cancelling_callback);

    sink.on_content(0, "tok1");   /* content_count = 1 → callback returns 1 */
    EXPECT_TRUE(flag.load());
    EXPECT_TRUE(sink.is_cancelled());
}

TEST(StreamingSink, ClosingChunkEmittedEvenWhenCancelled) {
    Capture cap;
    cap.cancel_after_n = 1;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag, false, Capture::cancelling_callback);

    sink.on_content(0, "tok1");   /* triggers cancel */
    size_t before = cap.chunks.size();
    sink.on_finish(0, "stop");    /* must still emit closing chunk */

    EXPECT_GT(cap.chunks.size(), before);
    EXPECT_EQ(cap.chunks.back()["choices"][0]["finish_reason"], "stop");
}

/* Both sinks produce the same aggregate text for identical event streams. */
TEST(BothSinks, IdenticalAggregateOutput) {
    /* CollectingSink result */
    ChatResponse resp;
    CollectingSink csink(resp);
    csink.on_role(0);
    csink.on_content(0, "foo");
    csink.on_content(0, "bar");
    csink.on_finish(0, "stop");
    std::string collected = resp.choices[0].message.content;

    /* StreamingSink result — concatenate all content deltas */
    Capture cap;
    std::atomic<bool> flag{false};
    auto ssink = make_sink(cap, flag);
    ssink.on_role(0);
    ssink.on_content(0, "foo");
    ssink.on_content(0, "bar");
    ssink.on_finish(0, "stop");

    std::string streamed;
    for (const auto& chunk : cap.chunks) {
        if (!chunk["choices"].empty()) {
            auto& delta = chunk["choices"][0]["delta"];
            if (delta.contains("content") && delta["content"].is_string())
                streamed += delta["content"].get<std::string>();
        }
    }
    EXPECT_EQ(collected, streamed);
}

/* ------------------------------------------------------------------ */
/*  CollectingSink: tool call accumulation                             */
/* ------------------------------------------------------------------ */

TEST(CollectingSink, ToolCallStartAndEnd) {
    ChatResponse resp;
    CollectingSink sink(resp);
    sink.on_role(0);
    sink.on_content(0, "Let me check");
    sink.on_tool_call_start(0, 0, "call_abc", "get_weather");
    sink.on_tool_call_arg_delta(0, 0, "{\"city\"");
    sink.on_tool_call_arg_delta(0, 0, ":\"Paris\"}");
    sink.on_tool_call_end(0, 0);
    sink.on_finish(0, "tool_calls");

    ASSERT_EQ(resp.choices.size(), 1u);
    ASSERT_EQ(resp.choices[0].message.tool_calls.size(), 1u);
    EXPECT_EQ(resp.choices[0].message.tool_calls[0].id, "call_abc");
    EXPECT_EQ(resp.choices[0].message.tool_calls[0].function_name, "get_weather");
    EXPECT_EQ(resp.choices[0].message.tool_calls[0].function_arguments, "{\"city\":\"Paris\"}");
    EXPECT_EQ(resp.choices[0].finish_reason, "tool_calls");
}

TEST(CollectingSink, MultipleToolCalls) {
    ChatResponse resp;
    CollectingSink sink(resp);
    sink.on_tool_call_start(0, 0, "id1", "fn_a");
    sink.on_tool_call_arg_delta(0, 0, "1");
    sink.on_tool_call_end(0, 0);
    sink.on_tool_call_start(0, 1, "id2", "fn_b");
    sink.on_tool_call_arg_delta(0, 1, "2");
    sink.on_tool_call_end(0, 1);
    sink.on_finish(0, "tool_calls");

    ASSERT_EQ(resp.choices[0].message.tool_calls.size(), 2u);
    EXPECT_EQ(resp.choices[0].message.tool_calls[0].function_name, "fn_a");
    EXPECT_EQ(resp.choices[0].message.tool_calls[1].function_name, "fn_b");
}

TEST(CollectingSink, PendingToolCallDroppedOnNoEnd) {
    ChatResponse resp;
    CollectingSink sink(resp);
    sink.on_tool_call_start(0, 0, "id1", "fn");
    sink.on_tool_call_arg_delta(0, 0, "data");
    /* No on_tool_call_end — simulate cancel/early finish */
    sink.on_finish(0, "stop");

    EXPECT_TRUE(resp.choices[0].message.tool_calls.empty());
}

/* ------------------------------------------------------------------ */
/*  CollectingSink: reasoning content                                  */
/* ------------------------------------------------------------------ */

TEST(CollectingSink, ReasoningDelta) {
    ChatResponse resp;
    CollectingSink sink(resp);
    sink.on_reasoning_delta(0, "thinking...");
    sink.on_reasoning_delta(0, " more");
    sink.on_content(0, "answer");
    sink.on_finish(0, "stop");

    ASSERT_TRUE(resp.choices[0].message.reasoning_content.has_value());
    EXPECT_EQ(*resp.choices[0].message.reasoning_content, "thinking... more");
    EXPECT_EQ(resp.choices[0].message.content, "answer");
}

TEST(CollectingSink, NoReasoningWhenNeverCalled) {
    ChatResponse resp;
    CollectingSink sink(resp);
    sink.on_content(0, "answer");
    sink.on_finish(0, "stop");

    EXPECT_FALSE(resp.choices[0].message.reasoning_content.has_value());
}

/* ------------------------------------------------------------------ */
/*  StreamingSink: tool call events                                    */
/* ------------------------------------------------------------------ */

TEST(StreamingSink, ToolCallStartChunk) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_tool_call_start(0, 0, "call_123", "search");

    ASSERT_EQ(cap.chunks.size(), 1u);
    auto& tc = cap.chunks[0]["choices"][0]["delta"]["tool_calls"][0];
    EXPECT_EQ(tc["id"], "call_123");
    EXPECT_EQ(tc["function"]["name"], "search");
    EXPECT_EQ(tc["function"]["arguments"], "");
    EXPECT_EQ(tc["index"], 0);
}

TEST(StreamingSink, ToolCallArgDeltaChunk) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_tool_call_arg_delta(0, 0, "{\"q\":\"hello\"}");

    ASSERT_EQ(cap.chunks.size(), 1u);
    auto& tc = cap.chunks[0]["choices"][0]["delta"]["tool_calls"][0];
    EXPECT_EQ(tc["function"]["arguments"], "{\"q\":\"hello\"}");
}

TEST(StreamingSink, ToolCallArgDeltaEmptyIgnored) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_tool_call_arg_delta(0, 0, "");
    EXPECT_TRUE(cap.chunks.empty());
}

TEST(StreamingSink, ReasoningDeltaChunk) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_reasoning_delta(0, "hmm");

    ASSERT_EQ(cap.chunks.size(), 1u);
    EXPECT_EQ(cap.chunks[0]["choices"][0]["delta"]["reasoning_content"], "hmm");
}

TEST(StreamingSink, ReasoningDeltaEmptyIgnored) {
    Capture cap;
    std::atomic<bool> flag{false};
    auto sink = make_sink(cap, flag);

    sink.on_reasoning_delta(0, "");
    EXPECT_TRUE(cap.chunks.empty());
}
