import XCTest
@testable import Helix

final class TypeTests: XCTestCase {

    // ── Error mapping ─────────────────────────────────────────────────────────

    func testErrorFromStatusValidation() {
        let err = HelixError.from(status: -3, lastErrorJson: #"{"error":{"message":"bad param","type":"invalid_request_error","param":"messages"}}"#)
        if case .validation(let msg, let param) = err {
            XCTAssertEqual(msg, "bad param")
            XCTAssertEqual(param, "messages")
        } else {
            XCTFail("Expected .validation, got \(err)")
        }
    }

    func testErrorFromStatusCancelled() {
        let err = HelixError.from(status: -9, lastErrorJson: "{}")
        if case .cancelled = err { } else {
            XCTFail("Expected .cancelled, got \(err)")
        }
    }

    func testErrorFromStatusModelNotFound() {
        let err = HelixError.from(status: -4, lastErrorJson: #"{"error":{"message":"no such model"}}"#)
        if case .modelNotFound(let m) = err {
            XCTAssertEqual(m, "no such model")
        } else {
            XCTFail("Expected .modelNotFound, got \(err)")
        }
    }

    // ── Message constructors ──────────────────────────────────────────────────

    func testUserMessageConstructor() {
        let msg = Message.user("hello")
        XCTAssertEqual(msg.role, "user")
        if case .text(let t) = msg.content {
            XCTAssertEqual(t, "hello")
        } else {
            XCTFail("Expected text content")
        }
    }

    func testSystemMessageConstructor() {
        let msg = Message.system("be helpful")
        XCTAssertEqual(msg.role, "system")
    }

    // ── ChatCompletion round-trip ─────────────────────────────────────────────

    func testChatCompletionDecode() throws {
        let json = """
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "qwen-test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        """
        let cc = try JSONDecoder().decode(ChatCompletion.self, from: json.data(using: .utf8)!)
        XCTAssertEqual(cc.id, "chatcmpl-1")
        XCTAssertEqual(cc.choices[0].message.content, "pong")
        XCTAssertEqual(cc.choices[0].finishReason, "stop")
        XCTAssertEqual(cc.usage.totalTokens, 7)
    }

    func testChatCompletionWithToolCalls() throws {
        let json = """
        {
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\\"location\\":\\"London\\""}}]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        }
        """
        let cc = try JSONDecoder().decode(ChatCompletion.self, from: json.data(using: .utf8)!)
        XCTAssertEqual(cc.choices[0].finishReason, "tool_calls")
        XCTAssertEqual(cc.choices[0].message.toolCalls?.first?.function.name, "get_weather")
    }

    func testChunkDecode() throws {
        let json = """
        {
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "po"}, "finish_reason": null}]
        }
        """
        let chunk = try JSONDecoder().decode(ChatCompletionChunk.self, from: json.data(using: .utf8)!)
        XCTAssertEqual(chunk.choices[0].delta.content, "po")
        XCTAssertEqual(chunk.choices[0].delta.role, "assistant")
        XCTAssertNil(chunk.choices[0].finishReason)
    }

    // ── ChatRequest encoding ───────────────────────────────────────────────────

    func testChatRequestEncodes() throws {
        let req = ChatRequest(model: "qwen-test", messages: [.user("hi")])
        let data = try JSONEncoder().encode(req)
        let v = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(v["model"] as? String, "qwen-test")
        XCTAssertNotNil(v["messages"])
    }

    func testResponseFormatJsonObject() throws {
        let rf = ResponseFormat.jsonObject
        let data = try JSONEncoder().encode(rf)
        let v = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(v["type"] as? String, "json_object")
    }

    // ── Reasoning content ─────────────────────────────────────────────────────

    func testReasoningContentDecodes() throws {
        let json = """
        {
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "7", "reasoning_content": "3+4=7"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
                      "completion_tokens_details": {"reasoning_tokens": 2}}
        }
        """
        let cc = try JSONDecoder().decode(ChatCompletion.self, from: json.data(using: .utf8)!)
        XCTAssertEqual(cc.choices[0].message.reasoningContent, "3+4=7")
        XCTAssertEqual(cc.usage.completionTokensDetails?.reasoningTokens, 2)
    }
}
