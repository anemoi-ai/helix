import Foundation

// ── Request types ─────────────────────────────────────────────────────────────

public struct Message: Codable, Sendable {
    public var role: String
    public var content: MessageContent
    public var toolCalls: [ToolCall]?
    public var toolCallId: String?
    public var reasoningContent: String?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
        case reasoningContent = "reasoning_content"
    }

    public static func user(_ text: String) -> Message {
        Message(role: "user", content: .text(text))
    }
    public static func system(_ text: String) -> Message {
        Message(role: "system", content: .text(text))
    }
    public static func assistant(_ text: String) -> Message {
        Message(role: "assistant", content: .text(text))
    }
    public static func tool(callId: String, content: String) -> Message {
        var m = Message(role: "tool", content: .text(content))
        m.toolCallId = callId
        return m
    }
}

public enum MessageContent: Codable, Sendable {
    case text(String)
    case parts([ContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else {
            self = .parts(try container.decode([ContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .parts(let p): try container.encode(p)
        }
    }
}

public struct ContentPart: Codable, Sendable {
    public var type: String
    public var text: String?
    public var imageUrl: ImageUrl?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageUrl = "image_url"
    }

    public static func text(_ t: String) -> ContentPart {
        ContentPart(type: "text", text: t, imageUrl: nil)
    }
    public static func imageUrl(_ url: String, detail: String? = nil) -> ContentPart {
        ContentPart(type: "image_url", text: nil,
                    imageUrl: ImageUrl(url: url, detail: detail))
    }
}

public struct ImageUrl: Codable, Sendable {
    public var url: String
    public var detail: String?
}

public struct Tool: Codable, Sendable {
    public var type: String
    public var function: FunctionDef

    public init(name: String, description: String, parameters: [String: Any]) {
        self.type = "function"
        self.function = FunctionDef(
            name: name,
            description: description,
            parameters: parameters
        )
    }
}

public struct FunctionDef: Codable, Sendable {
    public var name: String
    public var description: String
    public var parameters: JSONValue

    public init(name: String, description: String, parameters: [String: Any]) {
        self.name = name
        self.description = description
        self.parameters = JSONValue(parameters)
    }
}

public struct ResponseFormat: Codable, Sendable {
    public var type: String
    public var jsonSchema: JSONValue?

    enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }

    public static var jsonObject: ResponseFormat { ResponseFormat(type: "json_object") }
    public static var text: ResponseFormat { ResponseFormat(type: "text") }
}

public struct StreamOptions: Codable, Sendable {
    public var includeUsage: Bool?
    enum CodingKeys: String, CodingKey { case includeUsage = "include_usage" }
    public init(includeUsage: Bool? = nil) { self.includeUsage = includeUsage }
}

public struct ChatRequest: Codable, Sendable {
    public var model: String
    public var messages: [Message]
    public var temperature: Double?
    public var topP: Double?
    public var maxTokens: Int?
    public var stop: [String]?
    public var seed: Int?
    public var n: Int?
    public var tools: [Tool]?
    public var toolChoice: JSONValue?
    public var responseFormat: ResponseFormat?
    public var logprobs: Bool?
    public var topLogprobs: Int?
    public var streamOptions: StreamOptions?

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stop, seed, n, tools, logprobs
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case toolChoice = "tool_choice"
        case responseFormat = "response_format"
        case topLogprobs = "top_logprobs"
        case streamOptions = "stream_options"
    }

    public init(model: String, messages: [Message]) {
        self.model = model
        self.messages = messages
    }
}

// ── Response types ─────────────────────────────────────────────────────────────

public struct ChatCompletion: Codable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: Usage
}

public struct Choice: Codable, Sendable {
    public var index: Int
    public var message: ResponseMessage
    public var finishReason: String?
    public var logprobs: Logprobs?

    enum CodingKeys: String, CodingKey {
        case index, message, logprobs
        case finishReason = "finish_reason"
    }
}

public struct ResponseMessage: Codable, Sendable {
    public var role: String
    public var content: String?
    public var toolCalls: [ToolCall]?
    public var reasoningContent: String?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case reasoningContent = "reasoning_content"
    }
}

public struct ToolCall: Codable, Sendable {
    public var id: String
    public var type: String
    public var function: ToolCallFunction
}

public struct ToolCallFunction: Codable, Sendable {
    public var name: String
    public var arguments: String
}

public struct Usage: Codable, Sendable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int
    public var completionTokensDetails: CompletionTokensDetails?

    enum CodingKeys: String, CodingKey {
        case completionTokensDetails = "completion_tokens_details"
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

public struct CompletionTokensDetails: Codable, Sendable {
    public var reasoningTokens: Int?
    enum CodingKeys: String, CodingKey { case reasoningTokens = "reasoning_tokens" }
}

public struct Logprobs: Codable, Sendable {
    public var content: [LogprobContent]
}

public struct LogprobContent: Codable, Sendable {
    public var token: String
    public var logprob: Double
    public var bytes: [UInt8]?
    public var topLogprobs: [LogprobTokenInfo]
    enum CodingKeys: String, CodingKey {
        case token, logprob, bytes
        case topLogprobs = "top_logprobs"
    }
}

public struct LogprobTokenInfo: Codable, Sendable {
    public var token: String
    public var logprob: Double
    public var bytes: [UInt8]?
}

// ── Streaming chunk types ──────────────────────────────────────────────────────

public struct ChatCompletionChunk: Codable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [ChunkChoice]
    public var usage: Usage?
}

public struct ChunkChoice: Codable, Sendable {
    public var index: Int
    public var delta: Delta
    public var finishReason: String?
    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

public struct Delta: Codable, Sendable {
    public var role: String?
    public var content: String?
    public var toolCalls: [ToolCallDelta]?
    public var reasoningContent: String?
    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case reasoningContent = "reasoning_content"
    }
}

public struct ToolCallDelta: Codable, Sendable {
    public var index: Int?
    public var id: String?
    public var type: String?
    public var function: ToolCallFunctionDelta?
}

public struct ToolCallFunctionDelta: Codable, Sendable {
    public var name: String?
    public var arguments: String?
}

// ── JSONValue — type-erased Codable for dynamic JSON ─────────────────────────

public struct JSONValue: Codable, @unchecked Sendable, ExpressibleByDictionaryLiteral {
    private let value: Any

    public init(_ value: Any) { self.value = value }

    public init(dictionaryLiteral elements: (String, Any)...) {
        value = Dictionary(uniqueKeysWithValues: elements)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let d = try? container.decode([String: JSONValue].self) {
            value = d.mapValues { $0.value }
        } else if let a = try? container.decode([JSONValue].self) {
            value = a.map { $0.value }
        } else if let s = try? container.decode(String.self) {
            value = s
        } else if let n = try? container.decode(Double.self) {
            value = n
        } else if let b = try? container.decode(Bool.self) {
            value = b
        } else {
            value = NSNull()
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let d as [String: Any]:
            try container.encode(d.mapValues { JSONValue($0) })
        case let a as [Any]:
            try container.encode(a.map { JSONValue($0) })
        case let s as String:
            try container.encode(s)
        case let n as Double:
            try container.encode(n)
        case let b as Bool:
            try container.encode(b)
        default:
            try container.encodeNil()
        }
    }
}
