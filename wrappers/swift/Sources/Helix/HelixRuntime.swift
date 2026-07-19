import Foundation
import CHelix

// ── UncheckedSendable wrapper for OpaquePointer ───────────────────────────────
// OpaquePointer is safe to send between concurrency domains because libhelix
// uses internal locking. We assert this with @unchecked Sendable.
struct SendablePtr: @unchecked Sendable {
    let ptr: OpaquePointer
}

// ── Options ───────────────────────────────────────────────────────────────────

public struct RuntimeOptions: Codable {
    public var logLevel: Int?
    enum CodingKeys: String, CodingKey { case logLevel = "log_level" }
    public init(logLevel: Int? = nil) { self.logLevel = logLevel }
    public static var `default`: RuntimeOptions { RuntimeOptions(logLevel: 2) }
}

public struct ModelOptions: Codable {
    public var alias: String?
    public var nGpuLayers: Int?
    public var nCtx: Int?
    public var mmprojectPath: String?
    public var reasoningFormat: String?

    enum CodingKeys: String, CodingKey {
        case alias
        case nGpuLayers = "n_gpu_layers"
        case nCtx = "n_ctx"
        case mmprojectPath = "mmproj_path"
        case reasoningFormat = "reasoning_format"
    }

    public init(alias: String? = nil, nGpuLayers: Int? = nil,
                nCtx: Int? = nil, mmprojectPath: String? = nil,
                reasoningFormat: String? = nil) {
        self.alias = alias
        self.nGpuLayers = nGpuLayers
        self.nCtx = nCtx
        self.mmprojectPath = mmprojectPath
        self.reasoningFormat = reasoningFormat
    }

    public static var `default`: ModelOptions { ModelOptions() }
}

public struct SessionOptions: Codable {
    public init() {}
    public static var `default`: SessionOptions { SessionOptions() }
}

// ── Session ───────────────────────────────────────────────────────────────────

public actor Session {
    nonisolated let _sendablePtr: SendablePtr

    var ptr: OpaquePointer { _sendablePtr.ptr }

    init(ptr: OpaquePointer) {
        self._sendablePtr = SendablePtr(ptr: ptr)
    }

    deinit {
        helix_session_destroy(_sendablePtr.ptr)
    }

    public func chatCompletions(_ request: ChatRequest) throws -> ChatCompletion {
        let reqData = try JSONEncoder().encode(request)
        let reqStr = String(data: reqData, encoding: .utf8)!
        var outPtr: UnsafeMutablePointer<CChar>? = nil

        let status = reqStr.withCString { cStr in
            helix_chat_completions(ptr, cStr, &outPtr)
        }
        try check(status: status)

        guard let out = outPtr else {
            throw HelixError.internalError(message: "null response pointer")
        }
        let responseStr = String(cString: out)
        helix_free(out)

        let data = responseStr.data(using: .utf8)!
        return try JSONDecoder().decode(ChatCompletion.self, from: data)
    }

    public nonisolated func streamChatCompletions(_ request: ChatRequest) -> ChatCompletionStream {
        return ChatCompletionStream(session: self, request: request)
    }

    public func cancel() {
        helix_session_cancel(ptr)
    }
}

// ── ChatCompletionStream (AsyncSequence) ──────────────────────────────────────

// Box held by the C callback via Unmanaged — @unchecked Sendable because
// the continuation is only accessed from the single C callback thread.
private final class StreamBox: @unchecked Sendable {
    var cont: AsyncStream<Result<ChatCompletionChunk, Error>>.Continuation?
    init(_ c: AsyncStream<Result<ChatCompletionChunk, Error>>.Continuation) { cont = c }
}

// Top-level C callback: cannot capture Swift state; uses user_data.
private func helixStreamCallback(
    userData: UnsafeMutableRawPointer?,
    chunkJson: UnsafePointer<CChar>?
) -> Int32 {
    guard let userData = userData else { return 0 }
    let box = Unmanaged<StreamBox>.fromOpaque(userData).takeUnretainedValue()
    guard let chunkJson = chunkJson else { return 0 }
    let str = String(cString: chunkJson)
    guard let data = str.data(using: .utf8),
          let chunk = try? JSONDecoder().decode(ChatCompletionChunk.self, from: data)
    else { return 0 }
    box.cont?.yield(.success(chunk))
    return 0
}

public struct ChatCompletionStream: AsyncSequence, Sendable {
    public typealias Element = ChatCompletionChunk

    // Use _sendablePtr so we can pass the pointer across concurrency domains
    // without needing an actor hop inside the detached Task.
    let sessionPtr: SendablePtr
    let request: ChatRequest

    init(session: Session, request: ChatRequest) {
        self.sessionPtr = session._sendablePtr
        self.request = request
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(sessionPtr: sessionPtr, request: request)
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        let sessionPtr: SendablePtr
        let request: ChatRequest
        var inner: AsyncStream<Result<ChatCompletionChunk, Error>>.AsyncIterator?
        var taskHandle: Task<Void, Never>?

        public mutating func next() async throws -> ChatCompletionChunk? {
            if inner == nil {
                let (stream, cont) = AsyncStream<Result<ChatCompletionChunk, Error>>.makeStream()
                inner = stream.makeAsyncIterator()

                let box = StreamBox(cont)
                let reqData = try JSONEncoder().encode(request)
                guard let reqStr = String(data: reqData, encoding: .utf8) else { return nil }

                let rawPtr = sessionPtr.ptr
                taskHandle = Task.detached {
                    let unmanaged = Unmanaged.passRetained(box)
                    let userData = unmanaged.toOpaque()
                    _ = reqStr.withCString { cStr in
                        helix_chat_completions_stream(rawPtr, cStr, helixStreamCallback, userData)
                    }
                    box.cont?.finish()
                    unmanaged.release()
                }
            }

            guard let result = await inner?.next() else { return nil }
            switch result {
            case .success(let chunk): return chunk
            case .failure(let error): throw error
            }
        }

        public func cancel() {
            taskHandle?.cancel()
            helix_session_cancel(sessionPtr.ptr)
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

public final class HelixModel: @unchecked Sendable {
    let ptr: OpaquePointer

    init(ptr: OpaquePointer) {
        self.ptr = ptr
    }

    deinit {
        helix_model_release(ptr)
    }

    public func session(options: SessionOptions = .default) throws -> Session {
        let optsData = try JSONEncoder().encode(options)
        let optsStr = String(data: optsData, encoding: .utf8)!
        var outPtr: OpaquePointer? = nil
        let status = optsStr.withCString { cStr in
            helix_session_create(ptr, cStr, &outPtr)
        }
        try check(status: status)
        guard let out = outPtr else {
            throw HelixError.internalError(message: "null session pointer")
        }
        return Session(ptr: out)
    }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

public final class HelixRuntime: @unchecked Sendable {
    private let ptr: OpaquePointer

    public init(options: RuntimeOptions = .default) throws {
        let optsData = try JSONEncoder().encode(options)
        let optsStr = String(data: optsData, encoding: .utf8)!
        var outPtr: OpaquePointer? = nil
        let status = optsStr.withCString { cStr in
            helix_runtime_create(cStr, &outPtr)
        }
        try check(status: status)
        guard let out = outPtr else {
            throw HelixError.internalError(message: "null runtime pointer")
        }
        ptr = out
    }

    deinit {
        helix_runtime_destroy(ptr)
    }

    public func loadModel(path: String, options: ModelOptions = .default) throws -> HelixModel {
        let opts = options
        var modelJson: [String: Any] = ["model_path": path]
        if let alias = opts.alias { modelJson["alias"] = alias }
        if let n = opts.nGpuLayers { modelJson["n_gpu_layers"] = n }
        if let c = opts.nCtx { modelJson["n_ctx"] = c }
        if let m = opts.mmprojectPath { modelJson["mmproj_path"] = m }
        if let r = opts.reasoningFormat { modelJson["reasoning_format"] = r }
        let data = try JSONSerialization.data(withJSONObject: modelJson)
        let jsonStr = String(data: data, encoding: .utf8)!

        var outPtr: OpaquePointer? = nil
        let status = jsonStr.withCString { cStr in
            helix_model_load(ptr, cStr, &outPtr)
        }
        try check(status: status)
        guard let out = outPtr else {
            throw HelixError.internalError(message: "null model pointer")
        }
        return HelixModel(ptr: out)
    }

    public var versionString: String {
        String(cString: helix_version_string())
    }

    public var abiVersion: UInt32 {
        helix_abi_version()
    }
}
