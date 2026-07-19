import Foundation
import CHelix

/// All errors thrown by the Helix Swift wrapper.
public enum HelixError: Error, LocalizedError {
    case invalidArg(message: String, param: String?)
    case invalidJson(message: String)
    case validation(message: String, param: String?)
    case modelNotFound(message: String)
    case modelLoadFailed(message: String)
    case oom(message: String)
    case vramExhausted(message: String)
    case contextFull(message: String)
    case cancelled
    case backend(message: String)
    case unsupportedFeature(message: String)
    case internalError(message: String)
    case unknown(status: Int32, message: String)

    public var errorDescription: String? {
        switch self {
        case .invalidArg(let m, _):         return "Invalid argument: \(m)"
        case .invalidJson(let m):            return "Invalid JSON: \(m)"
        case .validation(let m, _):          return "Validation error: \(m)"
        case .modelNotFound(let m):          return "Model not found: \(m)"
        case .modelLoadFailed(let m):        return "Model load failed: \(m)"
        case .oom(let m):                    return "Out of memory: \(m)"
        case .vramExhausted(let m):          return "VRAM exhausted: \(m)"
        case .contextFull(let m):            return "Context full: \(m)"
        case .cancelled:                     return "Cancelled"
        case .backend(let m):                return "Backend error: \(m)"
        case .unsupportedFeature(let m):     return "Unsupported feature: \(m)"
        case .internalError(let m):          return "Internal error: \(m)"
        case .unknown(let s, let m):         return "Helix error \(s): \(m)"
        }
    }

    static func from(status: Int32, lastErrorJson: String) -> HelixError {
        let body = parseErrorBody(lastErrorJson)
        let msg = body.message
        let param = body.param

        switch status {
        case -1:  return .invalidArg(message: msg, param: param)
        case -2:  return .invalidJson(message: msg)
        case -3:  return .validation(message: msg, param: param)
        case -4:  return .modelNotFound(message: msg)
        case -5:  return .modelLoadFailed(message: msg)
        case -6:  return .oom(message: msg)
        case -7:  return .vramExhausted(message: msg)
        case -8:  return .contextFull(message: msg)
        case -9:  return .cancelled
        case -10: return .backend(message: msg)
        case -11: return .unsupportedFeature(message: msg)
        case -99: return .internalError(message: msg)
        default:  return .unknown(status: status, message: msg)
        }
    }
}

private struct ErrorEnvelope: Decodable {
    struct Body: Decodable {
        var message: String = ""
        var param: String?
    }
    var error: Body = Body()
}

private func parseErrorBody(_ json: String) -> (message: String, param: String?) {
    guard let data = json.data(using: .utf8),
          let envelope = try? JSONDecoder().decode(ErrorEnvelope.self, from: data)
    else { return (json, nil) }
    return (envelope.error.message, envelope.error.param)
}

func check(status: Int32) throws {
    guard status != 0 else { return }
    let errJson = String(cString: helix_last_error_json())
    throw HelixError.from(status: status, lastErrorJson: errJson)
}
