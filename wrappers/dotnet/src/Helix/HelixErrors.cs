using System.Text.Json;

namespace Helix;

/// <summary>Base exception for all libhelix errors.</summary>
public class HelixException : Exception
{
    public int HelixStatus { get; }
    public string? Param { get; }
    public string? Code { get; }

    internal HelixException(string message, int status, string? param = null, string? code = null)
        : base(message)
    {
        HelixStatus = status;
        Param = param;
        Code = code;
    }

    public static HelixException FromStatus(int rc)
    {
        var (msg, param, code) = ParseLastError();
        return rc switch
        {
            -1  => new HelixInvalidArgException(msg, param),
            -2  => new HelixInvalidJsonException(msg),
            -3  => new HelixValidationException(msg, param),
            -4  => new HelixModelNotFoundException(msg),
            -5  => new HelixModelLoadFailedException(msg),
            -6  => new HelixOomException(msg),
            -7  => new HelixVramExhaustedException(msg),
            -8  => new HelixContextFullException(msg),
            -9  => new HelixCancelledException(),
            -10 => new HelixBackendException(msg),
            -11 => new HelixUnsupportedFeatureException(msg),
            -99 => new HelixInternalException(msg),
            _   => new HelixException($"Helix error {rc}: {msg}", rc, param, code),
        };
    }

    internal static void Check(int rc)
    {
        if (rc == 0) return;
        throw FromStatus(rc);
    }

    private static (string msg, string? param, string? code) ParseLastError()
    {
        try
        {
            var json = Native.LastErrorJson();
            using var doc = JsonDocument.Parse(json);
            var error = doc.RootElement.GetProperty("error");
            var msg   = error.TryGetProperty("message", out var m) ? m.GetString() ?? "" : "";
            var param = error.TryGetProperty("param",   out var p) && p.ValueKind != JsonValueKind.Null ? p.GetString() : null;
            var code  = error.TryGetProperty("code",    out var c) && c.ValueKind != JsonValueKind.Null ? c.GetString() : null;
            return (msg, param, code);
        }
        catch { return ("", null, null); }
    }
}

public class HelixInvalidArgException       : HelixException { public HelixInvalidArgException(string m, string? p = null)       : base(m, -1,  p) {} }
public class HelixInvalidJsonException      : HelixException { public HelixInvalidJsonException(string m)                         : base(m, -2) {} }
public class HelixValidationException       : HelixException { public HelixValidationException(string m, string? p = null)        : base(m, -3,  p) {} }
public class HelixModelNotFoundException    : HelixException { public HelixModelNotFoundException(string m)                       : base(m, -4) {} }
public class HelixModelLoadFailedException  : HelixException { public HelixModelLoadFailedException(string m)                     : base(m, -5) {} }
public class HelixOomException              : HelixException { public HelixOomException(string m)                                 : base(m, -6) {} }
public class HelixVramExhaustedException    : HelixException { public HelixVramExhaustedException(string m)                       : base(m, -7) {} }
public class HelixContextFullException      : HelixException { public HelixContextFullException(string m)                         : base(m, -8) {} }
public class HelixCancelledException        : HelixException { public HelixCancelledException()                                    : base("Cancelled", -9) {} }
public class HelixBackendException          : HelixException { public HelixBackendException(string m)                             : base(m, -10) {} }
public class HelixUnsupportedFeatureException : HelixException { public HelixUnsupportedFeatureException(string m)                : base(m, -11) {} }
public class HelixInternalException         : HelixException { public HelixInternalException(string m)                            : base(m, -99) {} }
