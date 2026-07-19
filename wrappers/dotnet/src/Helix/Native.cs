using System.Runtime.InteropServices;

namespace Helix;

/// <summary>Low-level P/Invoke declarations for libhelix.</summary>
internal static partial class Native
{
    private const string LibName = "helix";

    // Streaming callback delegate: returns non-zero to cancel.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate int StreamCallback(IntPtr userData, IntPtr chunkJson);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int helix_runtime_create(string? optionsJson, out IntPtr outRuntime);

    [LibraryImport(LibName)]
    internal static partial void helix_runtime_destroy(IntPtr runtime);

    [LibraryImport(LibName)]
    internal static partial IntPtr helix_runtime_describe(IntPtr runtime);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int helix_model_load(IntPtr runtime, string modelJson, out IntPtr outModel);

    [LibraryImport(LibName)]
    internal static partial void helix_model_release(IntPtr model);

    [LibraryImport(LibName)]
    internal static partial IntPtr helix_model_describe(IntPtr model);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int helix_session_create(IntPtr model, string? sessionJson, out IntPtr outSession);

    [LibraryImport(LibName)]
    internal static partial void helix_session_destroy(IntPtr session);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int helix_chat_completions(IntPtr session, string requestJson, out IntPtr outResponse);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int helix_chat_completions_stream(IntPtr session, string requestJson,
        [MarshalAs(UnmanagedType.FunctionPtr)] StreamCallback callback, IntPtr userData);

    [LibraryImport(LibName)]
    internal static partial void helix_session_cancel(IntPtr session);

    [LibraryImport(LibName)]
    internal static partial void helix_free(IntPtr ptr);

    [LibraryImport(LibName)]
    internal static partial IntPtr helix_last_error_json();

    [LibraryImport(LibName)]
    internal static partial uint helix_abi_version();

    [LibraryImport(LibName)]
    internal static partial IntPtr helix_version_string();

    internal static string LastErrorJson()
        => Marshal.PtrToStringUTF8(helix_last_error_json()) ?? "{}";

    internal static string? PtrToString(IntPtr ptr)
        => Marshal.PtrToStringUTF8(ptr);
}
