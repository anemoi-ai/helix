using System.Text.Json;
using System.Text.Json.Nodes;

namespace Helix;

/// <summary>Options for the Helix runtime.</summary>
public record RuntimeOptions
{
    public int LogLevel { get; init; } = 2; // WARN
}

/// <summary>
/// Process-wide Helix runtime. Create exactly one per process.
/// </summary>
public sealed class HelixRuntime : IAsyncDisposable
{
    private IntPtr _ptr;

    /// <summary>Initialises the Helix runtime.</summary>
    public HelixRuntime(RuntimeOptions? options = null)
    {
        var opts = options ?? new RuntimeOptions();
        var json = $"{{\"log_level\":{opts.LogLevel}}}";
        HelixException.Check(Native.helix_runtime_create(json, out _ptr));
    }

    /// <summary>Loads a model from disk. May take seconds to minutes.</summary>
    public HelixModel LoadModel(string modelPath, ModelOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_ptr == IntPtr.Zero, this);
        var opts = options ?? new ModelOptions();
        var json = BuildModelJson(modelPath, opts);
        HelixException.Check(Native.helix_model_load(_ptr, json, out var outPtr));
        return new HelixModel(outPtr);
    }

    /// <summary>Returns the libhelix ABI version.</summary>
    public uint AbiVersion => Native.helix_abi_version();

    /// <summary>Returns the libhelix version string.</summary>
    public string VersionString
        => Native.PtrToString(Native.helix_version_string()) ?? "";

    public async ValueTask DisposeAsync()
    {
        if (_ptr != IntPtr.Zero)
        {
            Native.helix_runtime_destroy(_ptr);
            _ptr = IntPtr.Zero;
        }
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private static string BuildModelJson(string modelPath, ModelOptions opts)
    {
        var obj = new JsonObject { ["model_path"] = modelPath };
        if (opts.Alias           is not null) obj["alias"]            = opts.Alias;
        if (opts.NGpuLayers      is not null) obj["n_gpu_layers"]     = opts.NGpuLayers.Value;
        if (opts.NCtx            is not null) obj["n_ctx"]            = opts.NCtx.Value;
        if (opts.MmprojectPath   is not null) obj["mmproj_path"]      = opts.MmprojectPath;
        if (opts.ReasoningFormat is not null) obj["reasoning_format"] = opts.ReasoningFormat;
        return obj.ToJsonString();
    }
}
