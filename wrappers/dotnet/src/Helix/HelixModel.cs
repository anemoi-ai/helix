using System.Text.Json;

namespace Helix;

/// <summary>Options for model loading.</summary>
public record ModelOptions
{
    public string? Alias           { get; init; }
    public int?    NGpuLayers      { get; init; }
    public int?    NCtx            { get; init; }
    public string? MmprojectPath   { get; init; }
    public string? ReasoningFormat { get; init; }
}

/// <summary>A loaded model. Thread-safe; share across sessions.</summary>
public sealed class HelixModel : IAsyncDisposable
{
    private IntPtr _ptr;

    internal HelixModel(IntPtr ptr) => _ptr = ptr;

    /// <summary>Creates a new inference session on this model.</summary>
    public HelixSession CreateSession()
    {
        ObjectDisposedException.ThrowIf(_ptr == IntPtr.Zero, this);
        HelixException.Check(Native.helix_session_create(_ptr, "{}", out var outPtr));
        return new HelixSession(outPtr);
    }

    public async ValueTask DisposeAsync()
    {
        if (_ptr != IntPtr.Zero)
        {
            Native.helix_model_release(_ptr);
            _ptr = IntPtr.Zero;
        }
        await Task.CompletedTask.ConfigureAwait(false);
    }
}
