using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading.Channels;

namespace Helix;

/// <summary>
/// A Helix inference session. Owns a KV cache; one in-flight request at a time.
/// </summary>
public sealed class HelixSession : IAsyncDisposable
{
    private IntPtr _ptr;

    internal HelixSession(IntPtr ptr) => _ptr = ptr;

    /// <summary>Runs a synchronous chat completion.</summary>
    public ChatCompletion ChatCompletions(ChatRequest request)
    {
        ObjectDisposedException.ThrowIf(_ptr == IntPtr.Zero, this);
        var json = JsonSerializer.Serialize(request, HelixJsonContext.Default.ChatRequest);
        HelixException.Check(Native.helix_chat_completions(_ptr, json, out var outPtr));
        try
        {
            var str = Native.PtrToString(outPtr) ?? throw new HelixInternalException("null response");
            return JsonSerializer.Deserialize(str, HelixJsonContext.Default.ChatCompletion)
                   ?? throw new HelixInternalException("failed to deserialise response");
        }
        finally { Native.helix_free(outPtr); }
    }

    /// <summary>Streams chat completion chunks as an <see cref="IAsyncEnumerable{T}"/>.</summary>
    public async IAsyncEnumerable<ChatCompletionChunk> StreamChatCompletionsAsync(
        ChatRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_ptr == IntPtr.Zero, this);

        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
        });

        // Register cancellation to call helix_session_cancel.
        using var reg = cancellationToken.Register(() =>
        {
            if (_ptr != IntPtr.Zero) Native.helix_session_cancel(_ptr);
            channel.Writer.TryComplete();
        });

        var json = JsonSerializer.Serialize(request, HelixJsonContext.Default.ChatRequest);
        var sessionPtr = _ptr;

        // Run the blocking C call on a thread-pool thread.
        var streamTask = Task.Run(() =>
        {
            Native.StreamCallback cb = (_, chunkPtr) =>
            {
                if (chunkPtr == IntPtr.Zero)
                {
                    channel.Writer.TryComplete();
                    return 0;
                }
                var chunk = Marshal.PtrToStringUTF8(chunkPtr);
                if (chunk != null) channel.Writer.TryWrite(chunk);
                return cancellationToken.IsCancellationRequested ? 1 : 0;
            };

            // Keep the delegate alive for the duration of the call.
            GC.KeepAlive(cb);
            int rc = Native.helix_chat_completions_stream(sessionPtr, json, cb, IntPtr.Zero);
            if (rc != 0 && rc != -9) // -9 = cancelled — not an error
                channel.Writer.TryComplete(HelixException.FromStatus(rc));
            else
                channel.Writer.TryComplete();
        }, CancellationToken.None);

        await foreach (var raw in channel.Reader.ReadAllAsync(cancellationToken))
        {
            var chunk = JsonSerializer.Deserialize(raw, HelixJsonContext.Default.ChatCompletionChunk);
            if (chunk is not null) yield return chunk;
        }

        // Propagate any stream error.
        await streamTask.ConfigureAwait(false);
    }

    /// <summary>Cooperatively cancels any in-flight request on this session.</summary>
    public void Cancel()
    {
        if (_ptr != IntPtr.Zero) Native.helix_session_cancel(_ptr);
    }

    public async ValueTask DisposeAsync()
    {
        if (_ptr != IntPtr.Zero)
        {
            Native.helix_session_destroy(_ptr);
            _ptr = IntPtr.Zero;
        }
        await Task.CompletedTask.ConfigureAwait(false);
    }
}
