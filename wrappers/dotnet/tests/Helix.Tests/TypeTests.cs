using System.Text.Json;
using Helix;
using Xunit;

namespace Helix.Tests;

public class TypeTests
{
    // ── Error class hierarchy ─────────────────────────────────────────────────

    [Fact]
    public void ValidationException_IsHelixException()
    {
        var ex = new HelixValidationException("bad param", "messages");
        Assert.IsAssignableFrom<HelixException>(ex);
        Assert.Equal(-3, ex.HelixStatus);
        Assert.Equal("messages", ex.Param);
    }

    [Fact]
    public void CancelledException_HasCorrectStatus()
    {
        var ex = new HelixCancelledException();
        Assert.Equal(-9, ex.HelixStatus);
        Assert.IsAssignableFrom<HelixException>(ex);
    }

    [Fact]
    public void ModelNotFoundException_HasCorrectStatus()
    {
        var ex = new HelixModelNotFoundException("no such model");
        Assert.Equal(-4, ex.HelixStatus);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-2)]
    [InlineData(-3)]
    [InlineData(-4)]
    [InlineData(-5)]
    [InlineData(-6)]
    [InlineData(-7)]
    [InlineData(-8)]
    [InlineData(-9)]
    [InlineData(-10)]
    [InlineData(-11)]
    [InlineData(-99)]
    public void FromStatus_ReturnsHelixException(int status)
    {
        var ex = HelixException.FromStatus(status);
        Assert.IsAssignableFrom<HelixException>(ex);
        Assert.Equal(status, ex.HelixStatus);
    }

    // ── ChatCompletion JSON round-trip ────────────────────────────────────────

    [Fact]
    public void ChatCompletion_Deserialises()
    {
        const string json = """
        {
            "id":"chatcmpl-1","object":"chat.completion","created":0,"model":"qwen-test",
            "choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
        }
        """;
        var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var cc = JsonSerializer.Deserialize<ChatCompletion>(json, opts);
        Assert.NotNull(cc);
        Assert.Equal("chatcmpl-1", cc.Id);
        Assert.Equal("pong", cc.Choices[0].Message.Content);
        Assert.Equal("stop", cc.Choices[0].FinishReason);
        Assert.Equal(7, cc.Usage.TotalTokens);
    }

    [Fact]
    public void ChatCompletion_WithToolCalls_Deserialises()
    {
        const string json = """
        {
            "id":"x","object":"chat.completion","created":0,"model":"m",
            "choices":[{"index":0,"message":{"role":"assistant","content":null,
                "tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},
                "finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}
        }
        """;
        var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var cc = JsonSerializer.Deserialize<ChatCompletion>(json, opts);
        Assert.NotNull(cc);
        Assert.Equal("tool_calls", cc.Choices[0].FinishReason);
        Assert.Equal("get_weather", cc.Choices[0].Message.ToolCalls![0].Function.Name);
    }

    [Fact]
    public void ChatCompletionChunk_Deserialises()
    {
        const string json = """
        {
            "id":"x","object":"chat.completion.chunk","created":0,"model":"m",
            "choices":[{"index":0,"delta":{"role":"assistant","content":"po"},"finish_reason":null}]
        }
        """;
        var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var chunk = JsonSerializer.Deserialize<ChatCompletionChunk>(json, opts);
        Assert.NotNull(chunk);
        Assert.Equal("po", chunk.Choices[0].Delta.Content);
        Assert.Equal("assistant", chunk.Choices[0].Delta.Role);
        Assert.Null(chunk.Choices[0].FinishReason);
    }

    [Fact]
    public void ReasoningContent_Deserialises()
    {
        const string json = """
        {
            "id":"x","object":"chat.completion","created":0,"model":"m",
            "choices":[{"index":0,"message":{"role":"assistant","content":"7","reasoning_content":"3+4=7"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,
                     "completion_tokens_details":{"reasoning_tokens":2}}
        }
        """;
        var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var cc = JsonSerializer.Deserialize<ChatCompletion>(json, opts);
        Assert.NotNull(cc);
        Assert.Equal("3+4=7", cc.Choices[0].Message.ReasoningContent);
        Assert.Equal(2, cc.Usage.Details?.ReasoningTokens);
    }

    // ── ChatRequest serialisation ─────────────────────────────────────────────

    [Fact]
    public void ChatRequest_Serialises_OmitsNulls()
    {
        var req = new ChatRequest
        {
            Model = "qwen-test",
            Messages = [Message.User("hi")],
            Temperature = 0.7,
        };
        var serOpts = new JsonSerializerOptions {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        };
        var json = JsonSerializer.Serialize(req, serOpts);
        var doc = JsonDocument.Parse(json);
        Assert.Equal("qwen-test", doc.RootElement.GetProperty("model").GetString());
        Assert.False(doc.RootElement.TryGetProperty("tools", out _), "tools should be omitted");
        Assert.False(doc.RootElement.TryGetProperty("max_tokens", out _), "max_tokens should be omitted");
    }

    [Fact]
    public void Message_StaticConstructors_HaveCorrectRole()
    {
        Assert.Equal("user",      Message.User("x").Role);
        Assert.Equal("system",    Message.System("x").Role);
        Assert.Equal("assistant", Message.Assistant("x").Role);
    }

    [Fact]
    public void ResponseFormat_JsonObject_HasCorrectType()
    {
        var rf = ResponseFormat.JsonObject;
        Assert.Equal("json_object", rf.Type);
    }
}
