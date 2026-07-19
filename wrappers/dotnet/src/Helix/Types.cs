using System.Text.Json.Serialization;

namespace Helix;

// ── Request types ─────────────────────────────────────────────────────────────

public record Message
{
    [JsonPropertyName("role")]    public required string Role    { get; init; }
    [JsonPropertyName("content")] public string? Content { get; init; }
    [JsonPropertyName("tool_calls")] public List<ToolCall>? ToolCalls { get; init; }
    [JsonPropertyName("tool_call_id")] public string? ToolCallId { get; init; }
    [JsonPropertyName("reasoning_content")] public string? ReasoningContent { get; init; }

    public static Message User(string content)      => new() { Role = "user",      Content = content };
    public static Message System(string content)    => new() { Role = "system",    Content = content };
    public static Message Assistant(string content) => new() { Role = "assistant", Content = content };
}

public record Tool
{
    [JsonPropertyName("type")]     public required string Type     { get; init; } = "function";
    [JsonPropertyName("function")] public required FunctionDef Function { get; init; }
}

public record FunctionDef
{
    [JsonPropertyName("name")]        public required string Name        { get; init; }
    [JsonPropertyName("description")] public required string Description { get; init; }
    [JsonPropertyName("parameters")]  public required object Parameters  { get; init; }
}

public record ResponseFormat
{
    [JsonPropertyName("type")]        public required string Type       { get; init; }
    [JsonPropertyName("json_schema")] public object? JsonSchema { get; init; }

    public static ResponseFormat JsonObject => new() { Type = "json_object" };
    public static ResponseFormat Text       => new() { Type = "text" };
}

public record StreamOptions
{
    [JsonPropertyName("include_usage")] public bool? IncludeUsage { get; init; }
}

public record ChatRequest
{
    [JsonPropertyName("model")]           public required string Model    { get; init; }
    [JsonPropertyName("messages")]        public required List<Message> Messages { get; init; }
    [JsonPropertyName("temperature")]     public double? Temperature  { get; init; }
    [JsonPropertyName("top_p")]           public double? TopP         { get; init; }
    [JsonPropertyName("max_tokens")]      public int? MaxTokens       { get; init; }
    [JsonPropertyName("stop")]            public List<string>? Stop   { get; init; }
    [JsonPropertyName("seed")]            public long? Seed           { get; init; }
    [JsonPropertyName("n")]              public int? N               { get; init; }
    [JsonPropertyName("tools")]           public List<Tool>? Tools    { get; init; }
    [JsonPropertyName("tool_choice")]     public object? ToolChoice  { get; init; }
    [JsonPropertyName("response_format")] public ResponseFormat? ResponseFormat { get; init; }
    [JsonPropertyName("logprobs")]        public bool? Logprobs       { get; init; }
    [JsonPropertyName("top_logprobs")]    public int? TopLogprobs     { get; init; }
    [JsonPropertyName("stream_options")]  public StreamOptions? StreamOptions { get; init; }
}

// ── Response types ─────────────────────────────────────────────────────────────

public record ToolCallFunction
{
    [JsonPropertyName("name")]      public required string Name      { get; init; }
    [JsonPropertyName("arguments")] public required string Arguments { get; init; }
}

public record ToolCall
{
    [JsonPropertyName("id")]       public required string Id       { get; init; }
    [JsonPropertyName("type")]     public required string Type     { get; init; }
    [JsonPropertyName("function")] public required ToolCallFunction Function { get; init; }
}

public record CompletionTokensDetails
{
    [JsonPropertyName("reasoning_tokens")] public int? ReasoningTokens { get; init; }
}

public record Usage
{
    [JsonPropertyName("prompt_tokens")]     public int PromptTokens     { get; init; }
    [JsonPropertyName("completion_tokens")] public int CompletionTokens { get; init; }
    [JsonPropertyName("total_tokens")]      public int TotalTokens      { get; init; }
    [JsonPropertyName("completion_tokens_details")] public CompletionTokensDetails? Details { get; init; }
}

public record LogprobInfo
{
    [JsonPropertyName("token")]  public required string Token  { get; init; }
    [JsonPropertyName("logprob")] public double Logprob { get; init; }
    [JsonPropertyName("bytes")]  public List<byte>? Bytes { get; init; }
}

public record LogprobContent : LogprobInfo
{
    [JsonPropertyName("top_logprobs")] public required List<LogprobInfo> TopLogprobs { get; init; }
}

public record Logprobs
{
    [JsonPropertyName("content")] public required List<LogprobContent> Content { get; init; }
}

public record ResponseMessage
{
    [JsonPropertyName("role")]              public required string Role             { get; init; }
    [JsonPropertyName("content")]          public string? Content          { get; init; }
    [JsonPropertyName("tool_calls")]       public List<ToolCall>? ToolCalls { get; init; }
    [JsonPropertyName("reasoning_content")] public string? ReasoningContent { get; init; }
}

public record Choice
{
    [JsonPropertyName("index")]         public int Index              { get; init; }
    [JsonPropertyName("message")]       public required ResponseMessage Message { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason  { get; init; }
    [JsonPropertyName("logprobs")]      public Logprobs? Logprobs     { get; init; }
}

public record ChatCompletion
{
    [JsonPropertyName("id")]      public required string Id      { get; init; }
    [JsonPropertyName("object")]  public required string Object  { get; init; }
    [JsonPropertyName("created")] public long Created            { get; init; }
    [JsonPropertyName("model")]   public required string Model   { get; init; }
    [JsonPropertyName("choices")] public required List<Choice> Choices { get; init; }
    [JsonPropertyName("usage")]   public required Usage Usage   { get; init; }
}

// ── Streaming chunk types ──────────────────────────────────────────────────────

public record Delta
{
    [JsonPropertyName("role")]              public string? Role             { get; init; }
    [JsonPropertyName("content")]          public string? Content          { get; init; }
    [JsonPropertyName("tool_calls")]       public List<ToolCall>? ToolCalls { get; init; }
    [JsonPropertyName("reasoning_content")] public string? ReasoningContent { get; init; }
}

public record ChunkChoice
{
    [JsonPropertyName("index")]         public int Index              { get; init; }
    [JsonPropertyName("delta")]         public required Delta Delta   { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason  { get; init; }
}

public record ChatCompletionChunk
{
    [JsonPropertyName("id")]      public required string Id      { get; init; }
    [JsonPropertyName("object")]  public required string Object  { get; init; }
    [JsonPropertyName("created")] public long Created            { get; init; }
    [JsonPropertyName("model")]   public required string Model   { get; init; }
    [JsonPropertyName("choices")] public required List<ChunkChoice> Choices { get; init; }
    [JsonPropertyName("usage")]   public Usage? Usage            { get; init; }
}

// ── JSON context for source-gen serialisation ─────────────────────────────────

[JsonSerializable(typeof(ChatRequest))]
[JsonSerializable(typeof(ChatCompletion))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(Usage))]
[JsonSerializable(typeof(Message))]
[JsonSerializable(typeof(ToolCall))]
[JsonSerializable(typeof(ResponseFormat))]
[JsonSourceGenerationOptions(
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal partial class HelixJsonContext : JsonSerializerContext { }
