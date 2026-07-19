# Helix Language Wrappers

Six idiomatic wrappers for `libhelix` — pick your language.

> **Principle**: wrappers do not invent. They expose `Runtime`, `Model`, `Session`, and `chat_completions` / `stream_chat_completions` in each language's native idiom. All inference logic lives in the C++ core.

## Install

| Language | Package | Install |
|---|---|---|
| Python | `helix-llm` on PyPI | `pip install helix-llm` |
| Rust | `helix` on crates.io | `cargo add helix` |
| Swift | `Helix` (SPM) | see Package.swift |
| Node.js | `@helix/node` on npm | `npm install @helix/node` |
| Go | `github.com/anemoi-ai/helix-go` | `go get github.com/anemoi-ai/helix-go` |
| .NET | `Helix.Net` on NuGet | `dotnet add package Helix.Net` |

---

## Hello world — all six languages

### Python

```python
from helix import Helix, ModelOptions, LogLevel, RuntimeOptions

with Helix(RuntimeOptions(log_level=LogLevel.WARN)) as h:
    model = h.load_model("qwen.gguf", ModelOptions(alias="qwen"))
    session = model.session()
    resp = session.chat_completions(
        model="qwen",
        messages=[{"role": "user", "content": "Hello, Helix!"}],
        max_tokens=64,
    )
    print(resp.choices[0].message.content)
```

**OpenAI drop-in (Python only):**
```python
from helix.openai_compat import OpenAICompatClient

client = OpenAICompatClient(model_path="qwen.gguf")
# Existing OpenAI SDK code works unchanged ↓
resp = client.chat.completions.create(
    model="qwen", messages=[{"role": "user", "content": "Hello!"}]
)
```

---

### Rust

```rust
use helix::{Runtime, RuntimeOptions, ModelOptions, SessionOptions, ChatCompletionRequest, Message};

let runtime = Runtime::new(RuntimeOptions::default())?;
let model   = runtime.load_model("qwen.gguf", ModelOptions::default())?;
let session = model.session(SessionOptions::default())?;

let resp = session.chat_completions(ChatCompletionRequest {
    model:     "qwen".into(),
    messages:  vec![Message::user("Hello, Helix!")],
    max_tokens: Some(64),
    ..Default::default()
})?;
println!("{}", resp.choices[0].message.content.as_deref().unwrap_or(""));
```

**Streaming:**
```rust
use futures::StreamExt;
let mut stream = Arc::clone(&session).stream_chat_completions(req).await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?.choices[0].delta.content.as_deref().unwrap_or(""));
}
```

---

### Swift

```swift
import Helix

let runtime = try HelixRuntime()
let model   = try runtime.loadModel(path: "qwen.gguf")
let session = try model.createSession()

let resp = try await session.chatCompletions(ChatRequest(
    model: "qwen",
    messages: [.user("Hello, Helix!")]
))
print(resp.choices.first?.message.content ?? "")
```

**Streaming:**
```swift
for try await chunk in session.streamChatCompletions(request) {
    print(chunk.choices.first?.delta.content ?? "", terminator: "")
}
```

---

### Node / TypeScript

```typescript
import { Helix } from "@helix/node";

const helix   = new Helix();
const model   = helix.loadModel("qwen.gguf");
const session = model.session();

const resp = session.chatCompletions({
    model: "qwen",
    messages: [{ role: "user", content: "Hello, Helix!" }],
    max_tokens: 64,
});
console.log(resp.choices[0].message.content);
```

**Streaming:**
```typescript
const stream = session.streamChatCompletions({ model: "qwen", messages: [...] });
for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

**OpenAI drop-in (Node only):**
```typescript
import { createOpenAICompat } from "@helix/node/openai-compat";
const client = createOpenAICompat({ modelPath: "qwen.gguf" });
const resp = await client.chat.completions.create({ model: "qwen", messages: [...] });
```

---

### Go

```go
import helix "github.com/anemoi-ai/helix-go"

h, _       := helix.New(helix.Options{})
model, _   := h.LoadModel("qwen.gguf", helix.ModelOptions{Alias: "qwen"})
session, _ := model.NewSession(helix.SessionOptions{})

resp, err := session.ChatCompletions(ctx, helix.ChatRequest{
    Model:    "qwen",
    Messages: []helix.Message{{Role: "user", Content: "Hello, Helix!"}},
})
fmt.Println(resp.Choices[0].Message.Content)
```

**Streaming:**
```go
ch, errCh := session.StreamChatCompletions(ctx, req)
for chunk := range ch {
    fmt.Print(chunk.Choices[0].Delta.Content)
}
if err := <-errCh; err != nil { log.Fatal(err) }
```

> **Note:** Go users must have `libhelix` installed system-wide or in `LD_LIBRARY_PATH`.  
> `CGO_LDFLAGS="-L/path/to/helix/build" go get github.com/anemoi-ai/helix-go`

---

### .NET / C\#

```csharp
using Helix;

await using var runtime = new HelixRuntime();
await using var model   = runtime.LoadModel("qwen.gguf");
await using var session = model.CreateSession();

var resp = session.ChatCompletions(new ChatRequest
{
    Model    = "qwen",
    Messages = [Message.User("Hello, Helix!")],
    MaxTokens = 64,
});
Console.WriteLine(resp.Choices[0].Message.Content);
```

**Streaming:**
```csharp
await foreach (var chunk in session.StreamChatCompletionsAsync(request, cancellationToken))
{
    Console.Write(chunk.Choices[0].Delta?.Content);
}
```

---

## Cancellation

| Language | Primitive |
|---|---|
| Python (sync) | `session.cancel()` / `break` in iterator (RAII) |
| Python (async) | `asyncio.CancelledError` propagation |
| Rust | `Drop` on the stream handle |
| Swift | Cancel the enclosing `Task` |
| Node | `AbortSignal` passed to `streamChatCompletions` |
| Go | `context.Context` cancellation |
| .NET | `CancellationToken` passed to `StreamChatCompletionsAsync` |

All paths converge on `helix_session_cancel` in the C ABI.

---

## Error handling

| C status | Python | Rust | Swift | Node | Go | .NET |
|---|---|---|---|---|---|---|
| `HELIX_E_VALIDATION` | `HelixValidationError` | `Error::Validation` | `HelixError.validation` | `HelixValidationError` | `ErrValidation` (wrapped) | `HelixValidationException` |
| `HELIX_E_MODEL_NOT_FOUND` | `HelixModelNotFoundError` | `Error::ModelNotFound` | `.modelNotFound` | `HelixModelNotFoundError` | `ErrModelNotFound` | `HelixModelNotFoundException` |
| `HELIX_E_OOM` | `HelixOomError` | `Error::Oom` | `.oom` | `HelixOomError` | `ErrOom` | `HelixOomException` |
| `HELIX_E_CANCELLED` | `HelixCancelledError` | `Error::Cancelled` | `.cancelled` | `HelixCancelledError` | `ErrCancelled` | `HelixCancelledException` |
| `HELIX_E_CONTEXT_FULL` | `HelixContextFullError` | `Error::ContextFull` | `.contextFull` | `HelixContextFullError` | `ErrContextFull` | `HelixContextFullException` |

---

## Versioning

All wrappers are versioned to match `libhelix`: `helix-llm 0.8.x` requires `libhelix 0.8.x`. Patch-version drift is allowed pre-1.0; minor versions align post-1.0.

## Conformance

All six wrappers are validated against `tests/wrapper-conformance/cases.yaml`. To run:

```bash
HELIX_SHIM_URL=http://localhost:8080 \
HELIX_TEST_MODEL=qwen-test \
python tests/wrapper-conformance/run.py --wrappers python,rust,go,node,dotnet,swift
```

## Further reading

- [Helix README](../../README.md) — architecture, build, C ABI reference
- [User Guide](../../USER_GUIDE.md) — chat format, tool calling, structured output, vision
