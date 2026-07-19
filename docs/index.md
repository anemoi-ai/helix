# Helix

**OpenAI-isomorphic on-device LLM library — v1.6.0**

Helix is a single C/C++ shared library that turns any GGUF model into a fully
in-process, OpenAI Chat Completions-compatible LLM endpoint. No subprocess.
No open ports. No network calls.

```c
#include "helix.h"

helix_runtime_t *rt;
helix_runtime_create(NULL, &rt);

helix_model_t *model;
helix_model_load(rt,
    "{\"model_path\":\"/models/llama-3.1-8b-q4.gguf\",\"alias\":\"llama\"}",
    &model);

helix_session_t *sess;
helix_session_create(model, NULL, &sess);

char *out;
helix_chat_completions(sess,
    "{\"model\":\"llama\","
    "\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}",
    &out);
// out → standard OpenAI chat.completion JSON
helix_free(out);
```

## Why Helix?

| Feature | Description |
|---------|-------------|
| **Single header** | The entire public API is 29 C functions in `helix.h` |
| **Auto-tuned** | Probes your hardware and sets optimal parameters automatically |
| **All backends** | CPU, CUDA, Metal, Vulkan, ROCm — same API, same code |
| **OpenAI-compatible** | Drop the same JSON you'd send to `api.openai.com` |
| **ABI-stable** | Frozen for 18 months — link once, run on future 1.x releases |
| **6 wrappers** | Python, Rust, Swift, Node, Go, .NET |

## Quick install

=== "Python"
    ```sh
    pip install helix-llm
    ```

=== "Rust"
    ```sh
    cargo add helix
    ```

=== "Node"
    ```sh
    npm install @helix/node
    ```

=== "Homebrew"
    ```sh
    brew install anemoi-ai/tap/helix
    ```

=== "Chocolatey"
    ```powershell
    choco install helix
    ```

=== "Debian/Ubuntu"
    ```sh
    # Download from GitHub Releases
    apt install ./libhelix1_1.0.0-1_amd64.deb ./libhelix-dev_1.0.0-1_amd64.deb
    ```

## Next steps

- [Installation guide](install.md) — all platforms, all package managers
- [User Guide](USER_GUIDE.md) — configuration and integration guide
- [Benchmarks](benchmarks.md) — reference matrix across 8 machines
- [API Reference](api-reference.md) — full C API documentation
- [ABI Policy](abi-policy.md) — the 18-month stability commitment
