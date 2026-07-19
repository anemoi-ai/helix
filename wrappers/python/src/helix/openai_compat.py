"""
OpenAI SDK compatibility shim for helix-llm.

Usage:
    from helix.openai_compat import OpenAICompatClient

    client = OpenAICompatClient(model_path="/path/to/model.gguf")
    resp = client.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.choices[0].message.content)

Supported parameters (chat.completions.create):
    messages, model, temperature, top_p, max_tokens, stop, seed, n,
    stream, stream_options, tools, tool_choice, response_format,
    logit_bias, logprobs, top_logprobs

Unsupported OpenAI-specific parameters raise HelixUnsupportedFeatureError
with a clear message rather than being silently ignored.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, Iterator, Optional

from helix._api import (
    ChatCompletion,
    ChatCompletionChunk,
    Helix,
    HelixUnsupportedFeatureError,
    Model,
    ModelOptions,
    RuntimeOptions,
    Session,
    SessionOptions,
)

# OpenAI parameters we don't support — raise instead of silently ignore
_UNSUPPORTED_PARAMS = frozenset({
    "audio", "modalities", "prediction", "parallel_tool_calls",
    "service_tier", "store", "metadata", "user", "function_call",
    "functions", "presence_penalty", "frequency_penalty",
    "best_of", "echo", "suffix",
})

_SUPPORTED_PARAMS = frozenset({
    "messages", "model", "temperature", "top_p", "max_tokens",
    "stop", "seed", "n", "stream", "stream_options", "tools",
    "tool_choice", "response_format", "logit_bias", "logprobs",
    "top_logprobs", "max_completion_tokens",
})


def _build_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    for key in kwargs:
        if key in _UNSUPPORTED_PARAMS:
            raise HelixUnsupportedFeatureError(
                f"Helix does not support the '{key}' parameter in this version. "
                "See helix-llm docs for the full supported surface.",
                param=key,
            )
    _ALL_KNOWN = _SUPPORTED_PARAMS | _UNSUPPORTED_PARAMS
    unknown = set(kwargs) - _ALL_KNOWN
    if unknown:
        warnings.warn(
            f"Unknown parameters ignored: {', '.join(sorted(unknown))}",
            UserWarning,
            stacklevel=3,
        )
    req: dict[str, Any] = {}
    for key, val in kwargs.items():
        if key in _SUPPORTED_PARAMS and val is not None:
            dest = "max_tokens" if key == "max_completion_tokens" else key
            req[dest] = val
    return req


class _Completions:
    def __init__(self, client: "OpenAICompatClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        stream = kwargs.pop("stream", False)
        req = _build_request(kwargs)
        session = self._client._get_session()
        if stream:
            return session.stream_chat_completions(**req)
        return session.chat_completions(**req)


class _Chat:
    def __init__(self, client: "OpenAICompatClient") -> None:
        self.completions = _Completions(client)


class OpenAICompatClient:
    """
    Drop-in replacement for openai.OpenAI() scoped to chat completions.

    client = OpenAICompatClient(model_path="qwen.gguf")
    # Existing code using openai.OpenAI() works unchanged (except construction).
    resp = client.chat.completions.create(model="...", messages=[...])
    """

    def __init__(
        self,
        model_path: str,
        *,
        alias: str = "",
        model_options: Optional[ModelOptions] = None,
        runtime_options: Optional[RuntimeOptions] = None,
    ) -> None:
        self._runtime = Helix(runtime_options)
        opts = model_options or ModelOptions()
        if alias:
            opts.alias = alias
        self._model: Model = self._runtime.load_model(model_path, opts)
        self._session: Optional[Session] = None
        self._session_lock = threading.Lock()
        self.chat = _Chat(self)

    def _get_session(self) -> Session:
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    self._session = self._model.session(SessionOptions())
        return self._session

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
        self._model.close()
        self._runtime.close()

    def __enter__(self) -> "OpenAICompatClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
