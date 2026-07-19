"""Unit tests: FFI error mapping and type round-trip (no live library needed)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from helix._ffi import (
    HELIX_E_BACKEND,
    HELIX_E_CANCELLED,
    HELIX_E_CONTEXT_FULL,
    HELIX_E_INTERNAL,
    HELIX_E_INVALID_ARG,
    HELIX_E_INVALID_JSON,
    HELIX_E_MODEL_LOAD_FAILED,
    HELIX_E_MODEL_NOT_FOUND,
    HELIX_E_OOM,
    HELIX_E_UNSUPPORTED_FEATURE,
    HELIX_E_VALIDATION,
    HELIX_E_VRAM_EXHAUSTED,
    HELIX_OK,
)
from helix._api import (
    ChatCompletion,
    ChatCompletionChunk,
    HelixBackendError,
    HelixCancelledError,
    HelixContextFullError,
    HelixInternalError,
    HelixInvalidArgError,
    HelixInvalidJsonError,
    HelixModelLoadFailedError,
    HelixModelNotFoundError,
    HelixOomError,
    HelixUnsupportedFeatureError,
    HelixValidationError,
    HelixVramExhaustedError,
    Message,
    Usage,
)


# ── Error code mapping ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("rc,exc_cls", [
    (HELIX_E_INVALID_ARG,         HelixInvalidArgError),
    (HELIX_E_INVALID_JSON,        HelixInvalidJsonError),
    (HELIX_E_VALIDATION,          HelixValidationError),
    (HELIX_E_MODEL_NOT_FOUND,     HelixModelNotFoundError),
    (HELIX_E_MODEL_LOAD_FAILED,   HelixModelLoadFailedError),
    (HELIX_E_OOM,                 HelixOomError),
    (HELIX_E_VRAM_EXHAUSTED,      HelixVramExhaustedError),
    (HELIX_E_CONTEXT_FULL,        HelixContextFullError),
    (HELIX_E_CANCELLED,           HelixCancelledError),
    (HELIX_E_BACKEND,             HelixBackendError),
    (HELIX_E_UNSUPPORTED_FEATURE, HelixUnsupportedFeatureError),
    (HELIX_E_INTERNAL,            HelixInternalError),
])
def test_error_code_mapping(rc: int, exc_cls: type) -> None:
    error_json = json.dumps({
        "error": {"message": "test error", "type": "test", "param": None, "code": None}
    })
    fake_ffi = MagicMock()
    fake_lib = MagicMock()
    fake_lib.helix_last_error_json.return_value = error_json.encode()
    fake_ffi.string.return_value = error_json.encode()

    with patch("helix._ffi.get_lib", return_value=fake_lib), \
         patch("helix._ffi._ffi", fake_ffi):
        from helix._ffi import check
        # Re-import to pick up patched ffi
        import importlib
        import helix._ffi as ffi_mod
        orig_get_lib = ffi_mod.get_lib
        orig_ffi = ffi_mod._ffi
        ffi_mod.get_lib = lambda: fake_lib
        ffi_mod._ffi = fake_ffi
        try:
            with pytest.raises(exc_cls):
                ffi_mod.check(rc, "test context")
        finally:
            ffi_mod.get_lib = orig_get_lib
            ffi_mod._ffi = orig_ffi


def test_helix_ok_does_not_raise() -> None:
    from helix._ffi import check
    check(HELIX_OK)  # must not raise


# ── ChatCompletion deserialization ─────────────────────────────────────────────

_SAMPLE_COMPLETION = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "qwen-test",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "pong"},
        "finish_reason": "stop",
        "logprobs": None,
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

def test_chat_completion_from_dict_basic() -> None:
    cc = ChatCompletion.from_dict(_SAMPLE_COMPLETION)
    assert cc.id == "chatcmpl-abc123"
    assert cc.object == "chat.completion"
    assert len(cc.choices) == 1
    assert cc.choices[0].message.content == "pong"
    assert cc.choices[0].message.role == "assistant"
    assert cc.choices[0].finish_reason == "stop"
    assert cc.usage.prompt_tokens == 10
    assert cc.usage.completion_tokens == 2
    assert cc.usage.total_tokens == 12


def test_chat_completion_tool_calls() -> None:
    d = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location":"London"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }
    cc = ChatCompletion.from_dict(d)
    assert cc.choices[0].finish_reason == "tool_calls"
    tc = cc.choices[0].message.tool_calls
    assert tc is not None and len(tc) == 1
    assert tc[0].function.name == "get_weather"
    assert tc[0].id == "call_1"


def test_chat_completion_logprobs() -> None:
    d = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "yes"},
            "finish_reason": "stop",
            "logprobs": {
                "content": [{
                    "token": "yes",
                    "logprob": -0.1,
                    "bytes": [121, 101, 115],
                    "top_logprobs": [{"token": "yes", "logprob": -0.1, "bytes": [121]}],
                }]
            },
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    cc = ChatCompletion.from_dict(d)
    lp = cc.choices[0].logprobs
    assert lp is not None
    assert lp.content[0].token == "yes"
    assert len(lp.content[0].top_logprobs) == 1


def test_chat_completion_reasoning_content() -> None:
    d = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "7", "reasoning_content": "3+4=7"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
                  "completion_tokens_details": {"reasoning_tokens": 2}},
    }
    cc = ChatCompletion.from_dict(d)
    assert cc.choices[0].message.reasoning_content == "3+4=7"
    assert cc.usage.completion_tokens_details is not None
    assert cc.usage.completion_tokens_details.reasoning_tokens == 2


# ── ChatCompletionChunk deserialization ────────────────────────────────────────

_SAMPLE_CHUNK = {
    "id": "chatcmpl-xyz",
    "object": "chat.completion.chunk",
    "created": 1700000000,
    "model": "qwen-test",
    "choices": [{
        "index": 0,
        "delta": {"role": "assistant", "content": "po"},
        "finish_reason": None,
    }],
}

def test_chunk_from_dict_basic() -> None:
    chunk = ChatCompletionChunk.from_dict(_SAMPLE_CHUNK)
    assert chunk.object == "chat.completion.chunk"
    assert chunk.choices[0].delta.content == "po"
    assert chunk.choices[0].delta.role == "assistant"
    assert chunk.choices[0].finish_reason is None
    assert chunk.usage is None


def test_chunk_with_usage() -> None:
    d = {**_SAMPLE_CHUNK, "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}
    chunk = ChatCompletionChunk.from_dict(d)
    assert chunk.usage is not None
    assert chunk.usage.total_tokens == 4


# ── OpenAI compat shim — parameter validation ──────────────────────────────────

def test_openai_compat_unsupported_param_raises() -> None:
    from helix.openai_compat import _build_request, HelixUnsupportedFeatureError
    with pytest.raises(HelixUnsupportedFeatureError) as exc_info:
        _build_request({"messages": [], "audio": {"voice": "alloy"}})
    assert "audio" in str(exc_info.value)


def test_openai_compat_supported_params_pass_through() -> None:
    from helix.openai_compat import _build_request
    req = _build_request({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "qwen-test",
        "temperature": 0.7,
        "max_tokens": 100,
    })
    assert req["messages"][0]["content"] == "hi"
    assert req["temperature"] == 0.7
    assert req["max_tokens"] == 100


def test_openai_compat_max_completion_tokens_mapped() -> None:
    from helix.openai_compat import _build_request
    req = _build_request({"messages": [], "max_completion_tokens": 50})
    assert "max_tokens" in req
    assert "max_completion_tokens" not in req
