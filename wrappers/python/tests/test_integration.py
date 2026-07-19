"""Integration tests: require HELIX_LIB_PATH and HELIX_TEST_MODEL env vars."""

import os
import pytest

MODEL_PATH = os.environ.get("HELIX_TEST_MODEL", "")
LIB_PATH   = os.environ.get("HELIX_LIB_PATH", "")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not LIB_PATH,
    reason="HELIX_TEST_MODEL and HELIX_LIB_PATH required for integration tests",
)


@pytest.fixture(scope="module")
def helix_session():
    from helix import Helix, ModelOptions, RuntimeOptions, LogLevel
    h = Helix(RuntimeOptions(log_level=LogLevel.ERROR))
    model = h.load_model(MODEL_PATH, ModelOptions(alias="qwen-test"))
    session = model.session()
    yield session
    session.close()
    model.close()
    h.close()


def test_hello_world(helix_session):
    resp = helix_session.chat_completions(
        model="qwen-test",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0,
        max_tokens=64,
        enable_thinking=False,
    )
    assert resp.choices[0].message.content is not None
    content = (resp.choices[0].message.content or "").lower()
    reasoning = (resp.choices[0].message.reasoning_content or "").lower()
    assert "pong" in content or "pong" in reasoning
    assert resp.choices[0].finish_reason in ("stop", "length")
    assert resp.usage.completion_tokens >= 1


def test_streaming(helix_session):
    chunks = list(helix_session.stream_chat_completions(
        model="qwen-test",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0,
        max_tokens=64,
        enable_thinking=False,
    ))
    assert len(chunks) >= 1
    content = "".join(c.choices[0].delta.content or "" for c in chunks)
    reasoning = "".join(c.choices[0].delta.reasoning_content or "" for c in chunks)
    assert "pong" in content.lower() or "pong" in reasoning.lower()
    finish = chunks[-1].choices[0].finish_reason
    assert finish in ("stop", "length")


def test_streaming_cancel(helix_session):
    received = 0
    for chunk in helix_session.stream_chat_completions(
        model="qwen-test",
        messages=[{"role": "user", "content": "Count from 1 to 1000."}],
        temperature=0,
        max_tokens=512,
    ):
        received += 1
        if received >= 3:
            break  # consumer break triggers cancel via finally
    assert received >= 1  # must have received at least one chunk


def test_tool_call_invocation(helix_session):
    resp = helix_session.chat_completions(
        model="qwen-test",
        messages=[{"role": "user", "content": "What is the weather in London?"}],
        temperature=0,
        max_tokens=256,
        enable_thinking=False,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }],
    )
    # Small models may not reliably tool-call; accept stop or tool_calls
    assert resp.choices[0].finish_reason in ("tool_calls", "stop")
    if resp.choices[0].finish_reason == "tool_calls":
        tc = resp.choices[0].message.tool_calls
        assert tc is not None and len(tc) >= 1
        assert tc[0].function.name == "get_weather"


def test_model_not_found_error(helix_session):
    from helix import HelixModelNotFoundError, HelixValidationError, HelixInvalidArgError
    with pytest.raises((HelixModelNotFoundError, HelixValidationError, HelixInvalidArgError)):
        helix_session.chat_completions(
            model="nonexistent-model-xyz",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
        )


def test_structured_output_json_object(helix_session):
    import json
    resp = helix_session.chat_completions(
        model="qwen-test",
        messages=[{"role": "user", "content": "Return JSON with key 'result' set to 42."}],
        temperature=0,
        max_tokens=32,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    assert content is not None
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


def test_openai_compat_shim():
    from helix.openai_compat import OpenAICompatClient
    with OpenAICompatClient(model_path=MODEL_PATH, alias="qwen-test") as client:
        resp = client.chat.completions.create(
            model="qwen-test",
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            temperature=0,
            max_tokens=64,
            enable_thinking=False,
        )
        assert resp.choices[0].message.content is not None
        content = (resp.choices[0].message.content or "").lower()
        reasoning = (resp.choices[0].message.reasoning_content or "").lower()
        assert "pong" in content or "pong" in reasoning


def test_openai_compat_streaming():
    from helix.openai_compat import OpenAICompatClient
    with OpenAICompatClient(model_path=MODEL_PATH, alias="qwen-test") as client:
        chunks = list(client.chat.completions.create(
            model="qwen-test",
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            temperature=0,
            max_tokens=64,
            enable_thinking=False,
            stream=True,
        ))
    content = "".join(c.choices[0].delta.content or "" for c in chunks)
    reasoning = "".join(c.choices[0].delta.reasoning_content or "" for c in chunks)
    assert "pong" in content.lower() or "pong" in reasoning.lower()
