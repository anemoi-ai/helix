"""High-level Python API for libhelix."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator, Iterator, Optional

import cffi

from helix._ffi import _ffi, check, get_lib

_STREAM_TIMEOUT = 300


# ── Errors ────────────────────────────────────────────────────────────────────

class HelixError(Exception):
    def __init__(self, message: str, *, param: Optional[str] = None,
                 code: Optional[str] = None, status: int = 0) -> None:
        super().__init__(message)
        self.param = param
        self.code = code
        self.status = status

class HelixInvalidArgError(HelixError): pass
class HelixInvalidJsonError(HelixError): pass
class HelixValidationError(HelixError): pass
class HelixModelNotFoundError(HelixError): pass
class HelixModelLoadFailedError(HelixError): pass
class HelixOomError(HelixError): pass
class HelixVramExhaustedError(HelixError): pass
class HelixContextFullError(HelixError): pass
class HelixCancelledError(HelixError): pass
class HelixBackendError(HelixError): pass
class HelixUnsupportedFeatureError(HelixError): pass
class HelixInternalError(HelixError): pass


# ── Log level ─────────────────────────────────────────────────────────────────

class LogLevel(IntEnum):
    OFF   = 0
    ERROR = 1
    WARN  = 2
    INFO  = 3
    DEBUG = 4
    TRACE = 5


# ── Options dataclasses ───────────────────────────────────────────────────────

_LEVEL_NAMES = {0: "off", 1: "error", 2: "warn", 3: "info", 4: "debug", 5: "trace"}


@dataclass
class RuntimeOptions:
    log_level: LogLevel = LogLevel.WARN
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d: dict[str, Any] = dict(self.extra)
        d["log_level"] = _LEVEL_NAMES.get(self.log_level.value, "warn")
        return json.dumps(d)


@dataclass
class ModelOptions:
    alias: str = ""
    # n_gpu_layers: number of model layers to offload to GPU.
    #   -1 (default) means "auto" — the runtime decides based on available VRAM.
    #   0 means CPU-only.  Positive values request that many layers.
    #   The string forms "auto" and "all" are NOT supported via this field;
    #   use -1 for auto or set n_gpu_layers in `extra` if the backend accepts strings.
    n_gpu_layers: int = -1
    n_ctx: int = 0
    mmproj_path: str = ""
    reasoning_format: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self, model_path: str) -> str:
        d: dict[str, Any] = {}
        d.update(self.extra)
        d["model_path"] = model_path
        if self.alias:
            d["alias"] = self.alias
        if self.n_gpu_layers != -1:
            d["n_gpu_layers"] = self.n_gpu_layers
        if self.n_ctx:
            d["n_ctx"] = self.n_ctx
        if self.mmproj_path:
            d["mmproj_path"] = self.mmproj_path
        if self.reasoning_format != "auto":
            d["reasoning_format"] = self.reasoning_format
        return json.dumps(d)


@dataclass
class SpeculativeOptions:
    """Multi-token-prediction (MTP) speculative decoding options.

    Set ``type="draft-mtp"`` to enable. For Gemma-4 and similar architectures
    that ship the MTP head as a separate model, ``model_path`` must point to
    the draft GGUF. When ``type="none"`` (default), speculative decoding is
    disabled and all other fields are ignored.
    """
    type: str = "none"               # "none" | "draft-mtp"
    model_path: Optional[str] = None
    n_max: int = 3
    n_min: int = 0
    p_min: float = 0.0
    backend_sampling: bool = True
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"

    def to_dict(self) -> dict[str, Any]:
        if self.type == "none":
            return {}
        d: dict[str, Any] = {"type": self.type}
        if self.model_path is not None:
            d["model_path"] = self.model_path
        if self.n_max != 3:
            d["n_max"] = self.n_max
        if self.n_min != 0:
            d["n_min"] = self.n_min
        if self.p_min != 0.0:
            d["p_min"] = self.p_min
        if not self.backend_sampling:
            d["backend_sampling"] = False
        if self.cache_type_k != "f16":
            d["cache_type_k"] = self.cache_type_k
        if self.cache_type_v != "f16":
            d["cache_type_v"] = self.cache_type_v
        return d


@dataclass
class SessionOptions:
    n_ctx: int = 0
    swa_full: bool = True
    speculative: Optional[SpeculativeOptions] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d: dict[str, Any] = dict(self.extra)
        if self.n_ctx:
            d["n_ctx"] = self.n_ctx
        if not self.swa_full:
            d["swa_full"] = False
        if self.speculative is not None:
            spec = self.speculative.to_dict()
            if spec:
                d["speculative"] = spec
        return json.dumps(d) if d else "{}"


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class ToolCallFunction:
    name: str
    arguments: str

@dataclass
class ToolCall:
    id: str
    type: str
    function: ToolCallFunction

@dataclass
class LogprobTokenInfo:
    token: str
    logprob: float
    bytes: Optional[list[int]] = None

@dataclass
class LogprobContent:
    token: str
    logprob: float
    bytes: Optional[list[int]] = None
    top_logprobs: list[LogprobTokenInfo] = field(default_factory=list)

@dataclass
class Logprobs:
    content: list[LogprobContent] = field(default_factory=list)

@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None

@dataclass
class CompletionTokensDetails:
    reasoning_tokens: int = 0

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: Optional[CompletionTokensDetails] = None

@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: Optional[str]
    logprobs: Optional[Logprobs] = None

@dataclass
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatCompletion":
        try:
            choices = []
            for c in d.get("choices", []):
                m = c.get("message", {})
                tc_list = None
                if m.get("tool_calls"):
                    tc_list = [
                        ToolCall(
                            id=tc["id"],
                            type=tc["type"],
                            function=ToolCallFunction(
                                name=tc["function"]["name"],
                                arguments=tc["function"]["arguments"],
                            ),
                        )
                        for tc in m["tool_calls"]
                    ]
                lp = None
                if c.get("logprobs") and c["logprobs"].get("content"):
                    lp = Logprobs(content=[
                        LogprobContent(
                            token=lc["token"],
                            logprob=lc["logprob"],
                            bytes=lc.get("bytes"),
                            top_logprobs=[
                                LogprobTokenInfo(t["token"], t["logprob"], t.get("bytes"))
                                for t in lc.get("top_logprobs", [])
                            ],
                        )
                        for lc in c["logprobs"]["content"]
                    ])
                msg = Message(
                    role=m.get("role", "assistant"),
                    content=m.get("content"),
                    tool_calls=tc_list,
                    tool_call_id=m.get("tool_call_id"),
                    reasoning_content=m.get("reasoning_content"),
                )
                choices.append(Choice(
                    index=c.get("index", 0),
                    message=msg,
                    finish_reason=c.get("finish_reason"),
                    logprobs=lp,
                ))
            u = d.get("usage", {})
            ctd = None
            if u.get("completion_tokens_details"):
                ctd = CompletionTokensDetails(
                    reasoning_tokens=u["completion_tokens_details"].get("reasoning_tokens", 0)
                )
            usage = Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
                completion_tokens_details=ctd,
            )
            return cls(
                id=d.get("id", ""),
                object=d.get("object", "chat.completion"),
                created=d.get("created", 0),
                model=d.get("model", ""),
                choices=choices,
                usage=usage,
            )
        except (KeyError, TypeError, ValueError) as e:
            raise HelixError(f"Malformed response data: {e}") from e


@dataclass
class Delta:
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    reasoning_content: Optional[str] = None

@dataclass
class ChunkChoice:
    index: int
    delta: Delta
    finish_reason: Optional[str]
    logprobs: Optional[Logprobs] = None

@dataclass
class ChatCompletionChunk:
    id: str
    object: str
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Optional[Usage] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatCompletionChunk":
        try:
            choices = []
            for c in d.get("choices", []):
                dlt = c.get("delta", {})
                tc_list = None
                if dlt.get("tool_calls"):
                    tc_list = [
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=ToolCallFunction(
                                name=tc.get("function", {}).get("name", ""),
                                arguments=tc.get("function", {}).get("arguments", ""),
                            ),
                        )
                        for tc in dlt["tool_calls"]
                    ]
                delta = Delta(
                    role=dlt.get("role"),
                    content=dlt.get("content"),
                    tool_calls=tc_list,
                    reasoning_content=dlt.get("reasoning_content"),
                )
                choices.append(ChunkChoice(
                    index=c.get("index", 0),
                    delta=delta,
                    finish_reason=c.get("finish_reason"),
                ))
            u = d.get("usage")
            usage = None
            if u:
                usage = Usage(
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    total_tokens=u.get("total_tokens", 0),
                )
            return cls(
                id=d.get("id", ""),
                object=d.get("object", "chat.completion.chunk"),
                created=d.get("created", 0),
                model=d.get("model", ""),
                choices=choices,
                usage=usage,
            )
        except (KeyError, TypeError, ValueError) as e:
            raise HelixError(f"Malformed response data: {e}") from e


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    def __init__(self, ptr: Any) -> None:
        self._ptr = ptr
        self._stream_active = threading.Lock()

    def chat_completions(self, **req: Any) -> ChatCompletion:
        if self._ptr is None:
            raise HelixError("Operation on closed Session")
        request_json = json.dumps(req).encode()
        lib = get_lib()
        out = _ffi.new("char**")
        rc = lib.helix_chat_completions(self._ptr, request_json, out)
        check(rc, "chat_completions")
        raw = _ffi.string(out[0]).decode()
        lib.helix_free(out[0])
        return ChatCompletion.from_dict(json.loads(raw))

    def stream_chat_completions(self, **req: Any) -> Iterator[ChatCompletionChunk]:
        if self._ptr is None:
            raise HelixError("Operation on closed Session")
        if not self._stream_active.acquire(blocking=False):
            raise HelixError("A stream is already active on this Session")
        cancel_flag = threading.Event()
        q: queue.Queue[Optional[str]] = queue.Queue()
        lib = get_lib()
        req["stream"] = True
        request_json = json.dumps(req).encode()
        exc_holder: list[Exception] = []

        @_ffi.callback("int(void*, const char*)")
        def _cb(user_data: Any, chunk_json: Any) -> int:
            if chunk_json == _ffi.NULL:
                q.put(None)
                return 0
            q.put(_ffi.string(chunk_json).decode(errors='replace'))
            return 1 if cancel_flag.is_set() else 0

        def _run() -> None:
            try:
                rc = lib.helix_chat_completions_stream(self._ptr, request_json, _cb, _ffi.NULL)
                if rc != 0 and rc != -9:
                    from helix._ffi import check as _check
                    try:
                        _check(rc, "stream_chat_completions")
                    except Exception as e:
                        exc_holder.append(e)
            except Exception as e:
                exc_holder.append(e)
            finally:
                q.put(None)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            while True:
                item = q.get(timeout=_STREAM_TIMEOUT)
                if item is None:
                    break
                yield ChatCompletionChunk.from_dict(json.loads(item))
        finally:
            cancel_flag.set()
            t.join()
            self._stream_active.release()
            if exc_holder:
                raise exc_holder[0]

    def cancel(self) -> None:
        if self._ptr is not None:
            get_lib().helix_session_cancel(self._ptr)

    def close(self) -> None:
        if self._ptr is not None:
            get_lib().helix_session_destroy(self._ptr)
            self._ptr = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class AsyncSession:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def chat_completions(self, **req: Any) -> ChatCompletion:
        return await asyncio.to_thread(self._session.chat_completions, **req)

    async def stream_chat_completions(self, **req: Any) -> AsyncIterator[ChatCompletionChunk]:
        q: asyncio.Queue[Optional[ChatCompletionChunk]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _produce() -> None:
            try:
                for chunk in self._session.stream_chat_completions(**req):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        t = threading.Thread(target=_produce, daemon=True)
        t.start()
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            self._session.cancel()

    async def cancel(self) -> None:
        self._session.cancel()

    async def close(self) -> None:
        await asyncio.to_thread(self._session.close)

    async def __aenter__(self) -> "AsyncSession":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


# ── Model ─────────────────────────────────────────────────────────────────────

class Model:
    def __init__(self, ptr: Any) -> None:
        self._ptr = ptr

    def session(self, options: Optional[SessionOptions] = None) -> Session:
        if self._ptr is None:
            raise HelixError("Operation on closed Model")
        opts = options or SessionOptions()
        lib = get_lib()
        out = _ffi.new("helix_session_t**")
        rc = lib.helix_session_create(self._ptr, opts.to_json().encode(), out)
        check(rc, "session_create")
        return Session(out[0])

    def async_session(self, options: Optional[SessionOptions] = None) -> AsyncSession:
        return AsyncSession(self.session(options))

    def describe(self) -> dict[str, Any]:
        if self._ptr is None:
            raise HelixError("Operation on closed Model")
        raw = _ffi.string(get_lib().helix_model_describe(self._ptr)).decode()
        return json.loads(raw)

    def close(self) -> None:
        if self._ptr is not None:
            get_lib().helix_model_release(self._ptr)
            self._ptr = None

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── Helix runtime ─────────────────────────────────────────────────────────────

class Helix:
    def __init__(self, options: Optional[RuntimeOptions] = None) -> None:
        opts = options or RuntimeOptions()
        lib = get_lib()
        out = _ffi.new("helix_runtime_t**")
        rc = lib.helix_runtime_create(opts.to_json().encode(), out)
        check(rc, "runtime_create")
        self._ptr = out[0]

    def load_model(self, model_path: str,
                   options: Optional[ModelOptions] = None) -> Model:
        if self._ptr is None:
            raise HelixError("Operation on closed Helix runtime")
        opts = options or ModelOptions()
        lib = get_lib()
        out = _ffi.new("helix_model_t**")
        rc = lib.helix_model_load(self._ptr, opts.to_json(model_path).encode(), out)
        check(rc, "model_load")
        return Model(out[0])

    def describe(self) -> dict[str, Any]:
        if self._ptr is None:
            raise HelixError("Operation on closed Helix runtime")
        raw = _ffi.string(get_lib().helix_runtime_describe(self._ptr)).decode()
        return json.loads(raw)

    def abi_version(self) -> int:
        return get_lib().helix_abi_version()

    def version_string(self) -> str:
        return _ffi.string(get_lib().helix_version_string()).decode()

    def close(self) -> None:
        if self._ptr is not None:
            get_lib().helix_runtime_destroy(self._ptr)
            self._ptr = None

    def __enter__(self) -> "Helix":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
