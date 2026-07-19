"""Low-level cffi bindings for libhelix."""

from __future__ import annotations

import json
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Any

import cffi

_ffi = cffi.FFI()

_ffi.cdef("""
    typedef int32_t helix_status_t;

    typedef enum {
        HELIX_LOG_OFF   = 0,
        HELIX_LOG_ERROR = 1,
        HELIX_LOG_WARN  = 2,
        HELIX_LOG_INFO  = 3,
        HELIX_LOG_DEBUG = 4,
        HELIX_LOG_TRACE = 5
    } helix_log_level_t;

    typedef struct helix_runtime helix_runtime_t;
    typedef struct helix_model   helix_model_t;
    typedef struct helix_session helix_session_t;

    typedef int (*helix_stream_cb)(void* user_data, const char* chunk_json);
    typedef void (*helix_log_cb)(void* user_data, helix_log_level_t level, const char* msg);

    uint32_t    helix_abi_version(void);
    const char* helix_version_string(void);
    const char* helix_last_error_json(void);

    helix_status_t helix_runtime_create(const char* options_json, helix_runtime_t** out_runtime);
    void           helix_runtime_destroy(helix_runtime_t* runtime);
    const char*    helix_runtime_describe(helix_runtime_t* runtime);

    helix_status_t helix_model_load(helix_runtime_t* runtime, const char* model_json, helix_model_t** out_model);
    void           helix_model_release(helix_model_t* model);
    const char*    helix_model_describe(helix_model_t* model);

    helix_status_t helix_session_create(helix_model_t* model, const char* session_json, helix_session_t** out_session);
    void           helix_session_destroy(helix_session_t* session);

    helix_status_t helix_chat_completions(helix_session_t* session, const char* request_json, char** out_response_json);
    helix_status_t helix_chat_completions_stream(helix_session_t* session, const char* request_json,
                                                  helix_stream_cb on_chunk, void* user_data);
    void           helix_session_cancel(helix_session_t* session);
    void           helix_free(char* ptr);

    void helix_set_log_callback(helix_log_cb cb, void* user_data, helix_log_level_t min_level);
""")

# Status codes
HELIX_OK                    =   0
HELIX_E_INVALID_ARG         =  -1
HELIX_E_INVALID_JSON        =  -2
HELIX_E_VALIDATION          =  -3
HELIX_E_MODEL_NOT_FOUND     =  -4
HELIX_E_MODEL_LOAD_FAILED   =  -5
HELIX_E_OOM                 =  -6
HELIX_E_VRAM_EXHAUSTED      =  -7
HELIX_E_CONTEXT_FULL        =  -8
HELIX_E_CANCELLED           =  -9
HELIX_E_BACKEND             = -10
HELIX_E_UNSUPPORTED_FEATURE = -11
HELIX_E_INTERNAL            = -99


def _find_lib() -> str:
    """Return path to libhelix shared library."""
    # 1. Bundled alongside this package (wheel distribution)
    pkg_dir = Path(__file__).parent
    for name in ("libhelix.so", "libhelix.dylib", "helix.dll"):
        candidate = pkg_dir / name
        if candidate.exists():
            return str(candidate)

    # 2. HELIX_LIB_PATH environment variable
    env_path = os.environ.get("HELIX_LIB_PATH")
    if env_path and Path(env_path).exists():
        warnings.warn(
            "HELIX_LIB_PATH is set; loading shared library from untrusted path. "
            "Ensure this is intentional.",
            RuntimeWarning,
            stacklevel=3,
        )
        return env_path

    # 3. Standard system search (let the OS find it)
    if sys.platform == "win32":
        return "helix.dll"
    if sys.platform == "darwin":
        return "libhelix.dylib"
    return "libhelix.so"


_lib: Any = None
_lib_lock = threading.Lock()


def get_lib() -> Any:
    global _lib
    if _lib is None:
        with _lib_lock:
            if _lib is None:
                _lib = _ffi.dlopen(_find_lib())
    return _lib


def _last_error() -> dict[str, Any]:
    raw = _ffi.string(get_lib().helix_last_error_json()).decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": {"message": raw, "type": "internal_error", "code": None, "param": None}}


def check(rc: int, context: str = "") -> None:
    """Raise the appropriate HelixError subclass for a non-OK status code."""
    if rc == HELIX_OK:
        return
    err = _last_error().get("error", {})
    msg = err.get("message", context or f"helix error {rc}")
    typ = err.get("type", "")
    param = err.get("param")
    code = err.get("code")

    from helix._api import (
        HelixInvalidArgError,
        HelixInvalidJsonError,
        HelixValidationError,
        HelixModelNotFoundError,
        HelixModelLoadFailedError,
        HelixOomError,
        HelixVramExhaustedError,
        HelixContextFullError,
        HelixCancelledError,
        HelixBackendError,
        HelixUnsupportedFeatureError,
        HelixInternalError,
        HelixError,
    )

    _map = {
        HELIX_E_INVALID_ARG:         HelixInvalidArgError,
        HELIX_E_INVALID_JSON:        HelixInvalidJsonError,
        HELIX_E_VALIDATION:          HelixValidationError,
        HELIX_E_MODEL_NOT_FOUND:     HelixModelNotFoundError,
        HELIX_E_MODEL_LOAD_FAILED:   HelixModelLoadFailedError,
        HELIX_E_OOM:                 HelixOomError,
        HELIX_E_VRAM_EXHAUSTED:      HelixVramExhaustedError,
        HELIX_E_CONTEXT_FULL:        HelixContextFullError,
        HELIX_E_CANCELLED:           HelixCancelledError,
        HELIX_E_BACKEND:             HelixBackendError,
        HELIX_E_UNSUPPORTED_FEATURE: HelixUnsupportedFeatureError,
        HELIX_E_INTERNAL:            HelixInternalError,
    }
    exc_cls = _map.get(rc, HelixError)
    raise exc_cls(msg, param=param, code=code, status=rc)
