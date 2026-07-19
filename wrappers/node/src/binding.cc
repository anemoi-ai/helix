/**
 * @helix/node — N-API binding for libhelix.
 *
 * Exposes 5 JS constructors/functions:
 *   createRuntime(optionsJson)   → runtime handle
 *   loadModel(runtime, modelJson) → model handle
 *   createSession(model, sessionJson) → session handle
 *   chatCompletions(session, requestJson) → responseJson string
 *   streamChatCompletions(session, requestJson, onChunk, onDone)
 *   cancelSession(session)
 *   destroySession(session)
 *   destroyModel(model)
 *   destroyRuntime(runtime)
 *   lastErrorJson() → string
 *   abiVersion() → number
 *   versionString() → string
 */

#include <napi.h>
#include <helix.h>
#include <string>
#include <cstring>
#include <thread>

// ── Wrapper objects stored as external values ─────────────────────────────────

struct RuntimeWrap  { helix_runtime_t* ptr; };
struct ModelWrap    { helix_model_t*   ptr; Napi::Reference<Napi::External<RuntimeWrap>> runtime_ref; };
struct SessionWrap  { helix_session_t* ptr; Napi::Reference<Napi::External<ModelWrap>> model_ref; };

// ── Helpers ────────────────────────────────────────────────────────────────────

static Napi::Error helix_error(Napi::Env env, helix_status_t rc) {
    std::string msg = helix_last_error_json();
    auto err = Napi::Error::New(env, msg);
    err.Set("helixStatus", Napi::Number::New(env, rc));
    err.Set("helixErrorJson", Napi::String::New(env, msg));
    return err;
}

static std::string get_string(const Napi::Value& val) {
    return val.IsString() ? val.As<Napi::String>().Utf8Value() : "{}";
}

// ── Runtime ────────────────────────────────────────────────────────────────────

static Napi::Value CreateRuntime(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::string opts = info.Length() > 0 ? get_string(info[0]) : "{}";

    auto* wrap = new RuntimeWrap{};
    helix_status_t rc = helix_runtime_create(opts.c_str(), &wrap->ptr);
    if (rc != HELIX_OK) {
        delete wrap;
        helix_error(env, rc).ThrowAsJavaScriptException();
        return env.Null();
    }

    auto ext = Napi::External<RuntimeWrap>::New(env, wrap,
        [](Napi::Env, RuntimeWrap* w) { helix_runtime_destroy(w->ptr); delete w; });
    return ext;
}

static Napi::Value LoadModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) { Napi::TypeError::New(env, "expected (runtime, modelJson)").ThrowAsJavaScriptException(); return env.Null(); }

    auto runtime_ext = info[0].As<Napi::External<RuntimeWrap>>();
    auto* rw = runtime_ext.Data();
    std::string model_json = get_string(info[1]);

    auto* wrap = new ModelWrap{};
    helix_status_t rc = helix_model_load(rw->ptr, model_json.c_str(), &wrap->ptr);
    if (rc != HELIX_OK) {
        delete wrap;
        helix_error(env, rc).ThrowAsJavaScriptException();
        return env.Null();
    }

    wrap->runtime_ref = Napi::Reference<Napi::External<RuntimeWrap>>::New(
        runtime_ext, 1);

    return Napi::External<ModelWrap>::New(env, wrap,
        [](Napi::Env, ModelWrap* w) { helix_model_release(w->ptr); delete w; });
}

static Napi::Value CreateSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto model_ext = info[0].As<Napi::External<ModelWrap>>();
    auto* mw = model_ext.Data();
    std::string session_json = info.Length() > 1 ? get_string(info[1]) : "{}";

    auto* wrap = new SessionWrap{};
    helix_status_t rc = helix_session_create(mw->ptr, session_json.c_str(), &wrap->ptr);
    if (rc != HELIX_OK) {
        delete wrap;
        helix_error(env, rc).ThrowAsJavaScriptException();
        return env.Null();
    }

    wrap->model_ref = Napi::Reference<Napi::External<ModelWrap>>::New(
        model_ext, 1);

    return Napi::External<SessionWrap>::New(env, wrap,
        [](Napi::Env, SessionWrap* w) { helix_session_destroy(w->ptr); delete w; });
}

// ── Chat completions (sync) ───────────────────────────────────────────────────

static Napi::Value ChatCompletions(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto* sw = info[0].As<Napi::External<SessionWrap>>().Data();
    std::string req = get_string(info[1]);

    char* out = nullptr;
    helix_status_t rc = helix_chat_completions(sw->ptr, req.c_str(), &out);
    if (rc != HELIX_OK) {
        helix_error(env, rc).ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::String result = Napi::String::New(env, out);
    helix_free(out);
    return result;
}

// ── Streaming (threadsafe-function) ──────────────────────────────────────────

struct StreamCtx {
    Napi::ThreadSafeFunction tsfn;
    Napi::Reference<Napi::External<SessionWrap>> session_ref;
    helix_session_t* session_ptr;
    std::string request_json;
};

// Data ferried from C callback thread → JS main thread
struct ChunkData {
    std::string json;  // empty = done sentinel
    bool is_done;
};

static int stream_callback(void* user_data, const char* chunk_json) {
    auto* ctx = static_cast<StreamCtx*>(user_data);
    auto* data = new ChunkData{
        chunk_json ? std::string(chunk_json) : "",
        chunk_json == nullptr
    };

    auto status = ctx->tsfn.NonBlockingCall(data,
        [](Napi::Env env, Napi::Function js_cb, ChunkData* d) {
            Napi::HandleScope scope(env);
            if (d->is_done) {
                js_cb.Call({env.Null(), Napi::Boolean::New(env, true)});
            } else {
                js_cb.Call({Napi::String::New(env, d->json), Napi::Boolean::New(env, false)});
            }
            delete d;
        });

    if (status != napi_ok) {
        delete data;
        return 1; // signal cancel
    }
    return 0;
}

static Napi::Value StreamChatCompletions(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "expected (session, requestJson, callback)").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto* sw = info[0].As<Napi::External<SessionWrap>>().Data();
    std::string req = get_string(info[1]);
    Napi::Function cb = info[2].As<Napi::Function>();

    auto session_ext = info[0].As<Napi::External<SessionWrap>>();

    auto* ctx = new StreamCtx{};
    ctx->session_ptr = sw->ptr;
    ctx->request_json = req;
    ctx->session_ref = Napi::Reference<Napi::External<SessionWrap>>::New(
        session_ext, 1);

    ctx->tsfn = Napi::ThreadSafeFunction::New(
        env, cb, "helix_stream", 0, 1,
        [ctx](Napi::Env) { delete ctx; });

    ctx->tsfn.Acquire();

    // Spawn worker thread that calls helix_chat_completions_stream (blocking)
    std::thread([ctx]() {
        helix_chat_completions_stream(
            ctx->session_ptr,
            ctx->request_json.c_str(),
            stream_callback,
            ctx
        );
        ctx->tsfn.Release();
        ctx->tsfn.Release();
    }).detach();

    return env.Undefined();
}

static Napi::Value CancelSession(const Napi::CallbackInfo& info) {
    auto* sw = info[0].As<Napi::External<SessionWrap>>().Data();
    helix_session_cancel(sw->ptr);
    return info.Env().Undefined();
}

static Napi::Value LastErrorJson(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), helix_last_error_json());
}

static Napi::Value AbiVersion(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), helix_abi_version());
}

static Napi::Value VersionString(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), helix_version_string());
}

// ── Module init ────────────────────────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("createRuntime",         Napi::Function::New(env, CreateRuntime));
    exports.Set("loadModel",             Napi::Function::New(env, LoadModel));
    exports.Set("createSession",         Napi::Function::New(env, CreateSession));
    exports.Set("chatCompletions",       Napi::Function::New(env, ChatCompletions));
    exports.Set("streamChatCompletions", Napi::Function::New(env, StreamChatCompletions));
    exports.Set("cancelSession",         Napi::Function::New(env, CancelSession));
    exports.Set("lastErrorJson",         Napi::Function::New(env, LastErrorJson));
    exports.Set("abiVersion",            Napi::Function::New(env, AbiVersion));
    exports.Set("versionString",         Napi::Function::New(env, VersionString));
    return exports;
}

NODE_API_MODULE(helix_node, Init)
