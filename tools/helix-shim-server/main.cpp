/* helix-shim-server — thin HTTP/1.1 adapter over libhelix for testing.
 *
 * NOT a product. Used only for running conformance tests against the
 * OpenAI Python SDK.
 *
 * Usage:
 *   helix-shim-server --model /path/to/model.gguf [--alias name]
 *                     [--host 127.0.0.1] [--port 8080]
 */

#define CPPHTTPLIB_THREAD_POOL_COUNT 4
#include "httplib.h"
#include "helix.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using nj = nlohmann::json;

static helix_runtime_t* g_runtime = nullptr;
static helix_model_t*   g_model   = nullptr;
static std::string       g_alias;

static void log_cb(void* /*ud*/, helix_log_level_t level, const char* msg) {
    if (level <= HELIX_LOG_INFO)
        fprintf(stderr, "[helix/%d] %s\n", level, msg);
}

/* ------------------------------------------------------------------ */
/*  Non-streaming handler                                              */
/* ------------------------------------------------------------------ */

static void handle_completions_sync(helix_session_t* sess,
                                     const std::string& body,
                                     httplib::Response& res) {
    char* response_json = nullptr;
    helix_status_t st = helix_chat_completions(sess, body.c_str(), &response_json);
    helix_session_destroy(sess);

    if (st != HELIX_OK) {
        res.status = (st == HELIX_E_VALIDATION || st == HELIX_E_INVALID_JSON ||
                      st == HELIX_E_INVALID_ARG) ? 400 : 500;
        res.set_content(helix_last_error_json(), "application/json");
        return;
    }

    res.status = 200;
    res.set_content(response_json, "application/json");
    helix_free(response_json);
}

/* ------------------------------------------------------------------ */
/*  Streaming handler (SSE)                                            */
/* ------------------------------------------------------------------ */

struct SseCbData {
    httplib::DataSink* sink;
    bool               closed = false;
};

static int sse_callback(void* user_data, const char* chunk_json) {
    auto* d = static_cast<SseCbData*>(user_data);
    if (d->closed) return 1;
    if (chunk_json == nullptr) {
        /* Stream done: write terminating event. */
        const char done[] = "data: [DONE]\n\n";
        if (!d->sink->write(done, sizeof(done) - 1)) {
            d->closed = true;
        }
        return 0;
    }
    /* Build SSE line: "data: <json>\n\n" */
    std::string line = "data: ";
    line += chunk_json;
    line += "\n\n";
    if (!d->sink->write(line.data(), line.size())) {
        d->closed = true;
        return 1;
    }
    return 0;
}

static void handle_completions_stream(helix_session_t* sess,
                                       const std::string& body,
                                       httplib::Response& res) {
    res.status = 200;
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    /* The content provider runs synchronously in the HTTP worker thread.
     * helix_chat_completions_stream blocks until generation is complete;
     * each token write via sse_callback flushes to the socket immediately. */
    res.set_chunked_content_provider("text/event-stream",
        [sess, body](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            SseCbData cbd{&sink, false};
            helix_status_t st = helix_chat_completions_stream(
                sess, body.c_str(), sse_callback, &cbd);
            helix_session_destroy(sess);

            if (st != HELIX_OK && !cbd.closed) {
                /* Emit the error as a data event so the client sees it. */
                std::string err = "data: ";
                err += helix_last_error_json();
                err += "\n\n";
                sink.write(err.data(), err.size());
            }
            return false;  /* tell httplib we are done */
        });
}

/* ------------------------------------------------------------------ */
/*  Unified /v1/chat/completions handler                              */
/* ------------------------------------------------------------------ */

static void handle_completions(const httplib::Request& req,
                                 httplib::Response& res) {
    helix_session_t* sess = nullptr;
    helix_status_t   st   = helix_session_create(g_model, nullptr, &sess);
    if (st != HELIX_OK) {
        res.status = 500;
        res.set_content(helix_last_error_json(), "application/json");
        return;
    }

    /* Detect stream:true without fully parsing the request. */
    bool is_stream = false;
    try {
        nj j = nj::parse(req.body);
        if (j.contains("stream") && j["stream"].is_boolean())
            is_stream = j["stream"].get<bool>();
    } catch (...) {
        /* JSON parse error will be caught properly inside helix. */
    }

    if (is_stream) {
        handle_completions_stream(sess, req.body, res);
    } else {
        handle_completions_sync(sess, req.body, res);
    }
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* alias_arg  = "local-helix";
    const char* host       = "127.0.0.1";
    int         port       = 8080;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--alias") && i+1 < argc) alias_arg  = argv[++i];
        else if (!strcmp(argv[i], "--host")  && i+1 < argc) host       = argv[++i];
        else if (!strcmp(argv[i], "--port")  && i+1 < argc) {
            try { port = std::stoi(argv[++i]); }
            catch (...) { fprintf(stderr, "Invalid port: %s\n", argv[i]); return 1; }
            if (port <= 0 || port > 65535) {
                fprintf(stderr, "Port out of range (1-65535): %d\n", port);
                return 1;
            }
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: helix-shim-server --model /path/to/model.gguf "
                        "[--alias name] [--host 127.0.0.1] [--port 8080]\n");
        return 1;
    }

    g_alias = alias_arg;

    helix_set_log_callback(log_cb, nullptr, HELIX_LOG_INFO);

    if (helix_runtime_create(nullptr, &g_runtime) != HELIX_OK) {
        fprintf(stderr, "helix_runtime_create failed: %s\n", helix_last_error_json());
        return 1;
    }

    nj model_json_obj;
    model_json_obj["model_path"] = model_path;
    model_json_obj["alias"]      = alias_arg;
    std::string model_json       = model_json_obj.dump();

    if (helix_model_load(g_runtime, model_json.c_str(), &g_model) != HELIX_OK) {
        fprintf(stderr, "helix_model_load failed: %s\n", helix_last_error_json());
        helix_runtime_destroy(g_runtime);
        return 1;
    }

    httplib::Server svr;

    /* Cap request bodies so a stray huge (or malicious) payload can't drive the
     * JSON/base64 image path into an OOM. 50 MiB is well above any realistic
     * multimodal request; httplib returns 413 Payload Too Large above it. */
    svr.set_payload_max_length(50ull * 1024 * 1024);

    svr.Post("/v1/chat/completions", handle_completions);

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        std::string body = "{\"object\":\"list\",\"data\":[{\"id\":\"" +
                           g_alias + "\",\"object\":\"model\",\"owned_by\":\"helix\"}]}";
        res.set_content(body, "application/json");
    });

    fprintf(stderr, "[helix-shim-server] Listening on %s:%d\n", host, port);
    fprintf(stderr, "[helix-shim-server] Model: %s  alias: %s\n", model_path, alias_arg);

    svr.listen(host, port);

    helix_model_release(g_model);
    helix_runtime_destroy(g_runtime);
    return 0;
}
