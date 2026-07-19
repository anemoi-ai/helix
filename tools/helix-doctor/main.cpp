/* helix-doctor — diagnostic tool for verifying a helix installation.
 *
 * Usage:
 *   helix-doctor [--model /path/to/model.gguf] [--alias my-model] [--explain]
 *
 * Without --explain: prints a brief runtime summary and optionally runs a
 * one-token completion.
 *
 * With --explain: prints the full hardware profile and the auto-tuned plan with
 * rationale annotations. Use this when debugging "Helix is slow on my machine".
 *
 * helix.h is a C ABI header; this tool links it from C++ and parses the
 * *_describe() JSON with the vendored nlohmann/json so the navigation is
 * structure-aware (e.g. cpu.model_name) rather than relying on fragile flat
 * string searches. A separate C translation unit (c_abi_check.c) keeps the
 * "helix.h compiles as C99" guarantee under test.
 */

#include "helix.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using nj = nlohmann::json;

/* Defined in c_abi_check.c (compiled as C) — see that file. */
extern "C" void helix_doctor_c_abi_check(void);

static void print_section(const char* title) {
    printf("\n=== %s ===\n", title);
}

static void check(helix_status_t status, const char* step) {
    if (status != HELIX_OK) {
        fprintf(stderr, "FAIL [%s]: status=%d\n", step, status);
        fprintf(stderr, "  error: %s\n", helix_last_error_json());
        exit(1);
    }
}

static void log_cb(void* /*userdata*/, helix_log_level_t level,
                   const char* msg) {
    if (level <= HELIX_LOG_WARN) {
        fprintf(stderr, "[helix] %s\n", msg);
    }
}

/* Render a JSON value (string/number/bool) as a plain display string.
 * Returns "?" when the key is absent so the output always has a placeholder. */
static std::string field(const nj& obj, const char* key) {
    if (!obj.is_object()) return "?";
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) return "?";
    if (it->is_string()) return it->get<std::string>();
    return it->dump(); /* numbers and bools render without quotes */
}

static void print_explain(helix_runtime_t* rt) {
    nj desc;
    try {
        desc = nj::parse(helix_runtime_describe(rt));
    } catch (const std::exception& e) {
        fprintf(stderr, "failed to parse runtime describe JSON: %s\n", e.what());
        return;
    }

    const nj& cpu = desc.value("cpu", nj::object());
    const nj& ram = desc.value("ram", nj::object());

    printf("\n");
    printf("─── Hardware ─────────────────────────────────────────────────\n");
    printf("CPU:     %s\n", field(cpu, "model_name").c_str());
    printf("         logical=%s  physical=%s  perf=%s  eff=%s\n",
           field(cpu, "logical_cores").c_str(),
           field(cpu, "physical_cores").c_str(),
           field(cpu, "performance_cores").c_str(),
           field(cpu, "efficiency_cores").c_str());
    printf("         ISA tier: %s  |  Memory bandwidth: %s\n",
           field(cpu, "isa_tier").c_str(),
           field(cpu, "memory_bandwidth").c_str());
    printf("RAM:     total=%s B  available=%s B\n",
           field(ram, "total_bytes").c_str(),
           field(ram, "available_bytes").c_str());

    /* Print GPU section if any accelerators are present. */
    const nj& accels = desc.value("accelerators", nj::array());
    for (size_t idx = 0; idx < accels.size(); ++idx) {
        const nj& a = accels[idx];
        if (idx == 0)
            printf("\n─── Accelerators ──────────────────────────────────────────────\n");
        printf("[%zu] %s (%s)\n", idx,
               field(a, "device_name").c_str(),
               field(a, "backend").c_str());
        printf("    VRAM: %s B total, %s B free\n",
               field(a, "total_vram_bytes").c_str(),
               field(a, "free_vram_bytes").c_str());
        if (a.contains("compute_capability"))
            printf("    Compute capability: %s\n", field(a, "compute_capability").c_str());
        if (a.contains("pcie_link_gen"))
            printf("    PCIe: Gen%s x%s\n",
                   field(a, "pcie_link_gen").c_str(),
                   a.contains("pcie_link_width") ? field(a, "pcie_link_width").c_str() : "?");
    }

    printf("\n─── Auto-tune plan ───────────────────────────────────────────\n");
    const nj& at = desc.value("auto_tune", nj::object());
    printf("n_threads        = %-6s  (decode threads)\n",      field(at, "n_threads").c_str());
    printf("n_threads_batch  = %-6s  (prefill threads)\n",     field(at, "n_threads_batch").c_str());
    printf("n_batch          = %-6s  (logical batch size)\n",  field(at, "n_batch").c_str());
    printf("n_ubatch         = %-6s  (physical micro-batch)\n", field(at, "n_ubatch").c_str());
    printf("flash_attention  = %-6s  (ISA-tier / GPU based)\n", field(at, "flash_attn").c_str());
    printf("─────────────────────────────────────────────────────────────\n");
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* alias      = "doctor-model";
    int explain            = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--alias") == 0 && i + 1 < argc) {
            alias = argv[++i];
        } else if (strcmp(argv[i], "--explain") == 0) {
            explain = 1;
        }
    }

    helix_doctor_c_abi_check(); /* keep the C-ABI canary TU linked */

    print_section("helix version");
    printf("  ABI version : 0x%08x\n", helix_abi_version());
    printf("  Version str : %s\n", helix_version_string());

    helix_set_log_callback(log_cb, nullptr, HELIX_LOG_WARN);

    print_section("runtime");
    helix_runtime_t* rt = nullptr;
    check(helix_runtime_create(nullptr, &rt), "helix_runtime_create");

    if (explain) {
        print_explain(rt);
    } else {
        printf("%s\n", helix_runtime_describe(rt));
    }

    if (!model_path) {
        printf("\nNo --model provided. Pass --model /path/to/model.gguf for full validation.\n");
        if (!explain) printf("Pass --explain for the full hardware profile and auto-tune plan.\n");
        helix_runtime_destroy(rt);
        printf("\n[helix-doctor] PASS (runtime only)\n");
        return 0;
    }

    print_section("model load");

    nj model_req;
    model_req["model_path"] = model_path;
    model_req["alias"]      = alias;
    const std::string model_json = model_req.dump();

    helix_model_t* model = nullptr;
    check(helix_model_load(rt, model_json.c_str(), &model), "helix_model_load");
    printf("%s\n", helix_model_describe(model));
    printf("  embed-dim: %u%s\n", helix_model_embedding_dim(model),
           helix_model_embedding_dim(model) == 0
               ? " (model cannot serve helix_embeddings)" : "");

    print_section("session create");
    helix_session_t* sess = nullptr;
    check(helix_session_create(model, nullptr, &sess), "helix_session_create");
    printf("  session created OK\n");

    print_section("one-token completion");
    nj chat_req;
    chat_req["model"]       = alias;
    chat_req["messages"]    = nj::array({ {{"role", "user"}, {"content", "Hi"}} });
    chat_req["max_tokens"]  = 1;
    chat_req["temperature"] = 0;
    const std::string req = chat_req.dump();

    char* response = nullptr;
    check(helix_chat_completions(sess, req.c_str(), &response), "helix_chat_completions");
    printf("  response: %s\n", response);
    helix_free(response);

    print_section("cleanup");
    helix_session_destroy(sess);
    helix_model_release(model);
    helix_runtime_destroy(rt);
    printf("  all handles destroyed OK\n");

    printf("\n[helix-doctor] PASS\n");
    return 0;
}
