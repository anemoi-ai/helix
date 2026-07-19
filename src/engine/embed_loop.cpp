#include "embed_loop.hpp"
#include "embed_packing.hpp"
#include "../session/session.hpp"
#include "../model/model.hpp"
#include "../json/request.hpp"
#include "../json/response.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"

#include "llama.h"
#include "common.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <vector>

namespace helix {

namespace {

struct BatchGuard {
    llama_batch& batch;
    explicit BatchGuard(llama_batch& b) : batch(b) {}
    ~BatchGuard() { llama_batch_free(batch); }
    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;
};

/* Embedding models trained with an EOS/SEP terminator produce degraded
 * vectors when the tokenizer metadata fails to append it — warn-only,
 * matching llama.cpp's examples/embedding. One warning per request. */
void warn_if_missing_eos_sep(const llama_vocab* vocab,
                             const std::vector<std::vector<llama_token>>& toks) {
    const llama_token eos = llama_vocab_eos(vocab);
    const llama_token sep = llama_vocab_sep(vocab);
    if (eos == LLAMA_TOKEN_NULL && sep == LLAMA_TOKEN_NULL) return;
    for (size_t i = 0; i < toks.size(); ++i) {
        const llama_token last = toks[i].back();
        if (last == eos || last == sep) continue;
        log_warn("embeddings: input[" + std::to_string(i) + "] does not end "
                 "with EOS/SEP; check the model's add_eos/add_sep tokenizer "
                 "metadata — embedding quality may degrade");
        return;
    }
}

int32_t run_one_batch(Session& sess, llama_batch& batch) {
    llama_context* ctx = sess.ctx();
    /* Clear all KV before each flush; for the encoder path this is a
     * harmless no-op that keeps the loop uniform. */
    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    return sess.embed_via_encode() ? llama_encode(ctx, batch)
                                   : llama_decode(ctx, batch);
}

} // namespace

helix_status_t run_embeddings(Session& sess,
                              const char* request_json,
                              char** out_response_json) {
    /* 1. Parse and validate. Mode and alias gating mirror the chat path:
     * the embedding-mode check lives in Session::embeddings. */
    EmbeddingsRequest req =
        EmbeddingsRequest::from_json(request_json ? request_json : "");
    req.validate();

    /* 2. Model alias check. */
    if (req.model != sess.model().alias()) {
        throw Error(HELIX_E_MODEL_NOT_FOUND,
                    "model '" + req.model + "' not loaded in this session",
                    "model_not_found", "model", "helix_e_model_not_found");
    }

    llama_context*     ctx   = sess.ctx();
    const llama_model* model = sess.model().llama_model_ptr();
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const uint32_t n_batch    = llama_n_batch(ctx);
    const uint32_t n_seq_max  = llama_n_seq_max(ctx);
    const int32_t  n_embd_out = llama_model_n_embd_out(model);

    /* 3. Tokenize every input up front — cheap, and it lets the whole
     * request be validated before any compute is spent. */
    std::vector<std::vector<llama_token>> toks(req.inputs.size());
    std::vector<size_t> lengths(req.inputs.size());
    int32_t total_tokens = 0;
    for (size_t i = 0; i < req.inputs.size(); ++i) {
        toks[i] = common_tokenize(vocab, req.inputs[i],
                                  /*add_special=*/true, /*parse_special=*/true);
        if (toks[i].empty()) {
            throw Error(HELIX_E_VALIDATION,
                        "input[" + std::to_string(i) + "] produced no tokens",
                        "invalid_request_error", "input", "helix_e_validation");
        }
        if (toks[i].size() > n_batch) {
            /* No silent truncation — mirrors the llama.cpp server guard. */
            throw Error(HELIX_E_CONTEXT_FULL,
                        "input[" + std::to_string(i) + "] is " +
                        std::to_string(toks[i].size()) + " tokens; the "
                        "session batch size is " + std::to_string(n_batch) +
                        ". Shorten the input or raise n_batch/n_ctx in "
                        "session options",
                        "context_length_exceeded", "input",
                        "helix_e_context_full");
        }
        lengths[i] = toks[i].size();
        total_tokens += static_cast<int32_t>(toks[i].size());
    }

    warn_if_missing_eos_sep(vocab, toks);

    /* 4. Pack → flush. The planner is the determinism-critical seam
     * (insertion-ordered; see embed_packing.hpp). */
    const auto flushes = plan_embed_flushes(lengths, n_batch, n_seq_max);

    EmbeddingsResponse resp;
    resp.model         = sess.model().alias();
    resp.encoding      = req.encoding;
    resp.prompt_tokens = total_tokens;
    resp.vectors.resize(req.inputs.size());

    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch),
                                         /*embd=*/0,
                                         static_cast<int32_t>(n_seq_max));
    BatchGuard guard(batch);

    for (const auto& group : flushes) {
        /* Flush-boundary cancellation, same granularity as the chat path.
         * State is only the KV memory, which the next flush clears, so the
         * session stays reusable after cancellation. */
        if (sess.cancel_flag().load(std::memory_order_relaxed)) {
            throw Error(HELIX_E_CANCELLED, "request cancelled",
                        "request_cancelled", "", "helix_e_cancelled");
        }

        common_batch_clear(batch);
        for (size_t s = 0; s < group.size(); ++s) {
            const auto& t = toks[group[s]];
            for (size_t p = 0; p < t.size(); ++p) {
                /* All tokens marked for output: with graph-side pooling the
                 * per-token outputs feed the pooled vector. */
                common_batch_add(batch, t[p], static_cast<llama_pos>(p),
                                 {static_cast<llama_seq_id>(s)}, true);
            }
        }

        const int32_t rc = run_one_batch(sess, batch);
        if (rc == 2) {
            /* Abort-callback path. No abort_callback is installed in 1.1,
             * but map it defensively so a future installation needs no
             * change here. */
            throw Error(HELIX_E_CANCELLED, "request cancelled",
                        "request_cancelled", "", "helix_e_cancelled");
        }
        if (rc != 0) {
            throw Error(HELIX_E_BACKEND,
                        "llama_decode failed for embedding batch (rc=" +
                        std::to_string(rc) + ")",
                        "backend_error", "", "helix_e_backend");
        }

        for (size_t s = 0; s < group.size(); ++s) {
            const float* emb = llama_get_embeddings_seq(
                ctx, static_cast<llama_seq_id>(s));
            if (!emb) {
                throw Error(HELIX_E_INTERNAL,
                            "null sequence embedding (pooling misconfigured)",
                            "internal_error", "", "helix_e_internal");
            }
            /* L2-normalize (euclidean), matching OpenAI's unit vectors. */
            std::vector<float> v(static_cast<size_t>(n_embd_out));
            common_embd_normalize(emb, v.data(), n_embd_out, /*embd_norm=*/2);
            resp.vectors[group[s]] = std::move(v);
        }
    }

    /* 5. Serialize. */
    const std::string json_str = resp.to_json();
    if (out_response_json) {
        /* Allocated with C malloc() so the caller frees it via helix_free()
         * — see the pairing note above helix_free in helix_abi.cpp. */
        *out_response_json = static_cast<char*>(malloc(json_str.size() + 1));
        if (!*out_response_json) throw std::bad_alloc();
        std::copy_n(json_str.c_str(), json_str.size() + 1, *out_response_json);
    }

    return HELIX_OK;
}

helix_status_t run_rerank(Session& sess,
                          const char* request_json,
                          char** out_response_json) {
    /* 1. Parse and validate. Mode gating (rerank sessions only) lives in
     * Session::rerank, mirroring the embeddings path. */
    RerankRequest req = RerankRequest::from_json(request_json ? request_json : "");
    req.validate();

    /* 2. Model alias check. */
    if (req.model != sess.model().alias()) {
        throw Error(HELIX_E_MODEL_NOT_FOUND,
                    "model '" + req.model + "' not loaded in this session",
                    "model_not_found", "model", "helix_e_model_not_found");
    }

    llama_context*     ctx   = sess.ctx();
    const llama_model* model = sess.model().llama_model_ptr();
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const uint32_t n_batch   = llama_n_batch(ctx);
    const uint32_t n_seq_max = llama_n_seq_max(ctx);

    /* 3. Build one token sequence per query+document pair, the same two
     * shapes llama-server uses: a model-supplied "rerank" chat template
     * with {query}/{document} placeholders (Qwen3-Reranker family), or the
     * classic [BOS] query [EOS] [SEP] doc [EOS] assembly driven by the
     * tokenizer's add_bos/add_eos/add_sep metadata (bge-reranker family). */
    const char* rr_tmpl = llama_model_chat_template(model, "rerank");

    std::vector<llama_token> query_toks;
    if (!rr_tmpl) {
        query_toks = common_tokenize(vocab, req.query,
                                     /*add_special=*/false,
                                     /*parse_special=*/false);
    }

    std::vector<std::vector<llama_token>> toks(req.documents.size());
    std::vector<size_t> lengths(req.documents.size());
    int32_t total_tokens = 0;
    for (size_t i = 0; i < req.documents.size(); ++i) {
        if (rr_tmpl) {
            std::string prompt = rr_tmpl;
            string_replace_all(prompt, "{query}",    req.query);
            string_replace_all(prompt, "{document}", req.documents[i]);
            toks[i] = common_tokenize(vocab, prompt,
                                      /*add_special=*/false,
                                      /*parse_special=*/true);
        } else {
            llama_token eos = llama_vocab_eos(vocab);
            if (eos == LLAMA_TOKEN_NULL) eos = llama_vocab_sep(vocab);
            const std::vector<llama_token> doc_toks =
                common_tokenize(vocab, req.documents[i],
                                /*add_special=*/false, /*parse_special=*/false);
            std::vector<llama_token>& t = toks[i];
            t.reserve(query_toks.size() + doc_toks.size() + 4);
            if (llama_vocab_get_add_bos(vocab)) t.push_back(llama_vocab_bos(vocab));
            t.insert(t.end(), query_toks.begin(), query_toks.end());
            if (llama_vocab_get_add_eos(vocab) && eos != LLAMA_TOKEN_NULL)
                t.push_back(eos);
            if (llama_vocab_get_add_sep(vocab))
                t.push_back(llama_vocab_sep(vocab));
            t.insert(t.end(), doc_toks.begin(), doc_toks.end());
            if (llama_vocab_get_add_eos(vocab) && eos != LLAMA_TOKEN_NULL)
                t.push_back(eos);
        }

        if (toks[i].empty()) {
            throw Error(HELIX_E_VALIDATION,
                        "documents[" + std::to_string(i) +
                        "] produced no tokens",
                        "invalid_request_error", "documents",
                        "helix_e_validation");
        }
        if (toks[i].size() > n_batch) {
            /* No silent truncation — same guard as the embeddings path. */
            throw Error(HELIX_E_CONTEXT_FULL,
                        "query + documents[" + std::to_string(i) + "] is " +
                        std::to_string(toks[i].size()) + " tokens; the "
                        "session batch size is " + std::to_string(n_batch) +
                        ". Shorten the pair or raise n_batch/n_ctx in "
                        "session options",
                        "context_length_exceeded", "documents",
                        "helix_e_context_full");
        }
        lengths[i] = toks[i].size();
        total_tokens += static_cast<int32_t>(toks[i].size());
    }

    /* 4. Pack → flush → collect the rank-head score of each pair.
     * With rank pooling the pooled "embedding" is the classifier output;
     * scores are its first component (raw logit, matching llama-server). */
    const auto flushes = plan_embed_flushes(lengths, n_batch, n_seq_max);

    std::vector<float> scores(req.documents.size(), 0.0f);

    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch),
                                         /*embd=*/0,
                                         static_cast<int32_t>(n_seq_max));
    BatchGuard guard(batch);

    for (const auto& group : flushes) {
        if (sess.cancel_flag().load(std::memory_order_relaxed)) {
            throw Error(HELIX_E_CANCELLED, "request cancelled",
                        "request_cancelled", "", "helix_e_cancelled");
        }

        common_batch_clear(batch);
        for (size_t s = 0; s < group.size(); ++s) {
            const auto& t = toks[group[s]];
            for (size_t p = 0; p < t.size(); ++p) {
                common_batch_add(batch, t[p], static_cast<llama_pos>(p),
                                 {static_cast<llama_seq_id>(s)}, true);
            }
        }

        const int32_t rc = run_one_batch(sess, batch);
        if (rc == 2) {
            throw Error(HELIX_E_CANCELLED, "request cancelled",
                        "request_cancelled", "", "helix_e_cancelled");
        }
        if (rc != 0) {
            throw Error(HELIX_E_BACKEND,
                        "llama_decode failed for rerank batch (rc=" +
                        std::to_string(rc) + ")",
                        "backend_error", "", "helix_e_backend");
        }

        for (size_t s = 0; s < group.size(); ++s) {
            const float* emb = llama_get_embeddings_seq(
                ctx, static_cast<llama_seq_id>(s));
            if (!emb) {
                throw Error(HELIX_E_INTERNAL,
                            "null rank-head output (pooling misconfigured)",
                            "internal_error", "", "helix_e_internal");
            }
            scores[group[s]] = emb[0];
        }
    }

    /* 5. Sort descending by score; stable so equal scores keep input order
     * (determinism across runs and platforms). */
    std::vector<size_t> order(req.documents.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&scores](size_t a, size_t b) {
                         return scores[a] > scores[b];
                     });
    const size_t n_results =
        req.top_n > 0 ? std::min<size_t>(static_cast<size_t>(req.top_n),
                                         order.size())
                      : order.size();

    /* 6. Serialize. */
    nlohmann::ordered_json results = nlohmann::ordered_json::array();
    for (size_t r = 0; r < n_results; ++r) {
        results.push_back({
            {"index",           order[r]},
            {"relevance_score", scores[order[r]]},
        });
    }
    const nlohmann::ordered_json out = {
        {"object",  "list"},
        {"model",   sess.model().alias()},
        {"results", results},
        {"usage",   {{"prompt_tokens", total_tokens},
                     {"total_tokens",  total_tokens}}},
    };
    const std::string json_str = out.dump();

    if (out_response_json) {
        /* Allocated with C malloc() so the caller frees it via helix_free()
         * — see the pairing note above helix_free in helix_abi.cpp. */
        *out_response_json = static_cast<char*>(malloc(json_str.size() + 1));
        if (!*out_response_json) throw std::bad_alloc();
        std::copy_n(json_str.c_str(), json_str.size() + 1, *out_response_json);
    }

    return HELIX_OK;
}

void run_embed_warmup(Session& sess) {
    llama_context*     ctx   = sess.ctx();
    const llama_model* model = sess.model().llama_model_ptr();
    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_token tok = llama_vocab_bos(vocab);
    if (tok == LLAMA_TOKEN_NULL) tok = llama_vocab_eos(vocab);
    if (tok == LLAMA_TOKEN_NULL) tok = 0;

    llama_batch batch = llama_batch_init(1, /*embd=*/0, 1);
    BatchGuard guard(batch);
    common_batch_add(batch, tok, 0, {0}, true);

    const int32_t rc = run_one_batch(sess, batch);
    /* Clear KV so the warmup doesn't pollute the first real request. */
    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);

    if (rc != 0) {
        log_warn("embedding warmup failed (rc=" + std::to_string(rc) +
                 ") — skipping");
        return;
    }
    log_debug("embedding warmup: 1 token pass completed");
}

} // namespace helix
