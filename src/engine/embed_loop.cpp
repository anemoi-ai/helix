#include "embed_loop.hpp"
#include "embed_packing.hpp"
#include "../session/session.hpp"
#include "../model/model.hpp"
#include "../json/request.hpp"
#include "../json/response.hpp"
#include "../internal/caller_alloc.hpp"
#include "../internal/error.hpp"
#include "../internal/log.hpp"
#include "../util/json_dump.hpp"

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

/* Fill a "rerank" chat template's {query}/{document} placeholders in one
 * left-to-right pass. Sequential replace-all would re-substitute inside the
 * inserted text (a query containing "{document}" would receive the
 * document). */
std::string format_rerank_prompt(const std::string& tmpl,
                                 const std::string& query,
                                 const std::string& document) {
    static const std::string kQuery    = "{query}";
    static const std::string kDocument = "{document}";
    std::string out;
    out.reserve(tmpl.size() + query.size() + document.size());
    size_t i = 0;
    while (i < tmpl.size()) {
        if (tmpl.compare(i, kQuery.size(), kQuery) == 0) {
            out += query;
            i += kQuery.size();
        } else if (tmpl.compare(i, kDocument.size(), kDocument) == 0) {
            out += document;
            i += kDocument.size();
        } else {
            out += tmpl[i++];
        }
    }
    return out;
}

int32_t run_one_batch(Session& sess, llama_batch& batch) {
    llama_context* ctx = sess.ctx();
    /* Clear all KV before each flush; for the encoder path this is a
     * harmless no-op that keeps the loop uniform. */
    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    return sess.embed_via_encode() ? llama_encode(ctx, batch)
                                   : llama_decode(ctx, batch);
}

/* No silent truncation — mirrors the llama.cpp server guard. `item` names the
 * sequence in the message ("input[3]"), `shorten` what the caller should
 * shorten, `param` the request field. */
void require_fits_batch(const std::vector<llama_token>& t, uint32_t n_batch,
                        const std::string& item, const char* shorten,
                        const char* param) {
    if (t.size() <= n_batch) return;
    throw_context_full(static_cast<int64_t>(t.size()),
                       static_cast<int64_t>(n_batch),
                       item + " is " + std::to_string(t.size()) + " tokens; the "
                       "session batch size is " + std::to_string(n_batch) +
                       ". Shorten the " + shorten + " or raise n_batch/n_ctx in "
                       "session options",
                       param);
}

/* Packs `toks` into flushes, runs each one and hands every sequence's pooled
 * output to on_seq(item_index, pooled). The planner is the
 * determinism-critical seam (insertion-ordered; see embed_packing.hpp).
 * `what` ("embedding" / "rerank") and `output_desc` name the request kind
 * and the pooled output in error messages. */
template <typename OnSeq>
void run_packed_flushes(Session& sess,
                        const std::vector<std::vector<llama_token>>& toks,
                        const char* what, const char* output_desc,
                        OnSeq&& on_seq) {
    llama_context* ctx       = sess.ctx();
    const uint32_t n_batch   = llama_n_batch(ctx);
    const uint32_t n_seq_max = llama_n_seq_max(ctx);

    std::vector<size_t> lengths(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) lengths[i] = toks[i].size();
    const auto flushes = plan_embed_flushes(lengths, n_batch, n_seq_max);

    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch),
                                         /*embd=*/0,
                                         static_cast<int32_t>(n_seq_max));
    BatchGuard guard(batch);

    for (const auto& group : flushes) {
        /* Flush-boundary cancellation, same granularity as the chat path.
         * State is only the KV memory, which the next flush clears, so the
         * session stays reusable after cancellation. */
        if (sess.cancel_flag().load(std::memory_order_relaxed)) {
            throw_cancelled("request cancelled");
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
            /* Abort-callback path. No abort_callback is installed, but map
             * it defensively so a future installation needs no change here. */
            throw_cancelled("request cancelled");
        }
        if (rc != 0) {
            throw_backend(std::string("llama_decode failed for ") + what +
                          " batch (rc=" + std::to_string(rc) + ")");
        }

        for (size_t s = 0; s < group.size(); ++s) {
            const float* pooled = llama_get_embeddings_seq(
                ctx, static_cast<llama_seq_id>(s));
            if (!pooled) {
                throw_internal(std::string("null ") + output_desc +
                               " (pooling misconfigured)");
            }
            on_seq(group[s], pooled);
        }
    }
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
    sess.require_model_alias(req.model);

    const llama_model* model = sess.model().llama_model_ptr();
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const uint32_t n_batch    = llama_n_batch(sess.ctx());
    const int32_t  n_embd_out = llama_model_n_embd_out(model);

    /* 3. Tokenize every input up front — cheap, and it lets the whole
     * request be validated before any compute is spent. */
    std::vector<std::vector<llama_token>> toks(req.inputs.size());
    int32_t total_tokens = 0;
    for (size_t i = 0; i < req.inputs.size(); ++i) {
        toks[i] = common_tokenize(vocab, req.inputs[i],
                                  /*add_special=*/true, /*parse_special=*/true);
        if (toks[i].empty()) {
            throw_validation("input[" + std::to_string(i) + "] produced no tokens", "input");
        }
        require_fits_batch(toks[i], n_batch, "input[" + std::to_string(i) + "]",
                           "input", "input");
        total_tokens += static_cast<int32_t>(toks[i].size());
    }

    warn_if_missing_eos_sep(vocab, toks);

    /* 4. Pack → flush → L2-normalize (euclidean), matching OpenAI's unit
     * vectors. */
    EmbeddingsResponse resp;
    resp.model         = sess.model().alias();
    resp.encoding      = req.encoding;
    resp.prompt_tokens = total_tokens;
    resp.vectors.resize(req.inputs.size());

    run_packed_flushes(sess, toks, "embedding", "sequence embedding",
                       [&](size_t index, const float* pooled) {
        std::vector<float> v(static_cast<size_t>(n_embd_out));
        common_embd_normalize(pooled, v.data(), n_embd_out, /*embd_norm=*/2);
        resp.vectors[index] = std::move(v);
    });

    /* 5. Serialize. */
    if (out_response_json) {
        *out_response_json = dup_for_caller(resp.to_json());
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
    sess.require_model_alias(req.model);

    const llama_model* model = sess.model().llama_model_ptr();
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const uint32_t n_batch = llama_n_batch(sess.ctx());

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
    int32_t total_tokens = 0;
    for (size_t i = 0; i < req.documents.size(); ++i) {
        if (rr_tmpl) {
            const std::string prompt =
                format_rerank_prompt(rr_tmpl, req.query, req.documents[i]);
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
            throw_validation("documents[" + std::to_string(i) +
                             "] produced no tokens", "documents");
        }
        require_fits_batch(toks[i], n_batch,
                           "query + documents[" + std::to_string(i) + "]",
                           "pair", "documents");
        total_tokens += static_cast<int32_t>(toks[i].size());
    }

    /* 4. Pack → flush → collect the rank-head score of each pair.
     * With rank pooling the pooled "embedding" is the classifier output;
     * scores are its first component (raw logit, matching llama-server). */
    std::vector<float> scores(req.documents.size(), 0.0f);
    run_packed_flushes(sess, toks, "rerank", "rank-head output",
                       [&](size_t index, const float* pooled) {
        scores[index] = pooled[0];
    });

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
    if (out_response_json) {
        *out_response_json = dup_for_caller(dump_json(out));
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
