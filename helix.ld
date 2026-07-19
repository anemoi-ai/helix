/* HELIX_1.0 — the 16-symbol stable surface frozen at v1.0.0.
 *
 * Future 1.x additions get their own version node (HELIX_1.1, HELIX_1.2 …)
 * each inheriting from HELIX_1.0, so apps can require the exact level they
 * were built against.  A major-version bump (HELIX_2.0) is forbidden during
 * the 18-month 1.x ABI commitment window.
 */
HELIX_1.0 {
    global:
        helix_abi_version;
        helix_version_string;
        helix_last_error_json;
        helix_runtime_create;
        helix_runtime_destroy;
        helix_runtime_describe;
        helix_model_load;
        helix_model_release;
        helix_model_describe;
        helix_session_create;
        helix_session_destroy;
        helix_chat_completions;
        helix_chat_completions_stream;
        helix_session_cancel;
        helix_free;
        helix_set_log_callback;
    local:
        *;
};

/* v1.1.0 additions — embeddings endpoint. */
HELIX_1.1 {
    global:
        helix_embeddings;
        helix_model_embedding_dim;
} HELIX_1.0;

/* v1.2.0 additions — session describe (MTP telemetry). */
HELIX_1.2 {
    global:
        helix_session_describe;
} HELIX_1.1;

/* v1.3.0 additions — describe-copy, session encode, hidden-dim. */
HELIX_1.3 {
    global:
        helix_runtime_describe_copy;
        helix_model_describe_copy;
        helix_session_encode;
        helix_model_hidden_dim;
} HELIX_1.2;

/* v1.4.0 additions — tokenizer utilities, model-load progress callback. */
HELIX_1.4 {
    global:
        helix_count_tokens;
        helix_tokenize;
        helix_model_load_ex;
} HELIX_1.3;

/* v1.5.0 additions — session state persistence, rerank endpoint. */
HELIX_1.5 {
    global:
        helix_session_save;
        helix_session_restore;
        helix_rerank;
} HELIX_1.4;
