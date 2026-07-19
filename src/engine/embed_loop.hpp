#pragma once
#include "helix.h"

namespace helix {

class Session;

/* OpenAI-shape embeddings request → response (HELIX-IMPL-001 §6).
 * Called from Session::embeddings with the session mutex held.
 * Throws helix::Error; translated at the ABI boundary. */
helix_status_t run_embeddings(Session& sess,
                              const char* request_json,
                              char** out_response_json);

/* Cohere/Jina-shape rerank request → response (1.5). Called from
 * Session::rerank with the session mutex held; the session must be in
 * rerank mode (embedding + rank pooling).
 * Throws helix::Error; translated at the ABI boundary. */
helix_status_t run_rerank(Session& sess,
                          const char* request_json,
                          char** out_response_json);

/* One throwaway single-token embedding pass — same intent as the chat
 * warmup (backend graph compilation, weight paging), implemented in the
 * embedding path so the compiled graph matches real traffic.
 * Best-effort: failures are logged, never thrown. */
void run_embed_warmup(Session& sess);

} // namespace helix
