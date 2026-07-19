#pragma once
#include "helix.h"

namespace helix {

class Session;

/* OpenAI-shape chat completions request -> response (streaming or buffered).
 * Called from Session::do_chat_completions with the session mutex held.
 * Throws helix::Error; translated at the ABI boundary. */
helix_status_t run_chat_completions(Session& sess,
                                    const char* request_json,
                                    char** out_response_json,
                                    helix_stream_cb on_chunk,
                                    void* user_data,
                                    bool streaming);

/* Chat request -> prompt_tokens count after template rendering, without
 * touching the session KV cache. Called from Session::count_tokens WITHOUT
 * the session mutex — must stay read-only on session/model state.
 * Throws helix::Error; translated at the ABI boundary. */
helix_status_t run_count_tokens(Session& sess,
                                const char* request_json,
                                uint32_t* out_token_count);

} // namespace helix
