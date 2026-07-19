#include <gtest/gtest.h>
#include "llama.h"

// The unit tests link against Helix sources directly (not the shared library),
// so the production runtime never runs and the llama backend is never
// initialised. Tests that exercise pick_n_gpu_layers -> common_fit_params need
// a registered CPU backend, otherwise llama_model_load_from_file touches
// uninitialised backend state. Match the runtime by initialising once here.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    llama_backend_init();
    int result = RUN_ALL_TESTS();
    llama_backend_free();
    return result;
}
