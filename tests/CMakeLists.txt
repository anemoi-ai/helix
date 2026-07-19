include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)
# Avoid installing gtest alongside helix.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Common include path for all tests.
set(HELIX_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
    ${CMAKE_SOURCE_DIR}/src
)

# Platform-specific CPU probe source (mirrors the library build).
if(WIN32)
    set(HELIX_CPU_PROBE_SRC ${CMAKE_SOURCE_DIR}/src/hardware/cpu_windows.cpp)
elseif(APPLE)
    set(HELIX_CPU_PROBE_SRC ${CMAKE_SOURCE_DIR}/src/hardware/cpu_macos.cpp)
else()
    set(HELIX_CPU_PROBE_SRC ${CMAKE_SOURCE_DIR}/src/hardware/cpu_linux.cpp)
endif()

# ------------------------------------------------------------------
#  Unit tests (no model required)
#
#  Link directly against source files so we can access internal C++
#  symbols that are hidden in the shared library.
# ------------------------------------------------------------------
add_executable(helix_unit_tests
    unit/helix_test_main.cpp
    unit/request_parse_test.cpp
    unit/response_serialize_test.cpp
    unit/error_envelope_test.cpp
    unit/stop_strings_test.cpp
    unit/sampler_chain_test.cpp
    unit/utf8_prefix_test.cpp
    unit/event_sink_test.cpp
    unit/lcp_test.cpp
    unit/tools_request_test.cpp
    unit/tools_response_test.cpp
    unit/hardware_probe_test.cpp
    unit/hardware_selector_test.cpp
    unit/accelerator_probe_test.cpp
    unit/gpu_layer_select_test.cpp
    unit/cuda_lock_test.cpp
    unit/response_format_test.cpp
    unit/logit_bias_test.cpp
    unit/logprobs_format_test.cpp
    unit/conflict_validation_test.cpp
    unit/content_part_parse_test.cpp
    unit/data_uri_test.cpp
    unit/reasoning_extract_test.cpp
    unit/embeddings_request_test.cpp
    unit/rerank_request_test.cpp
    unit/embed_packing_test.cpp
    unit/speculative_options_test.cpp
    # Internal sources needed for symbol access:
    ${CMAKE_SOURCE_DIR}/src/session/options.cpp
    ${CMAKE_SOURCE_DIR}/src/json/request.cpp
    ${CMAKE_SOURCE_DIR}/src/json/response.cpp
    ${CMAKE_SOURCE_DIR}/src/internal/error.cpp
    ${CMAKE_SOURCE_DIR}/src/internal/log.cpp
    ${CMAKE_SOURCE_DIR}/src/sampling/chain.cpp
    ${CMAKE_SOURCE_DIR}/src/hardware/hardware.cpp
    ${HELIX_CPU_PROBE_SRC}
    ${CMAKE_SOURCE_DIR}/src/hardware/isa.cpp
    ${CMAKE_SOURCE_DIR}/src/hardware/ram.cpp
    ${CMAKE_SOURCE_DIR}/src/hardware/mem_bw_table.cpp
    ${CMAKE_SOURCE_DIR}/src/hardware/profile.cpp
    ${CMAKE_SOURCE_DIR}/src/hardware/accelerators.cpp
    ${CMAKE_SOURCE_DIR}/src/hardware/gpu_layer_select.cpp
    ${CMAKE_SOURCE_DIR}/src/model/cuda_lock.cpp
    ${CMAKE_SOURCE_DIR}/src/grammar/response_format.cpp
    ${CMAKE_SOURCE_DIR}/src/sampling/logit_bias.cpp
    ${CMAKE_SOURCE_DIR}/src/multimodal/image_decode.cpp
    ${CMAKE_SOURCE_DIR}/src/chat/reasoning.cpp
)

target_include_directories(helix_unit_tests PRIVATE ${HELIX_TEST_INCLUDES})
target_link_libraries(helix_unit_tests PRIVATE llama llama-common mtmd GTest::gtest)
target_compile_definitions(helix_unit_tests PRIVATE
    HELIX_BACKEND_CPU=1
)
target_compile_definitions(helix_unit_tests PRIVATE
    HELIX_LLAMACPP_SHA="test"
)
if(WIN32)
    target_compile_definitions(helix_unit_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

gtest_discover_tests(helix_unit_tests)

# ------------------------------------------------------------------
#  Integration tests (require a model; set HELIX_TEST_MODEL env var)
# ------------------------------------------------------------------
add_executable(helix_integration_tests
    integration/integration_test.cpp
)

target_include_directories(helix_integration_tests PRIVATE ${HELIX_TEST_INCLUDES})
target_link_libraries(helix_integration_tests PRIVATE helix GTest::gtest_main)

# Only registered with ctest when a model is provided at configure time;
# otherwise run the binary directly with HELIX_TEST_MODEL etc. set in the
# environment (what CI does).
if(HELIX_TEST_MODEL)
    gtest_discover_tests(helix_integration_tests
        PROPERTIES ENVIRONMENT
            "HELIX_TEST_MODEL=${HELIX_TEST_MODEL}\;HELIX_TEST_EMBED_MODEL=${HELIX_TEST_EMBED_MODEL}\;HELIX_TEST_RERANK_MODEL=${HELIX_TEST_RERANK_MODEL}"
    )
endif()
