cmake_minimum_required(VERSION 3.22)
project(helix VERSION 1.6.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Collocate the shared library and executables so Windows DLL search resolves
# at runtime (test exes link against helix; gtest_discover_tests runs them at
# build time). Linux/macOS use RPATH and are unaffected.
if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# ------------------------------------------------------------------
#  Build options
# ------------------------------------------------------------------
option(HELIX_BUILD_TOOLS  "Build helix-doctor and helix-shim-server" ON)
option(HELIX_BUILD_TESTS  "Build test suite" ON)

# Upper bound on auto-selected decode threads. Default 32 (coordination
# overhead dominates beyond this on most CPUs); raise for many-core servers.
set(HELIX_MAX_DECODE_THREADS "32" CACHE STRING
    "Maximum number of auto-selected decode threads")

set(HELIX_BACKEND "cpu" CACHE STRING
    "GPU backend: cpu | cuda | metal | vulkan | rocm | omni")
set_property(CACHE HELIX_BACKEND PROPERTY STRINGS cpu cuda metal vulkan rocm omni)

# ------------------------------------------------------------------
#  Backend flags → GGML switches (must be set before add_subdirectory)
# ------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(HELIX_BACKEND STREQUAL "metal" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(HELIX_BACKEND STREQUAL "vulkan" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(HELIX_BACKEND STREQUAL "rocm" OR HELIX_BACKEND STREQUAL "omni")
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

# ------------------------------------------------------------------
#  llama.cpp vendored build
# ------------------------------------------------------------------
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)  # needed for the mtmd library
set(MTMD_VIDEO           OFF CACHE BOOL "" FORCE)  # Helix does not expose video input (avoids ffmpeg dep)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
#  Read pinned commit SHA for version string
# ------------------------------------------------------------------
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp.commit"
     HELIX_LLAMACPP_SHA_RAW)
string(STRIP "${HELIX_LLAMACPP_SHA_RAW}" HELIX_LLAMACPP_SHA)
string(SUBSTRING "${HELIX_LLAMACPP_SHA}" 0 12 HELIX_LLAMACPP_SHA_SHORT)

# ------------------------------------------------------------------
#  Platform-specific CPU probe source
# ------------------------------------------------------------------
if(WIN32)
    set(HELIX_CPU_PROBE_SRC src/hardware/cpu_windows.cpp)
elseif(APPLE)
    set(HELIX_CPU_PROBE_SRC src/hardware/cpu_macos.cpp)   # added in Phase 4
else()
    set(HELIX_CPU_PROBE_SRC src/hardware/cpu_linux.cpp)
endif()

# ------------------------------------------------------------------
#  Source files
# ------------------------------------------------------------------
set(HELIX_SOURCES
    src/abi/helix_abi.cpp
    src/internal/error.cpp
    src/internal/log.cpp
    src/runtime/runtime.cpp
    src/model/model.cpp
    src/model/cuda_lock.cpp
    src/session/session.cpp
    src/session/options.cpp
    src/session/state.cpp
    src/chat/template.cpp
    src/sampling/chain.cpp
    src/engine/decode_loop.cpp
    src/engine/embed_loop.cpp
    src/engine/warmup.cpp
    src/json/request.cpp
    src/json/response.cpp
    src/hardware/hardware.cpp
    ${HELIX_CPU_PROBE_SRC}
    src/hardware/isa.cpp
    src/hardware/ram.cpp
    src/hardware/mem_bw_table.cpp
    src/hardware/profile.cpp
    src/hardware/accelerators.cpp
    src/hardware/gpu_layer_select.cpp
    src/grammar/response_format.cpp
    src/sampling/logit_bias.cpp
    src/engine/logprobs.cpp
    src/multimodal/image_decode.cpp
    src/multimodal/mmproj.cpp
    src/chat/reasoning.cpp
)

if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    find_package(CUDAToolkit REQUIRED)
    list(APPEND HELIX_SOURCES src/hardware/accel_cuda.cpp)
endif()

# ------------------------------------------------------------------
#  Shared library target
# ------------------------------------------------------------------
add_library(helix SHARED ${HELIX_SOURCES})

string(TOUPPER "${HELIX_BACKEND}" _HELIX_BACKEND_UPPER)
target_compile_definitions(helix PRIVATE
    HELIX_BUILD_SHARED
    HELIX_LLAMACPP_SHA="${HELIX_LLAMACPP_SHA_SHORT}"
    HELIX_BACKEND_${_HELIX_BACKEND_UPPER}=1
    HELIX_MAX_DECODE_THREADS=${HELIX_MAX_DECODE_THREADS}
)
if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    target_compile_definitions(helix PRIVATE HELIX_HAS_CUDA=1)
endif()
if(HELIX_BACKEND STREQUAL "metal" OR HELIX_BACKEND STREQUAL "omni")
    target_compile_definitions(helix PRIVATE HELIX_HAS_METAL=1)
endif()
if(HELIX_BACKEND STREQUAL "vulkan" OR HELIX_BACKEND STREQUAL "omni")
    target_compile_definitions(helix PRIVATE HELIX_HAS_VULKAN=1)
endif()
if(HELIX_BACKEND STREQUAL "rocm" OR HELIX_BACKEND STREQUAL "omni")
    target_compile_definitions(helix PRIVATE HELIX_HAS_ROCM=1)
endif()

target_include_directories(helix PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
)

# Hide all symbols by default — only the 16 exported ones are visible.
set_target_properties(helix PROPERTIES
    C_VISIBILITY_PRESET   hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION   ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME helix
)

# macOS: set compatibility/current version for dylib versioning.
# compatibility_version must not be bumped during the 1.x ABI commitment.
if(APPLE)
    set_target_properties(helix PROPERTIES
        MACOSX_RPATH ON
        INSTALL_NAME_DIR "@rpath"
        BUILD_WITH_INSTALL_RPATH ON
    )
    # Encode compatibility=1.0.0, current=project_version in the Mach-O header.
    target_link_options(helix PRIVATE
        "-Wl,-compatibility_version,1.0.0"
        "-Wl,-current_version,${PROJECT_VERSION}"
    )
    option(HELIX_UNIVERSAL_BINARY "Build arm64+x86_64 universal helix.dylib" OFF)
    if(HELIX_UNIVERSAL_BINARY)
        set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
    endif()
endif()

# Link llama + common + mtmd as PRIVATE (symbols hidden from consumers).
target_link_libraries(helix PRIVATE llama llama-common mtmd)
if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    target_link_libraries(helix PRIVATE CUDA::cudart)
endif()

# Platform-specific symbol export control.
if(UNIX AND NOT APPLE)
    target_link_options(helix PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/helix.ld"
    )
elseif(APPLE)
    target_link_options(helix PRIVATE
        "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/helix.exp"
    )
elseif(WIN32)
    # .def file is specified via MODULE_DEFINITION_FILE property
    set_target_properties(helix PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )
    target_sources(helix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helix.def)
endif()

# ------------------------------------------------------------------
#  Static library target (CPU variant only, for size-sensitive use)
# ------------------------------------------------------------------
add_library(helix_static STATIC ${HELIX_SOURCES})
# On Unix the .a/.so extensions differ so libhelix.a is unambiguous; on
# Windows the static archive would collide with the shared library's import
# library (both helix.lib), so keep the _static suffix there.
if(NOT WIN32)
    set_target_properties(helix_static PROPERTIES OUTPUT_NAME helix)
endif()
target_compile_definitions(helix_static PRIVATE
    HELIX_LLAMACPP_SHA="${HELIX_LLAMACPP_SHA_SHORT}"
    HELIX_BACKEND_${_HELIX_BACKEND_UPPER}=1
    HELIX_MAX_DECODE_THREADS=${HELIX_MAX_DECODE_THREADS}
)
if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    target_compile_definitions(helix_static PRIVATE HELIX_HAS_CUDA=1)
endif()
target_include_directories(helix_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
)
target_link_libraries(helix_static PRIVATE llama llama-common mtmd)
if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
    target_link_libraries(helix_static PRIVATE CUDA::cudart)
endif()

# Compiler warnings (after both shared and static targets exist).
if(MSVC)
    target_compile_options(helix        PRIVATE /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor)
    target_compile_options(helix_static PRIVATE /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor)
else()
    target_compile_options(helix        PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_compile_options(helix_static PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Windows: keep <windows.h> from defining min/max macros and pull in a leaner
# header surface (fewer macro collisions, faster compiles).
if(WIN32)
    target_compile_definitions(helix        PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_definitions(helix_static PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ------------------------------------------------------------------
#  Install rules
# ------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Only the SHARED target goes in the exported target set. helix_static links
# llama/llama-common/mtmd PRIVATELY; because a static archive does not embed its
# dependencies, CMake treats those as interface link requirements that an export
# set must also resolve. The vendored llama.cpp targets are EXCLUDE_FROM_ALL and
# are intentionally not installed, so they cannot be exported — and an exported
# helix_static referencing them would fail at the consumer's configure step
# anyway. We therefore install the static archive as a plain artifact (for
# size-sensitive consumers who link it manually alongside their own llama.cpp)
# but keep it out of helixTargets. find_package(helix) exposes helix::helix.
install(TARGETS helix
    EXPORT helixTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS helix_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES helix.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# CMake package config
set(HELIX_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/helix")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/helix-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/helix-config.cmake"
    INSTALL_DESTINATION "${HELIX_CMAKE_INSTALL_DIR}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/helix-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT helixTargets
    FILE helix-targets.cmake
    NAMESPACE helix::
    DESTINATION "${HELIX_CMAKE_INSTALL_DIR}"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/helix-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/helix-config-version.cmake"
    DESTINATION "${HELIX_CMAKE_INSTALL_DIR}"
)

# ------------------------------------------------------------------
#  pkg-config
# ------------------------------------------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/helix.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/helix.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/helix.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)

# ------------------------------------------------------------------
#  Tools
# ------------------------------------------------------------------
if(HELIX_BUILD_TOOLS)
    add_subdirectory(tools/helix-doctor)
    add_subdirectory(tools/helix-shim-server)
endif()

# ------------------------------------------------------------------
#  Tests
# ------------------------------------------------------------------
if(HELIX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
