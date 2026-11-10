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
#  llama.cpp commit SHA for the version string
# ------------------------------------------------------------------
# The version string must name the llama.cpp that is actually compiled, so
# the source of truth is the submodule's checked-out HEAD. The marker file
# third_party/llama.cpp.commit is the fallback for trees with no git metadata
# (source tarballs, packaging). Until 2026-11 the marker was the only source
# and had gone stale: it named the first fork patch while the submodule sat
# two commits later, so helix_version_string() reported a llama.cpp that was
# not the one built. CI's llama-pin job keeps the marker equal to the gitlink.
set(_helix_marker "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp.commit")
file(READ "${_helix_marker}" _helix_marker_raw)
string(STRIP "${_helix_marker_raw}" _helix_marker_sha)
set(HELIX_LLAMACPP_SHA "${_helix_marker_sha}")

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
                rev-parse HEAD --absolute-git-dir
        OUTPUT_VARIABLE _helix_git_out
        RESULT_VARIABLE _helix_git_rc
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_helix_git_rc EQUAL 0)
        string(REPLACE "\n" ";" _helix_git_out "${_helix_git_out}")
        list(GET _helix_git_out 0 _helix_head_sha)
        list(GET _helix_git_out 1 _helix_head_gitdir)
        # Inside a plain (non-submodule) helix checkout without llama.cpp
        # populated, rev-parse answers for the *parent* repo; only trust it
        # when the git dir is the submodule's own.
        if(_helix_head_gitdir MATCHES "llama\\.cpp$")
            set(HELIX_LLAMACPP_SHA "${_helix_head_sha}")
            # Re-configure when the submodule moves, or the string goes stale
            # in an existing build tree.
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                         "${_helix_head_gitdir}/HEAD")
            if(NOT _helix_head_sha STREQUAL _helix_marker_sha)
                message(WARNING
                    "third_party/llama.cpp is at ${_helix_head_sha} but "
                    "third_party/llama.cpp.commit says ${_helix_marker_sha}. "
                    "The version string uses the checkout; update the marker "
                    "before committing a pin change.")
            endif()
        endif()
    endif()
endif()
string(SUBSTRING "${HELIX_LLAMACPP_SHA}" 0 12 HELIX_LLAMACPP_SHA_SHORT)
message(STATUS "helix: llama.cpp ${HELIX_LLAMACPP_SHA_SHORT}")

# ------------------------------------------------------------------
#  Platform-specific CPU probe source
# ------------------------------------------------------------------
# cpu_linux.cpp carries the /proc-based Linux probe plus the portable
# std::thread fallback used on every other non-Windows platform (macOS).
if(WIN32)
    set(HELIX_CPU_PROBE_SRC src/hardware/cpu_windows.cpp)
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
#  Shared build settings for the shared and static library targets
# ------------------------------------------------------------------
# Include directories for library sources (tests/ reuses this list).
set(HELIX_PRIVATE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
)

# Backend -> HELIX_HAS_* mapping, spelled out once. cuda_lock.hpp keys the
# per-model decode lock on HELIX_HAS_CUDA || HELIX_HAS_ROCM, so every target
# compiling the library sources must receive the same set.
string(TOUPPER "${HELIX_BACKEND}" _HELIX_BACKEND_UPPER)
set(HELIX_BACKEND_DEFINITIONS HELIX_BACKEND_${_HELIX_BACKEND_UPPER}=1)
foreach(_backend cuda metal vulkan rocm)
    if(HELIX_BACKEND STREQUAL _backend OR HELIX_BACKEND STREQUAL "omni")
        string(TOUPPER "${_backend}" _upper)
        list(APPEND HELIX_BACKEND_DEFINITIONS HELIX_HAS_${_upper}=1)
    endif()
endforeach()

function(helix_configure_target tgt)
    target_compile_definitions(${tgt} PRIVATE
        HELIX_LLAMACPP_SHA="${HELIX_LLAMACPP_SHA_SHORT}"
        HELIX_MAX_DECODE_THREADS=${HELIX_MAX_DECODE_THREADS}
        ${HELIX_BACKEND_DEFINITIONS}
    )
    target_include_directories(${tgt} PRIVATE ${HELIX_PRIVATE_INCLUDE_DIRS})
    # llama + common + mtmd are PRIVATE (symbols hidden from consumers).
    target_link_libraries(${tgt} PRIVATE llama llama-common mtmd)
    if(HELIX_BACKEND STREQUAL "cuda" OR HELIX_BACKEND STREQUAL "omni")
        target_link_libraries(${tgt} PRIVATE CUDA::cudart)
    endif()
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor)
    else()
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
    # Windows: keep <windows.h> from defining min/max macros and pull in a
    # leaner header surface (fewer macro collisions, faster compiles).
    if(WIN32)
        target_compile_definitions(${tgt} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
endfunction()

# ------------------------------------------------------------------
#  Shared library target
# ------------------------------------------------------------------
add_library(helix SHARED ${HELIX_SOURCES})
helix_configure_target(helix)
target_compile_definitions(helix PRIVATE HELIX_BUILD_SHARED)

# Hide all symbols by default — only the helix.h exports are visible.
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
helix_configure_target(helix_static)

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
