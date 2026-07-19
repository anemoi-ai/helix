# main.cpp parses describe() JSON with nlohmann; c_abi_check.c is a tiny C TU
# that keeps the "helix.h is valid C99" guarantee under compilation.
add_executable(helix-doctor main.cpp c_abi_check.c)
target_link_libraries(helix-doctor PRIVATE helix)
target_include_directories(helix-doctor PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor
)
install(TARGETS helix-doctor RUNTIME DESTINATION bin)
