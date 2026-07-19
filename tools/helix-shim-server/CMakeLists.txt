add_executable(helix-shim-server main.cpp)

target_include_directories(helix-shim-server PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor
)

target_link_libraries(helix-shim-server PRIVATE helix cpp-httplib)

# cpp-httplib needs threads.
find_package(Threads REQUIRED)
target_link_libraries(helix-shim-server PRIVATE Threads::Threads)

# OpenSSL is optional; without it the shim is HTTP-only, which is fine
# for local testing. httplib.h gates the openssl includes on
# #ifdef CPPHTTPLIB_OPENSSL_SUPPORT, so it must be left UNDEFINED (not =0)
# when OpenSSL is absent.
find_package(OpenSSL)
if(OpenSSL_FOUND)
    target_compile_definitions(helix-shim-server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT=1)
    target_link_libraries(helix-shim-server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()

install(TARGETS helix-shim-server RUNTIME DESTINATION bin)
