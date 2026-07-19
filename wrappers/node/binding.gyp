{
  "targets": [{
    "target_name": "helix_node",
    "sources": ["src/binding.cc"],
    "include_dirs": [
      "<!@(node -p \"require('node-addon-api').include\")",
      "<(HELIX_INCLUDE_DIR)"
    ],
    "libraries": [
      "-L<(HELIX_LIB_DIR)",
      "-lhelix",
      "-Wl,-rpath,<(HELIX_LIB_DIR)"
    ],
    "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
    "cflags!": ["-fno-exceptions"],
    "cflags_cc!": ["-fno-exceptions"],
    "cflags_cc": ["-std=c++17"],
    "conditions": [
      ["OS=='linux'", {
        "libraries+": ["-Wl,-rpath,'$$ORIGIN/../lib'"]
      }],
      ["OS=='mac'", {
        "xcode_settings": {
          "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
          "OTHER_LDFLAGS": ["-Wl,-rpath,@loader_path/../lib"]
        }
      }]
    ],
    "variables": {
      "HELIX_INCLUDE_DIR%": "<!(echo ${HELIX_INCLUDE_DIR:-../../../../})",
      "HELIX_LIB_DIR%": "<!(echo ${HELIX_LIB_DIR:-../../../../build})"
    }
  }]
}
