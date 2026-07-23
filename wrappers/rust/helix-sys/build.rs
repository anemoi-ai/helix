use std::env;
use std::path::PathBuf;

fn main() {
    // ── 1. Locate libhelix ─────────────────────────────────────────────────
    let lib_dir = env::var("HELIX_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Walk up from this crate to find the repo's build/ directory
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let repo_root = manifest.ancestors().nth(3).unwrap().to_path_buf();
            repo_root.join("build")
        });

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=helix");
    println!("cargo:rerun-if-env-changed=HELIX_LIB_DIR");

    // ── 2. Locate helix.h ──────────────────────────────────────────────────
    let header_path = env::var("HELIX_HEADER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let repo_root = manifest.ancestors().nth(3).unwrap().to_path_buf();
            repo_root.join("helix.h")
        });

    println!("cargo:rerun-if-changed={}", header_path.display());

    // ── 3. Generate bindings ───────────────────────────────────────────────
    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .allowlist_function("helix_.*")
        .allowlist_type("helix_.*|HELIX_.*")
        .allowlist_var("HELIX_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate helix bindings");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
