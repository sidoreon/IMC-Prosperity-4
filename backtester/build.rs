//! On macOS, PyO3 links `libpython*.dylib` but Cargo often omits an rpath to it, so
//! `rust_backtester` fails at launch with "Library not loaded: @rpath/libpython...".
//! Record the interpreter's `sys.base_prefix/lib` as an rpath (same Python as `PYO3_PYTHON`).

use std::process::Command;

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("apple-darwin") {
        return;
    }

    let py = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&py)
        .args(["-c", "import sys; print(sys.base_prefix + '/lib')"])
        .output();

    let Ok(output) = out else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let lib = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if lib.is_empty() || !std::path::Path::new(&lib).is_dir() {
        return;
    }

    // Embed rpath so the installed binary finds libpython without DYLD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
}
