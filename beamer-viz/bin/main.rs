//! Thin native entry point for the GPU visualizer. The logic lives in the
//! `beamer-viz` library crate, which also compiles to a wasm-bindgen `cdylib`
//! (see `src/lib.rs`) for the browser.

// No console window for the release GUI build on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if let Err(e) = beamer_viz::run() {
        eprintln!("beamer-viz: {e}");
        std::process::exit(1);
    }
}
