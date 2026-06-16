//! Thin native entry point for the GPU visualizer. All the logic lives in the
//! library's [`beam_planner::viz`] module so it can also be compiled to a
//! wasm-bindgen `cdylib` for the browser.

// No console window for the release GUI build on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if let Err(e) = beam_planner::viz::run() {
        eprintln!("beamer: {e}");
        std::process::exit(1);
    }
}
