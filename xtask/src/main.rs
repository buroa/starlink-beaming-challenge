//! Build orchestration for the WASM assets and the GitHub Pages site — the Rust
//! replacement for the old `web/build.sh` + `web/dist.sh` shell scripts.
//!
//!   cargo xtask viz      build the visualizer render module   → web/viz-pkg
//!   cargo xtask solver   build the threaded solver module     → web/pkg
//!   cargo xtask dist     build both + assemble the static site → web/dist
//!
//! It shells out to `cargo` for the two wasm builds (resolving the right rustup
//! toolchain explicitly, so it works even when another cargo shadows rustup on
//! PATH), runs wasm-bindgen via its library API (so there's no separate
//! `wasm-bindgen` CLI to install), and does the bindgen patch + file assembly in
//! plain Rust.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};
use wasm_bindgen_cli_support::Bindgen;

const WASM: &str = "target/wasm32-unknown-unknown/release/beam_planner.wasm";

// The threaded build: this rustc/wasm-ld doesn't derive the threading setup from
// +atomics alone, so request it explicitly — an imported, shared, bounded memory
// (what wasm-bindgen's thread transform expects) plus the __heap_base / TLS
// globals it injects into.
const THREAD_RUSTFLAGS: &str = "\
    -C target-feature=+atomics,+bulk-memory,+mutable-globals \
    -C link-arg=--shared-memory -C link-arg=--import-memory \
    -C link-arg=--max-memory=2147483648 \
    -C link-arg=--export=__heap_base \
    -C link-arg=--export-if-defined=__wasm_init_tls \
    -C link-arg=--export-if-defined=__tls_size \
    -C link-arg=--export-if-defined=__tls_align \
    -C link-arg=--export-if-defined=__tls_base";

fn main() {
    let cmd = env::args().nth(1).unwrap_or_default();
    let result = match cmd.as_str() {
        "viz" => build_viz(),
        "solver" => build_solver(),
        "dist" => dist(),
        other => {
            eprintln!("usage: cargo xtask <viz|solver|dist>  (got {other:?})");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("xtask: {e:#}");
        std::process::exit(1);
    }
}

/// Repo root — the parent of this crate's manifest dir (robust to the cwd).
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent dir")
        .to_path_buf()
}

/// The `bin` directory of a rustup toolchain. Prepended to a child's PATH so its
/// `cargo` AND `rustc` resolve to that toolchain even if another shadows rustup.
fn toolchain_bin(toolchain: &str) -> Result<PathBuf> {
    let out = Command::new("rustup")
        .args(["which", "--toolchain", toolchain, "cargo"])
        .output()
        .context("running `rustup which` (is rustup installed?)")?;
    if !out.status.success() {
        bail!("`rustup which --toolchain {toolchain} cargo` failed — is the {toolchain} toolchain installed?");
    }
    let cargo = PathBuf::from(String::from_utf8(out.stdout)?.trim());
    Ok(cargo.parent().context("toolchain cargo has no parent")?.to_path_buf())
}

/// `cargo build` the library for wasm with a toolchain, feature set, extra args,
/// and optional RUSTFLAGS.
fn cargo_wasm(toolchain: &str, features: &str, extra: &[&str], rustflags: Option<&str>) -> Result<()> {
    let bin = toolchain_bin(toolchain)?;
    let path = match env::var_os("PATH") {
        Some(p) => format!("{}:{}", bin.display(), p.to_string_lossy()),
        None => bin.display().to_string(),
    };
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root())
        .env("PATH", path)
        .args([
            "build", "--release", "--lib", "--no-default-features",
            "--features", features, "--target", "wasm32-unknown-unknown",
        ])
        .args(extra);
    if let Some(rf) = rustflags {
        cmd.env("RUSTFLAGS", rf);
    }
    let status = cmd.status().context("spawning cargo")?;
    if !status.success() {
        bail!("cargo build failed (toolchain={toolchain}, features={features})");
    }
    Ok(())
}

/// Run wasm-bindgen (`--target web`) on the freshly built wasm into `out_dir`,
/// which is cleared first to avoid stale cross-build artifacts.
fn bindgen(out_dir: &Path) -> Result<()> {
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;
    Bindgen::new()
        .input_path(root().join(WASM))
        .web(true)?
        .out_name("beam_planner")
        .generate(out_dir)
        .context("wasm-bindgen generate")?;
    Ok(())
}

fn build_viz() -> Result<()> {
    eprintln!(">> visualizer (stable; eframe + wgpu WebGL2)");
    cargo_wasm("stable", "viz", &[], None)?;
    bindgen(&root().join("web/viz-pkg"))?;
    eprintln!(">> web/viz-pkg");
    Ok(())
}

fn build_solver() -> Result<()> {
    eprintln!(">> solver, threaded (nightly + build-std + shared memory)");
    // RUSTFLAGS is whitespace-split; the const's line continuations already
    // collapse it to a single spaced string.
    cargo_wasm(
        "nightly",
        "parallel,wire",
        &["-Z", "build-std=std,panic_abort"],
        Some(THREAD_RUSTFLAGS),
    )?;
    let out = root().join("web/pkg");
    bindgen(&out)?;
    patch_worker_import(&out)?;
    eprintln!(">> web/pkg");
    Ok(())
}

/// wasm-bindgen-rayon's worker helper imports the main module as a bare directory
/// (`import('../../..')`), which only resolves through a bundler — on a plain
/// static server it 404s and the pool never starts. Rewrite it to the explicit
/// module file.
fn patch_worker_import(pkg: &Path) -> Result<()> {
    let wh = find_file(&pkg.join("snippets"), "workerHelpers.js")?
        .context("workerHelpers.js not found in the threaded build")?;
    let patched = fs::read_to_string(&wh)?
        .replace("import('../../..')", "import('../../../beam_planner.js')");
    fs::write(&wh, patched)?;
    eprintln!(">> patched worker import → ../../../beam_planner.js ({})", wh.display());
    Ok(())
}

fn dist() -> Result<()> {
    build_viz()?;
    build_solver()?;

    let web = root().join("web");
    let dist = web.join("dist");
    eprintln!(">> assembling {}", dist.display());
    let _ = fs::remove_dir_all(&dist);
    fs::create_dir_all(dist.join("solver"))?;

    // viz at the site root
    copy(&web.join("index.html"), &dist.join("index.html"))?;
    copy(&web.join("viz-solver-worker.js"), &dist.join("viz-solver-worker.js"))?;
    copy(&web.join("coi-serviceworker.js"), &dist.join("coi-serviceworker.js"))?;
    copy_dir(&web.join("viz-pkg"), &dist.join("viz-pkg"))?;
    copy_dir(&web.join("pkg"), &dist.join("pkg"))?;
    copy_dir(&root().join("test_cases"), &dist.join("test_cases"))?;

    // solver at /solver (fetches ../test_cases → the shared root copy)
    copy(&web.join("solver.html"), &dist.join("solver/index.html"))?;
    copy(&web.join("solver-worker.js"), &dist.join("solver/solver-worker.js"))?;
    copy(&web.join("coi-serviceworker.js"), &dist.join("solver/coi-serviceworker.js"))?;
    copy_dir(&web.join("pkg"), &dist.join("solver/pkg"))?;

    // Serve everything verbatim (no Jekyll mangling).
    fs::write(dist.join(".nojekyll"), "")?;

    eprintln!(">> web/dist ready");
    eprintln!("   serve: (cd web/dist && python3 -m http.server)  →  http://localhost:8000/  (and /solver)");
    Ok(())
}

fn copy(from: &Path, to: &Path) -> Result<()> {
    fs::copy(from, to).with_context(|| format!("copy {} → {}", from.display(), to.display()))?;
    Ok(())
}

/// Recursively copy a directory tree.
fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from).with_context(|| format!("read_dir {}", from.display()))? {
        let entry = entry?;
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        if entry.file_type()?.is_dir() {
            copy_dir(&src, &dst)?;
        } else {
            fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

/// First file named `name` anywhere under `dir` (depth-first).
fn find_file(dir: &Path, name: &str) -> Result<Option<PathBuf>> {
    if !dir.exists() {
        return Ok(None);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if entry.file_type()?.is_dir() {
            if let Some(found) = find_file(&p, name)? {
                return Ok(Some(found));
            }
        } else if entry.file_name() == name {
            return Ok(Some(p));
        }
    }
    Ok(None)
}
