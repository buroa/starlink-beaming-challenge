// Solves a scenario off the render thread — in PARALLEL when possible.
//
// Prefers the THREADED solver module (web/pkg, built with `./web/build.sh
// --threaded`): it brings up the wasm-bindgen-rayon pool via initThreadPool —
// the same multi-core solve the solver harness (index.html) uses — so the viz
// gets the ~2.5x speedup on the big cases. Falls back to the serial viz module
// if the threaded build isn't present or the page isn't cross-origin isolated.
// Either way the solve runs here, not on the render thread.

let mode = { threaded: false, workers: 1 };

const loaded = (async () => {
  const workers = navigator.hardwareConcurrency || 4;
  try {
    if (!self.crossOriginIsolated) throw new Error("page is not cross-origin isolated");
    const m = await import("./pkg/beam_planner.js"); // threaded solver module
    await m.default();
    await m.initThreadPool(workers);
    mode = { threaded: true, workers };
    self.postMessage({ type: "ready", ...mode });
    return m;
  } catch (e) {
    mode = { threaded: false, workers: 1, why: String((e && e.message) || e) };
    self.postMessage({ type: "ready", ...mode });
    const m = await import("./viz-pkg/beam_planner.js"); // serial viz module
    await m.default();
    return m;
  }
})();

self.onmessage = async ({ data }) => {
  const { id, text, algo } = data;
  try {
    const m = await loaded;
    const t0 = performance.now();
    const bytes = m.trace_scenario(text, algo); // Uint8Array (Vec<u8>)
    const ms = Math.round(performance.now() - t0);
    self.postMessage({ id, bytes, ms, threaded: mode.threaded, workers: mode.workers }, [bytes.buffer]);
  } catch (e) {
    self.postMessage({ id, error: String((e && e.message) || e) });
  }
};
