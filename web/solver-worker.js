// Runs the WASM solver off the main thread.
//
// wasm-bindgen-rayon's parallel solve blocks the *calling* thread while its
// worker pool runs — and the browser main thread is not allowed to block
// (`Atomics.wait` throws there), so a threaded solve invoked on the main thread
// deadlocks. Running it inside this dedicated worker fixes that (the worker may
// block), and as a bonus keeps the UI responsive during the slower serial solve
// too. For the threaded build, initThreadPool here spawns the rayon pool as
// nested workers.
import init, * as wasm from "./pkg/beam_planner.js";

let threaded = false;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      await init();
      if (typeof wasm.initThreadPool === "function") {
        await wasm.initThreadPool(data.threads);
        threaded = true;
      }
      self.postMessage({ type: "ready", threaded, threads: data.threads });
    } else if (data.type === "solve") {
      const t0 = performance.now();
      const solution = wasm.solve_scenario(data.text, data.intense);
      self.postMessage({ type: "result", solution, ms: performance.now() - t0, threaded });
    }
  } catch (e) {
    self.postMessage({ type: "error", phase: data.type, message: String((e && e.message) || e) });
  }
};
