// Solves a scenario off the render thread, so heavy cases don't freeze the
// canvas. Imports the SAME viz wasm and runs the (serial) solver, returning
// (Scenario, Feasibility, Trace) postcard bytes. Driven by window.solveText in
// beamer.html, which the viz wasm calls via the `solveText` import.
import init, { trace_scenario } from "./viz-pkg/beam_planner.js";

const ready = init();

self.onmessage = async ({ data }) => {
  const { id, text, algo } = data;
  try {
    await ready;
    const bytes = trace_scenario(text, algo); // Uint8Array (Vec<u8>)
    self.postMessage({ id, bytes }, [bytes.buffer]); // transfer, no copy
  } catch (e) {
    self.postMessage({ id, error: String((e && e.message) || e) });
  }
};
