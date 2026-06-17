import { threads } from 'wasm-feature-detect';

// Solve worker for the visualizer: runs `trace_scenario` off the render thread
// and posts the postcard bytes back. It loads the same viz module the page does
// (multi-thread when the page is cross-origin isolated) and, in that case, brings
// up the rayon Worker pool so the solve is parallel.
let viz;
const ready = (async () => {
  // Two static imports (not a dynamic expression) so webpack bundles both.
  if (await threads()) {
    viz = await import('./pkg-parallel/beamer_viz.js');
    await viz.default();
    await viz.initThreadPool(navigator.hardwareConcurrency);
  } else {
    viz = await import('./pkg/beamer_viz.js');
    await viz.default();
  }
})();

self.onmessage = async ({ data }) => {
  const { id, text, algo } = data;
  try {
    await ready;
    const bytes = viz.trace_scenario(text, algo); // Uint8Array (postcard)
    self.postMessage({ id, bytes }, [bytes.buffer]);
  } catch (e) {
    self.postMessage({ id, error: String((e && e.message) || e) });
  }
};
