import { threads } from 'wasm-feature-detect';

const msg = document.getElementById('msg');

// The solve runs off the render thread in a Web Worker (solve.worker.js): the
// 100k solve is tens of seconds, and wasm-bindgen-rayon's parallel solve can't
// run on the browser main thread. `window.solveText` is the import the viz wasm
// calls; define it BEFORE start(). webpack bundles the worker from the URL.
const worker = new Worker(new URL('./solve.worker.js', import.meta.url), { type: 'module' });
let seq = 0;
const pending = new Map();
worker.onmessage = ({ data }) => {
  const p = pending.get(data.id);
  if (!p) return;
  pending.delete(data.id);
  if (data.error) p.reject(new Error(data.error));
  else p.resolve(data.bytes);
};
window.solveText = (text, algo) => new Promise((resolve, reject) => {
  const id = ++seq;
  pending.set(id, { resolve, reject });
  worker.postMessage({ id, text, algo });
});

// Feature-detect threads, load the matching viz module, and mount eframe on the
// canvas. The render thread itself doesn't need the rayon pool (the worker does),
// so no initThreadPool here.
(async function start() {
  try {
    // Two static imports (not a dynamic expression) so webpack bundles both.
    const viz = (await threads())
      ? await import('./pkg-parallel/beamer_viz.js')
      : await import('./pkg/beamer_viz.js');
    await viz.default();
    await viz.start(document.getElementById('canvas'));
    msg.remove(); // eframe owns the canvas now
  } catch (e) {
    msg.style.color = '#ff7b7b';
    msg.textContent = 'Beamer failed to start: ' + ((e && e.message) || e);
    console.error(e);
  }
})();
