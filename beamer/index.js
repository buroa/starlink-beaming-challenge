import { threads } from 'wasm-feature-detect';

const CASES = [
  '00_example', '01_simplest_possible', '02_two_users', '03_five_users',
  '04_one_interferer', '05_equatorial_plane', '06_partially_fullfillable',
  '07_eighteen_planes', '08_eighteen_planes_northern', '09_ten_thousand_users',
  '10_ten_thousand_users_geo_belt', '11_one_hundred_thousand_users'
];

const $ = (id) => document.getElementById(id);
const sel = $('scenario');
for (const c of CASES) {
  const o = document.createElement('option');
  o.value = c;
  o.textContent = c.replace(/_/g, ' ');
  sel.appendChild(o);
}
sel.value = '09_ten_thousand_users';

let wasm, threaded = false;

// Feature-detect WASM threads: load the multi-thread build (pkg-parallel) and
// bring up the rayon Worker pool, else fall back to the single-thread build (pkg).
(async function init() {
  if (await threads()) {
    wasm = await import('./pkg-parallel/beamer.js');
    await wasm.default();
    await wasm.initThreadPool(navigator.hardwareConcurrency);
    threaded = true;
  } else {
    wasm = await import('./pkg/beamer.js');
    await wasm.default();
  }
  $('status').innerHTML = `<span class="ok">ready</span> · ${threaded ? `threaded · ${navigator.hardwareConcurrency} workers` : 'serial'}`;
  $('solve').disabled = false;
})().catch((e) => { $('status').innerHTML = `<span class="err">init failed:</span> ${e.message || e}`; });

$('solve').onclick = async () => {
  $('solve').disabled = true;
  try {
    const name = sel.value + '.txt';
    $('status').textContent = `fetching ${name}…`;
    const text = await (await fetch(`test_cases/${name}`)).text();
    $('status').textContent = 'solving…';
    const t0 = performance.now();
    const solution = wasm.solve_scenario(text, $('intense').checked);
    const ms = performance.now() - t0;
    const head = solution.split('\n').slice(0, 6).join('\n');
    const lines = solution.split('\n').length;
    $('out').textContent = `${head}\n…\n(${lines.toLocaleString()} lines · ${ms.toFixed(0)} ms · ${threaded ? 'threaded' : 'serial'})`;
    $('out').hidden = false;
    $('status').innerHTML = `<span class="ok">done</span> in ${ms.toFixed(0)} ms`;
  } catch (e) {
    $('status').innerHTML = `<span class="err">error:</span> ${e.message || e}`;
  } finally {
    $('solve').disabled = false;
  }
};
