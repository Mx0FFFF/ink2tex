// Headless proof the browser demo's whole path works: real equation strokes through the
// wasm module, no browser required. Run: bash wasm-demo/build.sh && node scripts/wasm-smoke.js
const fs = require("fs");
const buf = fs.readFileSync("crates/core/tests/data/equation_2x_plus_3_eq_7.ink");
let off = 20; const nStrokes = buf.readUInt32LE(16);
const flat = [nStrokes];
for (let s = 0; s < nStrokes; s++) {
  const n = buf.readUInt32LE(off); off += 4;
  flat.push(n);
  for (let i = 0; i < n; i++) { flat.push(buf.readFloatLE(off), buf.readFloatLE(off + 4)); off += 28; }
}
(async () => {
  const mod = await WebAssembly.instantiate(fs.readFileSync("wasm-demo/ink2tex_wasm.wasm"));
  const w = mod.instance.exports;
  const put = b => { const p = w.alloc(b.length); new Uint8Array(w.memory.buffer, p, b.length).set(b); return p; };
  const f32 = new Float32Array(flat);
  const weights = fs.readFileSync("train/expr.iwt"), labels = fs.readFileSync("train/expr.labels.txt"),
        counts = fs.readFileSync("train/expr.counts.txt");
  const rp = w.recognize_expr(put(new Uint8Array(f32.buffer)), f32.length,
      put(weights), weights.length, put(labels), labels.length, put(counts), counts.length);
  const len = new DataView(w.memory.buffer).getUint32(rp, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(w.memory.buffer, rp + 4, len)));
  if (!(json.latex || "").startsWith("2x+3=")) { console.error("FAIL:", json); process.exit(1); }
  console.log("wasm smoke ✓", json.latex);
})();
