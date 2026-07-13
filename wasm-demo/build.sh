#!/usr/bin/env bash
# Assemble the self-contained browser demo. Serve the folder with any static server:
#     bash wasm-demo/build.sh && python3 -m http.server -d wasm-demo 8000
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p ink2tex-wasm --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/ink2tex_wasm.wasm wasm-demo/
cp train/expr.iwt train/expr.labels.txt train/expr.counts.txt wasm-demo/
echo "wasm-demo/ ready ($(du -sh wasm-demo | cut -f1) total)"
