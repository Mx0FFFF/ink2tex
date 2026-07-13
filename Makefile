# ink2tex — see DESIGN.md and docs/core-invariants.md. `make check-bans` is what CI runs.
CORE      := ink2tex-core
HOST      ?= root@10.11.99.1
RM_TARGET := armv7-unknown-linux-gnueabihf
RM_BIN    := /home/root/ink2tex-rm
DUR       ?= 30
FILE      ?=
MODEL     ?= train/model.iwt
LABELS    ?= train/model.labels.txt

DUMP      ?= $(HOME)/Downloads/detexify.sql.gz
NDJSON    := train/detexify_raw/detexify.ndjson
DATASET   ?= train/dataset_full
VALSET    ?= train/dataset_val
EPOCHS    ?= 60

.PHONY: test replay harness build-rm deploy probe record run ink recognize \
        deploy-model deploy-expr expr screenshot check-bans core-purity device-facts fmt clippy ci \
        dataset train eval ipk retrain-corrections serve wasm-demo

# --- your feedback loops -----------------------------------------------------

# The whole corpus. Run this constantly.
test:
	cargo test -p $(CORE)

# Headless pipeline -> PNG. Your eyes (you cannot see the E-Ink screen).
#   make replay FILE=tests/corpus/foo.ink   -> /tmp/out.png
replay:
	@test -n "$(FILE)" || { echo "usage: make replay FILE=path/to.ink"; exit 2; }
	cargo run -q -p ink2tex-desktop -- --replay "$(FILE)" --render-to /tmp/out.png
	@echo "wrote /tmp/out.png"

# Interactive desktop harness (needs a display). Stub until built out.
harness:
	cargo run -p ink2tex-desktop -- --harness

# --- training (offline, host-only; see train/README.md) ----------------------

# The Detexify bulk pg_dump → NDJSON → a dataset rasterized through core's OWN
# rasterizer (that is what keeps training and on-device inference pixel-identical).
# --classes pins the label space, so datasets stay concatenable and labels stable.
$(NDJSON): $(DUMP)
	@mkdir -p $(dir $@)
	python3 train/detexify_sql_to_ndjson.py $(DUMP) -o $@

$(DATASET)/meta.json: $(NDJSON) train/model.labels.txt
	cargo run -q --release -p ink2tex-desktop -- --prepare-detexify $(NDJSON) \
	  --out-dir $(DATASET) --classes train/model.labels.txt

dataset: $(DATASET)/meta.json

# Train, quantize to int8, export model.iwt + labels, and keep the held-out split.
train: dataset
	python3 train/train.py --data $(DATASET) --epochs $(EPOCHS) --out $(MODEL) \
	  --dump-val $(VALSET)

# Score a model on the held-out split through the *int8* kernel — i.e. the number the
# device will actually produce, not the float one PyTorch reports. MODEL= to compare.
eval:
	cargo run -q --release -p ink2tex-desktop -- --eval $(VALSET) --model $(MODEL)

# --- device -----------------------------------------------------------------

# Cross-compile the device frontend via `cross` (Docker) — no host arm linker
# needed. Alternative: install gcc-arm-linux-gnueabihf and swap in `cargo build`.
build-rm:
	cross build -p ink2tex-rm --release --target $(RM_TARGET)

deploy: build-rm
	scp target/$(RM_TARGET)/release/ink2tex-rm $(HOST):$(RM_BIN)

# Probe the digitizer (read-only; safe, no framebuffer). Fills DEVICE FACTS row 3.
probe: deploy
	ssh $(HOST) '$(RM_BIN) --probe'

# Capture strokes to .ink WITHOUT drawing (safe; works even without rm2fb).
# DUR=seconds. Pulls the result back so you can `make replay` it.
record: deploy
	ssh $(HOST) '$(RM_BIN) --record --out /home/root/out.ink --dur $(DUR)'
	scp $(HOST):/home/root/out.ink /tmp/out.ink && echo "view: make replay FILE=/tmp/out.ink"

# Live inking: stop xochitl, draw with a fast DU waveform, restore xochitl after.
# Needs rm2fb on the device. ⚠ BACK UP THE DEVICE FIRST (docs/device.md).
run ink: deploy
	ssh $(HOST) 'systemctl stop xochitl; $(RM_BIN) --ink --out /home/root/out.ink --dur $(DUR); systemctl start xochitl'
	scp $(HOST):/home/root/out.ink /tmp/out.ink && echo "view: make replay FILE=/tmp/out.ink"

# Push the trained model + labels to the device (MODEL=/LABELS= to override).
deploy-model:
	scp $(MODEL) $(HOST):/home/root/model.iwt
	scp $(LABELS) $(HOST):/home/root/model.labels.txt

# Push the EXPRESSION model (train/expr.* — the stable role-name; retrains re-point it)
# under the role-names the --expr mode looks for. Separate from the M1 lookup model on
# purpose: the two modes answer different questions (DESIGN §4.3).
deploy-expr:
	scp train/expr.iwt $(HOST):/home/root/expr.iwt
	scp train/expr.labels.txt $(HOST):/home/root/expr.labels.txt
	scp train/expr.counts.txt $(HOST):/home/root/expr.counts.txt

# Write a line of math on the tablet -> LaTeX in this terminal. All on-device compute.
expr: deploy deploy-expr
	ssh $(HOST) '$(RM_BIN) --expr --dur $(DUR)'

# On-device recognition: draw one symbol, get top-5 LaTeX in your terminal.
# Result goes to stdout over SSH, so this needs NO rm2fb.
recognize: deploy deploy-model
	ssh $(HOST) '$(RM_BIN) --recognize --model /home/root/model.iwt --labels /home/root/model.labels.txt --dur $(DUR)'

screenshot:
	ssh $(HOST) 'cat /tmp/screen.png' > /tmp/screen.png && echo "pulled /tmp/screen.png"

# The M4/M5 flywheel: pull the corrections the UI logged on the device, fold them into
# the collected corpus, rebuild the dataset, retrain. Every fix becomes training data.
retrain-corrections:
	scp $(HOST):/home/root/corrections.ndjson train/collected/corrections.ndjson || \
	  { echo "no corrections on the device yet"; exit 1; }
	cat train/collected/*.ndjson > /tmp/ink2tex_collected.ndjson
	cargo run -q --release -p ink2tex-desktop -- --prepare-detexify /tmp/ink2tex_collected.ndjson \
	  --out-dir train/dataset_collected --classes train/labels_v4.txt
	python3 train/train.py --data train/dataset_full4 train/dataset_hwrt4 train/dataset_collected \
	  --epochs 40 --class-weight none --out train/expr.iwt --dump-val train/dataset_val_expr

# Correction UI on the tablet (browser at http://10.11.99.1:8222)
serve: deploy deploy-expr
	ssh $(HOST) '$(RM_BIN) --serve'

# The browser demo: same model, same core, compiled to WASM.
wasm-demo:
	bash wasm-demo/build.sh
	node scripts/wasm-smoke.js

# --- packaging ---------------------------------------------------------------

# Build the Toltec/opkg package (.ipk) from the current tree — binary + weights +
# the ODbL attribution the weights oblige us to carry. See packaging/README.md.
ipk: build-rm
	bash packaging/build-ipk.sh

# SSH-probe the device and print the DEVICE FACTS table inputs.
device-facts:
	bash scripts/device-facts.sh $(HOST)

# --- guardrails (what CI enforces) ------------------------------------------

core-purity:
	bash scripts/check-core-purity.sh

check-bans: core-purity
	cargo deny check bans

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

# Everything CI runs, locally.
ci: check-bans test clippy
