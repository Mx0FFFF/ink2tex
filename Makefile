# ink2tex — see CLAUDE.md for the contract. `make check-bans` is what CI runs.
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
        deploy-model screenshot check-bans core-purity device-facts fmt clippy ci \
        dataset train eval

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

$(DATASET)/meta.json: $(NDJSON) train/dataset/classes.txt
	cargo run -q --release -p ink2tex-desktop -- --prepare-detexify $(NDJSON) \
	  --out-dir $(DATASET) --classes train/dataset/classes.txt

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
# Needs rm2fb on the device. ⚠ BACK UP THE DEVICE FIRST (.claude/rules/device.md).
run ink: deploy
	ssh $(HOST) 'systemctl stop xochitl; $(RM_BIN) --ink --out /home/root/out.ink --dur $(DUR); systemctl start xochitl'
	scp $(HOST):/home/root/out.ink /tmp/out.ink && echo "view: make replay FILE=/tmp/out.ink"

# Push the trained model + labels to the device (MODEL=/LABELS= to override).
deploy-model:
	scp $(MODEL) $(HOST):/home/root/model.iwt
	scp $(LABELS) $(HOST):/home/root/model.labels.txt

# On-device recognition: draw one symbol, get top-5 LaTeX in your terminal.
# Result goes to stdout over SSH, so this needs NO rm2fb.
recognize: deploy deploy-model
	ssh $(HOST) '$(RM_BIN) --recognize --model /home/root/model.iwt --labels /home/root/model.labels.txt --dur $(DUR)'

screenshot:
	ssh $(HOST) 'cat /tmp/screen.png' > /tmp/screen.png && echo "pulled /tmp/screen.png"

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
