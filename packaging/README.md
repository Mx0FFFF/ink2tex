# Packaging for Toltec

[Toltec](https://toltec-dev.org) is the community package repository for the reMarkable.
Installing from it is how a user gets this without an SSH session and a `scp`.

```bash
make ipk        # → target/ipk/ink2tex_<ver>-1_rm2.ipk
```

## What ships

| path | what |
|---|---|
| `/opt/bin/ink2tex` | the armv7 binary (~450 KB, no ML runtime, no dynamic ML deps) |
| `/opt/usr/share/ink2tex/model.iwt` | the int8 weights (~164 KB) |
| `/opt/usr/share/ink2tex/model.labels.txt` | class index → Detexify symbolId |
| `/opt/usr/share/ink2tex/ATTRIBUTION.md` | **required** — the weights are ODbL-derived |

The binary looks for its weights in `/opt/usr/share/ink2tex/` and then **next to itself**,
so both a packaged install and a bare `scp`-to-`/home/root` deployment work with no flags.

`installdepends` is **empty on purpose**: recognition reads the pen through evdev and prints
to stdout, so it needs neither rm2fb nor a launcher. It runs on a stock device with nothing
but Toltec. (On-screen *inking* would need `display`; that is not what this package does.)

## Two ways to build it, and why both exist

- **`make ipk`** packages *the tree you have*. Use it to test a package on a device before
  anything is published. It is what was used to verify the layout on real hardware.
- **`packaging/toltec/ink2tex/package`** is the Toltec *recipe*. Toltec builds it itself, in
  a clean `rust:v3.3` Docker image, from a **public source URL** checked against a sha256.

They install the same files. If you change one, change the other.

## To actually release it

1. **Publish the source.** The recipe's `source=` must be a public tarball — Toltec builds
   in a clean room and will not take a local path. Push the repo, tag `v0.1.0`.
2. Set `source=(.../v0.1.0.zip)` and replace `sha256sums=(SKIP)` with the real checksum.
3. Fork [toltec-dev/toltec](https://github.com/toltec-dev/toltec), drop the recipe in
   `package/ink2tex/package`, and open a PR against `testing`.
4. It lands in `testing`, then graduates to `stable`.

Until step 1, the recipe is correct but unbuildable by Toltec — that is the *only* thing
between here and a release, and it is a decision (publish the code), not a task.

## Licensing, which is not optional here

The shipped weights are trained on the **Detexify** dataset, which is **ODbL**. A trained
model is a Produced Work: it must carry the attribution notice, and it does. That is why
`ATTRIBUTION.md` is in the package payload and not just in the repo. Do not strip it.

CROHME is **evaluation only** — some distributions are CC BY-NC-SA, and non-commercial data
must never end up inside a binary users install. See `DESIGN.md` §5.
