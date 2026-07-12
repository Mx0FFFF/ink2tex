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

## Releasing it

✅ Published: <https://github.com/Mx0FFFF/ink2tex>, tagged `v0.1.0`, and the recipe's
`source=` is pinned to that tarball's sha256. The recipe is **submittable as-is**:

1. Fork [toltec-dev/toltec](https://github.com/toltec-dev/toltec).
2. Copy `packaging/toltec/ink2tex/package` to `package/ink2tex/package`.
3. Open a PR against the `testing` branch. It lands in `testing`, then graduates to `stable`.

Bumping the version later means: retag, recompute the sha256 of the new tarball, bump
`pkgver`, and PR again.

## Licensing, which is not optional here

The shipped weights are trained on the **Detexify** dataset, which is **ODbL**. A trained
model is a Produced Work: it must carry the attribution notice, and it does. That is why
`ATTRIBUTION.md` is in the package payload and not just in the repo. Do not strip it.

CROHME is **evaluation only** — some distributions are CC BY-NC-SA, and non-commercial data
must never end up inside a binary users install. See `DESIGN.md` §5.
