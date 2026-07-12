# Attribution

## The shipped weights (`model.iwt`)

`model.iwt` is a neural network trained on **two** handwritten-symbol databases, both
**ODbL**, and it must carry both notices.

### 1. Detexify — 210,454 drawings, 1,123 symbol classes

> **Detexify data** by Daniel Kirsch and the Detexify contributors.
> Source: <https://github.com/kirel/detexify-data>
> Licensed under the **Open Database License (ODbL) v1.0**:
> <https://opendatacommons.org/licenses/odbl/1-0/>

### 2. HWRT / write-math — 4,539 drawings, 65 classes

Detexify is a *symbol-lookup* corpus: nobody ever looks up how to type `2`, `x` or `+`, so
it contains **no digits, no letters and no arithmetic operators**. HWRT supplies them.

> **HWRT database of handwritten symbols** by Martin Thoma.
> Source: <https://doi.org/10.5281/zenodo.50022> · <https://github.com/MartinThoma/hwrt>
> Licensed under the **Open Database License (ODbL) v1.0**.

### What the licence requires of us

The ODbL governs the *database*. A trained model is a **Produced Work** under ODbL §4.6 —
it must carry these notices, but it does not oblige us to open-source the model itself. We
attribute them here **and inside the installed package**, because the data is the reason any
of this works. If you **redistribute either database** (or a substantial extract, such as
`train/dataset_full/`), the ODbL's share-alike terms apply to *that* — read the licence.

### What we deliberately did *not* use

**CROHME** and Google's **MathWriting** both contain the digits, letters and operators we
needed, and both are **CC BY-NC-SA** — *non-commercial*. Non-commercial data must never end
up inside a binary that strangers install from a package repository, so neither was used for
the shipped weights, and neither may be. See `DESIGN.md` §5. HWRT was chosen precisely
because it is ODbL, like Detexify, and so lands inside attribution we already owe.

CROHME is used for **evaluation only** and is never trained on: some CROHME distributions
are CC BY-NC-SA, and non-commercial data must not end up baked into a binary that users
install. See `DESIGN.md` §5.

## The code

`ink2tex` itself is licensed **MIT OR Apache-2.0** (see `Cargo.toml`).

## Dependencies of note

- [`libremarkable`](https://github.com/canselcik/libremarkable) — framebuffer/input for the
  reMarkable (MIT). Used only by `crates/rm`.

There is deliberately **no ML runtime dependency**: the int8 inference kernel is
hand-written (`crates/core/src/classify/`). See `docs/core-invariants.md`.
