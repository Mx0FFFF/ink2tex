# Attribution

## The shipped weights (`model.iwt`)

`model.iwt` is a neural network trained on the **Detexify** handwritten-symbol dataset —
210,454 drawings of 1,123 LaTeX symbols, contributed by users of
[detexify.kirelabs.org](https://detexify.kirelabs.org).

> **Detexify data** by Daniel Kirsch and the Detexify contributors.
> Source: <https://github.com/kirel/detexify-data>
> Licensed under the **Open Database License (ODbL) v1.0**:
> <https://opendatacommons.org/licenses/odbl/1-0/>

The ODbL governs the *database*. A trained model is a **Produced Work** under ODbL §4.6 —
it must carry this notice, but it does not oblige us to open-source the model itself. We
attribute it here, and in the package metadata, because the data is the reason this works
at all. If you **redistribute the Detexify database** (or a substantial extract of it, such
as `train/dataset_full/`), the ODbL's share-alike terms apply to *that* — read the licence.

CROHME is used for **evaluation only** and is never trained on: some CROHME distributions
are CC BY-NC-SA, and non-commercial data must not end up baked into a binary that users
install. See `DESIGN.md` §5.

## The code

`ink2tex` itself is licensed **MIT OR Apache-2.0** (see `Cargo.toml`).

## Dependencies of note

- [`libremarkable`](https://github.com/canselcik/libremarkable) — framebuffer/input for the
  reMarkable (MIT). Used only by `crates/rm`.

There is deliberately **no ML runtime dependency**: the int8 inference kernel is
hand-written (`crates/core/src/classify/`). See `CLAUDE.md`.
