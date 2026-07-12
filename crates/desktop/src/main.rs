//! ink2tex-desktop — dev harness and, crucially, the **headless replay renderer**.
//! You cannot see the E-Ink screen, so this is how you (and CI) verify visual work:
//!
//! ```text
//! ink2tex-desktop --replay <ink> --render-to <png>
//! ```
//!
//! renders an `.ink` through the pipeline to a PNG with no device and no display.
//! It also hosts training-time tooling: `--raster` (see the classifier's input),
//! `--prepare-detexify` (build a training dataset through the *same* rasterizer
//! inference uses — no skew), and `--dump-weights` (check a trained `.iwt` blob).

mod detexify;
mod render;
mod synth;

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use ink2tex_core::classify::raster::NUM_FEATURES;
use ink2tex_core::classify::{
    global_features, online_features, rasterize, recognize, Labels, Weights, ONLINE_CHANNELS,
    ONLINE_POINTS,
};
use ink2tex_core::Ink;

#[derive(Parser, Debug)]
#[command(
    name = "ink2tex-desktop",
    about = "Headless replay renderer + dev harness"
)]
struct Cli {
    /// Render this `.ink` file headlessly (pair with --render-to).
    #[arg(long, value_name = "INK")]
    replay: Option<PathBuf>,

    /// PNG output path for --replay.
    #[arg(long, value_name = "PNG")]
    render_to: Option<PathBuf>,

    /// Write a deterministic synthetic `.ink` here and exit (for demos/tests).
    #[arg(long, value_name = "INK")]
    synth: Option<PathBuf>,

    /// Rasterize an `.ink` to the classifier's 32×32 input and print it as ASCII —
    /// "see what the classifier sees".
    #[arg(long, value_name = "INK")]
    raster: Option<PathBuf>,

    /// Per-stroke geometry (size, duration, distance to its nearest neighbour) — the
    /// numbers you need to tell a stray tap from a deliberate `\cdot`.
    #[arg(long, value_name = "INK")]
    strokes: Option<PathBuf>,

    /// Preprocess a Detexify JSON export into a training dataset directory. Rasterizes
    /// through the SAME core rasterizer inference uses, so there is no train/infer skew.
    /// Streams NDJSON (`-` = stdin), so a 1 GB bulk dump costs one sample of memory.
    #[arg(long, value_name = "DETEXIFY_JSON")]
    prepare_detexify: Option<PathBuf>,

    /// Output directory for --prepare-detexify.
    #[arg(long, value_name = "DIR")]
    out_dir: Option<PathBuf>,

    /// Pin --prepare-detexify's label space to this class list (one symbolId per line;
    /// index = line order). Samples outside it are dropped. Datasets prepared with the
    /// same list share label ids and can simply be concatenated.
    #[arg(long, value_name = "TXT")]
    classes: Option<PathBuf>,

    /// Mint `tests/corpus` cases from a Detexify export: one `<class>.ink` +
    /// `<class>.expected.tex` per class listed in --classes. Needs --out-dir.
    #[arg(long, value_name = "DETEXIFY_JSON")]
    export_corpus: Option<PathBuf>,

    /// Parse an `.iwt` weights blob and print its tensors (verifies the trainer's
    /// output against core's parser).
    #[arg(long, value_name = "IWT")]
    dump_weights: Option<PathBuf>,

    /// Recognize the symbol in an `.ink`: rasterize → int8 CNN → top-5 LaTeX.
    /// Needs --model <iwt>; --labels maps class indices to commands.
    #[arg(long, value_name = "INK")]
    recognize: Option<PathBuf>,

    /// Trained `.iwt` model for --recognize.
    #[arg(long, value_name = "IWT")]
    model: Option<PathBuf>,

    /// Labels file (one LaTeX command per line) for --recognize output.
    #[arg(long, value_name = "TXT")]
    labels: Option<PathBuf>,

    /// Evaluate a model on a prepared dataset: top-1 / top-5 accuracy over the int8
    /// forward pass. Validates on-device inference against ground truth. Needs --model.
    #[arg(long, value_name = "DATASET_DIR")]
    eval: Option<PathBuf>,

    /// Recognize a linear expression: segment the `.ink` into symbols and classify
    /// each left-to-right (M2). Needs --model; --labels maps indices to commands.
    #[arg(long, value_name = "INK")]
    recognize_expr: Option<PathBuf>,

    /// Interactive harness — needs a display. Not implemented at M0.
    #[arg(long)]
    harness: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(out) = cli.synth {
        let ink = synth::sample_ink();
        std::fs::write(&out, ink.encode())
            .with_context(|| format!("writing synthetic ink to {}", out.display()))?;
        eprintln!(
            "wrote synthetic .ink ({} strokes, {} points) to {}",
            ink.strokes.len(),
            ink.point_count(),
            out.display()
        );
        return Ok(());
    }

    if let Some(input) = cli.prepare_detexify {
        let out = cli
            .out_dir
            .context("--prepare-detexify needs --out-dir <dir>")?;
        return prepare_detexify(&input, &out, 32, cli.classes.as_deref());
    }

    if let Some(input) = cli.export_corpus {
        let out = cli
            .out_dir
            .context("--export-corpus needs --out-dir <dir>")?;
        let want = cli
            .classes
            .context("--export-corpus needs --classes <txt>")?;
        return export_corpus(&input, &out, &want);
    }

    if let Some(path) = cli.strokes {
        return stroke_stats(&path);
    }

    if let Some(path) = cli.dump_weights {
        return dump_weights(&path);
    }

    if let Some(ink_path) = cli.recognize {
        let model_path = cli.model.context("--recognize needs --model <iwt>")?;
        let ink_bytes =
            std::fs::read(&ink_path).with_context(|| format!("reading {}", ink_path.display()))?;
        let ink = Ink::decode(&ink_bytes)
            .with_context(|| format!("parsing {} as .ink", ink_path.display()))?;
        let bitmap = rasterize(&ink.strokes, 32);
        let feats = global_features(&ink.strokes);
        let online = online_features(&ink.strokes, ONLINE_POINTS);
        let blob = std::fs::read(&model_path)
            .with_context(|| format!("reading {}", model_path.display()))?;
        let weights = Weights::parse(&blob).context("parsing model .iwt")?;
        let preds = recognize(&weights, &bitmap, &feats, &online, 32, 5)
            .context("classifier forward pass")?;
        let labels = match cli.labels {
            Some(p) => Some(Labels::from_lines(
                &std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?,
            )),
            None => None,
        };
        println!("top {} for {}:", preds.len(), ink_path.display());
        for (i, p) in preds.iter().enumerate() {
            // LaTeX first — that's the product. The symbolId stays visible because it is
            // what disambiguates look-alike classes when a prediction surprises you.
            match labels.as_ref().and_then(|l| l.get(p.class)) {
                Some(id) => println!(
                    "  {}. {:>5.1}%  {:<16} ({id})",
                    i + 1,
                    p.prob * 100.0,
                    ink2tex_core::latex::symbol_command(id)
                ),
                None => println!("  {}. {:>5.1}%  class {}", i + 1, p.prob * 100.0, p.class),
            }
        }
        return Ok(());
    }

    if let Some(ink_path) = cli.recognize_expr {
        let model_path = cli.model.context("--recognize-expr needs --model <iwt>")?;
        let bytes =
            std::fs::read(&ink_path).with_context(|| format!("reading {}", ink_path.display()))?;
        let ink = Ink::decode(&bytes)
            .with_context(|| format!("parsing {} as .ink", ink_path.display()))?;
        let blob = std::fs::read(&model_path)
            .with_context(|| format!("reading {}", model_path.display()))?;
        let weights = Weights::parse(&blob).context("parsing model .iwt")?;
        let line =
            ink2tex_core::recognize_line(&ink, &weights, 3).context("expression recognizer")?;
        let labels = match cli.labels {
            Some(p) => Some(Labels::from_lines(
                &std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?,
            )),
            None => None,
        };
        let name = |c: usize| {
            labels
                .as_ref()
                .and_then(|l| l.get(c))
                .map(str::to_string)
                .unwrap_or_else(|| format!("class {c}"))
        };
        // The headline: the full 2-D structure → LaTeX (needs labels for √/bar/op tokens).
        if let Some(l) = labels.as_ref() {
            match ink2tex_core::recognize_expression(&ink, &weights, l, 3) {
                Ok(latex) => println!("LaTeX: {latex}"),
                Err(e) => eprintln!("structure error: {e}"),
            }
        } else {
            eprintln!("(pass --labels for structured LaTeX)");
        }
        let seq: Vec<String> = line
            .iter()
            .filter_map(|s| s.predictions.first())
            .map(|p| name(p.class))
            .collect();
        println!(
            "segmented: {} symbol(s) → {}",
            line.len(),
            seq.join("  ·  ")
        );
        for (i, s) in line.iter().enumerate() {
            println!("  symbol {} (strokes {:?}):", i + 1, s.strokes);
            for p in &s.predictions {
                println!("      {:>5.1}%  {}", p.prob * 100.0, name(p.class));
            }
        }
        return Ok(());
    }

    if let Some(dir) = cli.eval {
        let model_path = cli.model.context("--eval needs --model <iwt>")?;
        return eval_dataset(&dir, &model_path);
    }

    if let Some(path) = cli.raster {
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let ink =
            Ink::decode(&bytes).with_context(|| format!("parsing {} as .ink", path.display()))?;
        print_ascii(&rasterize(&ink.strokes, 32), 32);
        return Ok(());
    }

    if cli.harness {
        bail!("interactive harness is not implemented at M0; use --replay for headless rendering");
    }

    match (cli.replay, cli.render_to) {
        (Some(ink_path), Some(png_path)) => {
            let bytes = std::fs::read(&ink_path)
                .with_context(|| format!("reading {}", ink_path.display()))?;
            let ink = Ink::decode(&bytes)
                .with_context(|| format!("parsing {} as .ink", ink_path.display()))?;
            render::render_to_png(&ink, &png_path)
                .with_context(|| format!("rendering to {}", png_path.display()))?;
            eprintln!(
                "rendered {} strokes / {} points -> {}",
                ink.strokes.len(),
                ink.point_count(),
                png_path.display()
            );
            Ok(())
        }
        (Some(_), None) => bail!("--replay needs --render-to <png>"),
        (None, Some(_)) => bail!("--render-to needs --replay <ink>"),
        (None, None) => {
            bail!("nothing to do: pass --replay <ink> --render-to <png> (or --synth <ink>)")
        }
    }
}

/// Print a `size×size` grayscale image as an ASCII intensity ramp.
fn print_ascii(img: &[f32], size: usize) {
    let ramp = [' ', '.', ':', '+', '*', '#'];
    for y in 0..size {
        let row: String = (0..size)
            .map(|x| {
                let v = img[y * size + x].clamp(0.0, 1.0);
                let i = (v * (ramp.len() - 1) as f32).round() as usize;
                ramp[i.min(ramp.len() - 1)]
            })
            .collect();
        println!("{row}");
    }
}

/// Open a training input: a path, or `-` for stdin (so the SQL→NDJSON converter can
/// pipe a 1 GB dump straight in without ever landing it on disk).
fn open_input(path: &Path) -> Result<Box<dyn BufRead>> {
    if path.as_os_str() == "-" {
        Ok(Box::new(BufReader::new(std::io::stdin())))
    } else {
        let f = File::open(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Box::new(BufReader::new(f)))
    }
}

/// Detexify JSON → a flat training dataset: `images.u8` (N×size²), `features.f32`
/// (N×NUM_FEATURES), `online.f32`, `labels.u32` (N), `classes.txt` (index→class),
/// `meta.json`. numpy reads these directly (`np.fromfile`). Rasterizing here — not in
/// Python — is what keeps training and on-device inference pixel-identical.
///
/// **Streams.** The classic bulk dump is ~1 GB of JSON / 210k records; parsing it into
/// one `Vec<Sample>` would cost several GB of inflated structs. Records are pulled a
/// line at a time and written straight through, so peak memory is *one sample* rather
/// than the corpus. A non-NDJSON input (the small cloudant single-document export)
/// falls back to the whole-file parser.
///
/// `class_space` pins the label vocabulary: the index becomes that file's line order and
/// samples outside it are dropped. That is what makes two datasets **concatenable** —
/// the bulk dump and the detexify-next corpus land on identical label ids — and it keeps
/// `model.labels.txt` stable across retrains. Without it the vocabulary is derived from
/// the data (sorted), which needs a second pass and therefore a seekable input.
fn prepare_detexify(
    input: &Path,
    out_dir: &Path,
    size: usize,
    class_space: Option<&Path>,
) -> Result<()> {
    let classes: Vec<String> = match class_space {
        Some(p) => {
            let text =
                std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
            let v: Vec<String> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();
            if v.is_empty() {
                bail!("--classes {} is empty", p.display());
            }
            eprintln!(
                "label space pinned to {} classes from {}",
                v.len(),
                p.display()
            );
            v
        }
        None => {
            if input.as_os_str() == "-" {
                bail!("reading from stdin needs --classes <txt> (deriving the vocabulary needs a second pass over the input)");
            }
            // Pass 1: learn the vocabulary. Sorted + deduped, as before.
            let mut set = BTreeSet::new();
            for_each_sample(input, |s| {
                set.insert(s.class.clone());
                Ok(())
            })?;
            set.into_iter().collect()
        }
    };
    let index: HashMap<&str, u32> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as u32))
        .collect();

    std::fs::create_dir_all(out_dir)?;
    let mut images = BufWriter::new(File::create(out_dir.join("images.u8"))?);
    let mut feats = BufWriter::new(File::create(out_dir.join("features.f32"))?);
    let mut online = BufWriter::new(File::create(out_dir.join("online.f32"))?);
    let mut labels = BufWriter::new(File::create(out_dir.join("labels.u32"))?);

    // Pass 2: rasterize and write through. `seen` counts per-class so we can report the
    // shape of the corpus — a class the data never covers still occupies a logit.
    let mut seen = vec![0u32; classes.len()];
    let (mut n, mut dropped) = (0usize, 0usize);
    for_each_sample(input, |s| {
        let Some(&id) = index.get(s.class.as_str()) else {
            dropped += 1; // out of vocabulary — the class-space filter at work
            return Ok(());
        };
        for v in rasterize(&s.strokes, size) {
            images.write_all(&[(v.clamp(0.0, 1.0) * 255.0).round() as u8])?;
        }
        for f in global_features(&s.strokes) {
            feats.write_all(&f.to_le_bytes())?;
        }
        for v in online_features(&s.strokes, ONLINE_POINTS) {
            online.write_all(&v.to_le_bytes())?;
        }
        labels.write_all(&id.to_le_bytes())?;
        seen[id as usize] += 1;
        n += 1;
        if n % 25_000 == 0 {
            eprintln!("  … {n} samples");
        }
        Ok(())
    })?;
    for w in [&mut images, &mut feats, &mut online, &mut labels] {
        w.flush()?;
    }

    std::fs::write(out_dir.join("classes.txt"), classes.join("\n") + "\n")?;
    std::fs::write(
        out_dir.join("meta.json"),
        format!(
            "{{\"n\": {}, \"size\": {}, \"num_features\": {}, \"num_classes\": {}, \"online_len\": {}}}\n",
            n,
            size,
            NUM_FEATURES,
            classes.len(),
            ONLINE_CHANNELS * ONLINE_POINTS
        ),
    )?;

    let empty = seen.iter().filter(|&&c| c == 0).count();
    eprintln!(
        "prepared {n} samples / {} classes → {}/ (images.u8, features.f32, online.f32, labels.u32, classes.txt, meta.json)",
        classes.len(),
        out_dir.display()
    );
    if dropped > 0 || empty > 0 {
        eprintln!(
            "  dropped {dropped} out-of-vocabulary samples; {empty} classes have no samples here"
        );
    }
    Ok(())
}

/// Mint `tests/corpus` fixtures from a Detexify export — one case per class in `want`.
///
/// The corpus suite is the project's immune system (docs/core-invariants.md), and it was guarding
/// the entire classifier with a *single* case. Detexify is real human handwriting, so its
/// samples make honest fixtures: any regression in the preprocessing contract, the int8
/// kernel, or the label space breaks them instantly. (This is exactly how the scale-
/// invariance bug in `global_features` would have been caught the moment it landed.)
///
/// Strokes are rescaled into the device's frame — bbox to `[0,1]`, aspect preserved —
/// because that is what an `.ink` *is*: normalized ink, not somebody's pixel grid.
fn export_corpus(input: &Path, out_dir: &Path, want: &Path) -> Result<()> {
    let text =
        std::fs::read_to_string(want).with_context(|| format!("reading {}", want.display()))?;
    let wanted: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    std::fs::create_dir_all(out_dir)?;

    let mut taken: HashMap<&str, ()> = HashMap::new();
    for_each_sample(input, |s| {
        let Some(&class) = wanted.iter().find(|c| **c == s.class) else {
            return Ok(()); // not one of the classes asked for
        };
        if taken.contains_key(class) {
            return Ok(()); // first sample of each class wins
        }

        // Detexify ships raw pixels; an `.ink` is normalized. Fit the bounding box into
        // [0,1] with a single uniform scale, so the glyph's aspect ratio survives.
        let pts = || s.strokes.iter().flat_map(|st| st.points.iter());
        let Some(p0) = pts().next() else {
            return Ok(());
        };
        let (mut nx, mut ny, mut mx, mut my) = (p0.x, p0.y, p0.x, p0.y);
        for p in pts() {
            nx = nx.min(p.x);
            ny = ny.min(p.y);
            mx = mx.max(p.x);
            my = my.max(p.y);
        }
        let span = (mx - nx).max(my - ny).max(1e-6);

        let ink = Ink {
            source_width: 1.0,
            source_height: 1.0,
            strokes: s
                .strokes
                .iter()
                .map(|st| ink2tex_core::Stroke {
                    points: st
                        .points
                        .iter()
                        .map(|p| {
                            ink2tex_core::Point::new(
                                (p.x - nx) / span,
                                (p.y - ny) / span,
                                p.pressure,
                                p.tilt_x,
                                p.tilt_y,
                                p.t_us,
                            )
                        })
                        .collect(),
                })
                .collect(),
        };

        let stem = class.replace(':', "_");
        std::fs::write(out_dir.join(format!("{stem}.ink")), ink.encode())?;
        std::fs::write(
            out_dir.join(format!("{stem}.expected.tex")),
            format!("{}\n", ink2tex_core::latex::symbol_command(class)),
        )?;
        eprintln!(
            "  {stem}.ink  ({} strokes) → {}",
            ink.strokes.len(),
            ink2tex_core::latex::symbol_command(class)
        );
        taken.insert(class, ());
        Ok(())
    })?;

    eprintln!(
        "exported {} corpus case(s) to {}/",
        taken.len(),
        out_dir.display()
    );
    for c in &wanted {
        if !taken.contains_key(c) {
            eprintln!("  ⚠ no sample found for {c}");
        }
    }
    Ok(())
}

/// Pull samples through `f` one at a time. NDJSON streams (constant memory); anything
/// else is re-read as a single document, which is fine — only the bulk dump is big.
fn for_each_sample(input: &Path, mut f: impl FnMut(&detexify::Sample) -> Result<()>) -> Result<()> {
    let mut any = false;
    for line in open_input(input)?.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(s) = detexify::parse_line(&line) {
            any = true;
            f(&s)?;
        }
    }
    if any {
        return Ok(());
    }
    // Not NDJSON — the cloudant export is one big JSON document.
    let text =
        std::fs::read_to_string(input).with_context(|| format!("reading {}", input.display()))?;
    for s in detexify::parse(&text)? {
        f(&s)?;
    }
    Ok(())
}

/// Per-stroke geometry, so a noise filter can be designed against numbers rather than
/// vibes. The hard part is that a stray tap and a deliberate `\cdot` are *both* tiny —
/// the thing that separates them is whether anything else is nearby.
fn stroke_stats(path: &Path) -> Result<()> {
    let ink = Ink::decode(&std::fs::read(path)?).context("decoding .ink")?;
    let bbox = |s: &ink2tex_core::Stroke| {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &s.points {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        (x0, y0, x1, y1)
    };
    let diag = |b: (f32, f32, f32, f32)| (b.2 - b.0).hypot(b.3 - b.1);
    let boxes: Vec<_> = ink.strokes.iter().map(bbox).collect();
    // Point-to-point, NOT bbox-to-bbox: a selection lasso has a bounding box that
    // encloses the whole page, so bbox distance calls every stray tap "close to" it. What
    // matters is how far the actual ink is.
    let gap = |a: &ink2tex_core::Stroke, b: &ink2tex_core::Stroke| {
        let mut best = f32::MAX;
        for p in &a.points {
            for q in &b.points {
                best = best.min((p.x - q.x).hypot(p.y - q.y));
            }
        }
        best
    };
    let diags: Vec<f32> = boxes.iter().map(|b| diag(*b)).collect();
    let mut sorted = diags.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    println!(
        "{} strokes — median stroke diagonal {:.4}",
        ink.strokes.len(),
        median
    );
    println!(
        "  {:>3}  {:>5}  {:>8}  {:>8}  {:>9}  {:>9}",
        "#", "pts", "diag", "diag/med", "ms", "nearest"
    );
    for (i, s) in ink.strokes.iter().enumerate() {
        let ms = s.points.last().map_or(0, |p| p.t_us) as f64 / 1000.0
            - s.points.first().map_or(0, |p| p.t_us) as f64 / 1000.0;
        let nearest = ink
            .strokes
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, o)| gap(s, o))
            .fold(f32::MAX, f32::min);
        println!(
            "  {:>3}  {:>5}  {:>8.4}  {:>8.2}  {:>9.0}  {:>9.4}",
            i,
            s.points.len(),
            diags[i],
            diags[i] / median.max(1e-6),
            ms,
            nearest
        );
    }
    Ok(())
}

/// Parse an `.iwt` blob with core's own parser and print each tensor — the
/// cross-language check that the Python trainer's output matches the Rust reader.
fn dump_weights(path: &Path) -> Result<()> {
    let blob = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let w = Weights::parse(&blob).context("parsing .iwt")?;
    println!("{} tensors in {}", w.len(), path.display());
    for t in w.tensors() {
        let head = match t.dtype {
            0 => join(t.as_i8().iter().take(6).map(|v| v.to_string())),
            1 => join(t.as_i32().iter().take(6).map(|v| v.to_string())),
            _ => join(t.as_f32().iter().take(6).map(|v| format!("{v:.4}"))),
        };
        println!(
            "  {:<18} dtype={} dims={:?} scale={:.6} head=[{head}]",
            t.name, t.dtype, t.dims, t.scale
        );
    }
    Ok(())
}

fn join(it: impl Iterator<Item = String>) -> String {
    it.collect::<Vec<_>>().join(", ")
}

/// Run the int8 forward pass over a whole prepared dataset and report top-1/top-5
/// accuracy vs. ground truth — the end-to-end check that on-device inference works
/// and how much the int8 quantization costs vs. the float model.
fn eval_dataset(dir: &Path, model_path: &Path) -> Result<()> {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)
            .context("parsing meta.json")?;
    let n = meta["n"].as_u64().unwrap_or(0) as usize;
    let size = meta["size"].as_u64().unwrap_or(32) as usize;
    let nf = meta["num_features"].as_u64().unwrap_or(0) as usize;

    let images = std::fs::read(dir.join("images.u8"))?;
    let feat_bytes = std::fs::read(dir.join("features.f32"))?;
    let label_bytes = std::fs::read(dir.join("labels.u32"))?;
    let online_len = meta["online_len"].as_u64().unwrap_or(0) as usize;
    let online_all: Vec<f32> = std::fs::read(dir.join("online.f32"))
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let blob = std::fs::read(model_path)?;
    let weights = Weights::parse(&blob).context("parsing model .iwt")?;

    let feats_all: Vec<f32> = feat_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let labels_all: Vec<u32> = label_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let px = size * size;
    let (mut top1, mut top5) = (0usize, 0usize);
    // Per-class tallies as well as the running totals. The corpus is heavily imbalanced
    // — `\int` has 3,937 samples where the median class has 53 — so a micro average is
    // dominated by the head and can read as excellent while the tail is unusable. The
    // macro average (mean per-class recall) is the number that notices.
    let n_classes = meta["num_classes"].as_u64().unwrap_or(0) as usize;
    let n_classes = if n_classes > 0 {
        n_classes
    } else {
        labels_all.iter().max().map_or(0, |m| *m as usize + 1)
    };
    let (mut cls_n, mut cls_hit1, mut cls_hit5) = (
        vec![0usize; n_classes],
        vec![0usize; n_classes],
        vec![0usize; n_classes],
    );
    for i in 0..n {
        let bitmap: Vec<f32> = images[i * px..(i + 1) * px]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        let feats = &feats_all[i * nf..(i + 1) * nf];
        let online: &[f32] = if online_len > 0 && !online_all.is_empty() {
            &online_all[i * online_len..(i + 1) * online_len]
        } else {
            &[]
        };
        let label = labels_all[i] as usize;
        let preds = recognize(&weights, &bitmap, feats, online, size, 5)?;
        cls_n[label] += 1;
        if preds.first().map(|p| p.class) == Some(label) {
            top1 += 1;
            cls_hit1[label] += 1;
        }
        if preds.iter().any(|p| p.class == label) {
            top5 += 1;
            cls_hit5[label] += 1;
        }
    }
    let pct = |c: usize| 100.0 * c as f64 / n.max(1) as f64;
    println!(
        "evaluated {n} samples (int8 forward pass): top-1 {:.1}%  top-5 {:.1}%",
        pct(top1),
        pct(top5)
    );

    // Macro: average the per-class rates, over the classes this split actually contains.
    let present: Vec<usize> = (0..n_classes).filter(|&c| cls_n[c] > 0).collect();
    if !present.is_empty() {
        let mean = |hits: &[usize]| {
            100.0
                * present
                    .iter()
                    .map(|&c| hits[c] as f64 / cls_n[c] as f64)
                    .sum::<f64>()
                / present.len() as f64
        };
        println!(
            "  macro (mean over the {} classes present): top-1 {:.1}%  top-5 {:.1}%",
            present.len(),
            mean(&cls_hit1),
            mean(&cls_hit5)
        );
    }
    Ok(())
}
