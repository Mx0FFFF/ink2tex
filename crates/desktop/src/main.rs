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

use std::collections::HashMap;
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

    /// Preprocess a Detexify JSON export into a training dataset directory. Rasterizes
    /// through the SAME core rasterizer inference uses, so there is no train/infer skew.
    #[arg(long, value_name = "DETEXIFY_JSON")]
    prepare_detexify: Option<PathBuf>,

    /// Output directory for --prepare-detexify.
    #[arg(long, value_name = "DIR")]
    out_dir: Option<PathBuf>,

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
        return prepare_detexify(&input, &out, 32);
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
            let name = labels
                .as_ref()
                .and_then(|l| l.get(p.class))
                .map(str::to_string)
                .unwrap_or_else(|| format!("class {}", p.class));
            println!("  {}. {:>5.1}%  {}", i + 1, p.prob * 100.0, name);
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

/// Detexify JSON → a flat training dataset: `images.u8` (N×size²), `features.f32`
/// (N×NUM_FEATURES), `labels.u32` (N), `classes.txt` (index→class), `meta.json`.
/// numpy reads these directly (`np.fromfile`). Rasterizing here — not in Python —
/// is what keeps training and on-device inference pixel-identical.
fn prepare_detexify(input: &Path, out_dir: &Path, size: usize) -> Result<()> {
    let text =
        std::fs::read_to_string(input).with_context(|| format!("reading {}", input.display()))?;
    let samples = detexify::parse(&text)?;

    let mut classes: Vec<String> = samples.iter().map(|s| s.class.clone()).collect();
    classes.sort();
    classes.dedup();
    let index: HashMap<&str, u32> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as u32))
        .collect();

    std::fs::create_dir_all(out_dir)?;
    let (mut images, mut feats, mut labels) = (Vec::new(), Vec::new(), Vec::new());
    let mut online = Vec::new();
    for s in &samples {
        for v in rasterize(&s.strokes, size) {
            images.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        for f in global_features(&s.strokes) {
            feats.extend_from_slice(&f.to_le_bytes());
        }
        for v in online_features(&s.strokes, ONLINE_POINTS) {
            online.extend_from_slice(&v.to_le_bytes());
        }
        labels.extend_from_slice(&index[s.class.as_str()].to_le_bytes());
    }

    std::fs::write(out_dir.join("images.u8"), &images)?;
    std::fs::write(out_dir.join("features.f32"), &feats)?;
    std::fs::write(out_dir.join("online.f32"), &online)?;
    std::fs::write(out_dir.join("labels.u32"), &labels)?;
    std::fs::write(out_dir.join("classes.txt"), classes.join("\n") + "\n")?;
    std::fs::write(
        out_dir.join("meta.json"),
        format!(
            "{{\"n\": {}, \"size\": {}, \"num_features\": {}, \"num_classes\": {}, \"online_len\": {}}}\n",
            samples.len(),
            size,
            NUM_FEATURES,
            classes.len(),
            ONLINE_CHANNELS * ONLINE_POINTS
        ),
    )?;
    eprintln!(
        "prepared {} samples / {} classes → {}/ (images.u8, features.f32, online.f32, labels.u32, classes.txt, meta.json)",
        samples.len(),
        classes.len(),
        out_dir.display()
    );
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
        if preds.first().map(|p| p.class) == Some(label) {
            top1 += 1;
        }
        if preds.iter().any(|p| p.class == label) {
            top5 += 1;
        }
    }
    let pct = |c: usize| 100.0 * c as f64 / n.max(1) as f64;
    println!(
        "evaluated {n} samples (int8 forward pass): top-1 {:.1}%  top-5 {:.1}%",
        pct(top1),
        pct(top5)
    );
    Ok(())
}
