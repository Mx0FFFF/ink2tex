//! Symbol classification (M1): a hand-rolled int8 CNN over a 32×32 rendering of a
//! stroke group, emitting **ranked** top-k LaTeX-symbol candidates. Ranked, never a
//! lone argmax — the correction UI is the product and depends on alternatives
//! (NON-NEGOTIABLE #5 / DESIGN.md §7).
//!
//! This module currently ships the device-free foundation: the integer inference
//! `kernel` and the mmap-able `weights` blob, both fully unit-tested. The concrete
//! layer wiring and the stroke→32×32 rasterizer arrive with the trained model
//! (ROADMAP M1). Intended baseline: conv→relu→maxpool → conv→relu→maxpool →
//! dense→relu → dense→softmax over ~120 classes (DESIGN.md §4.3).

pub mod kernel;
pub mod model;
pub mod raster;
pub mod weights;

pub use model::recognize;
pub use raster::{global_features, rasterize};
pub use weights::{Weights, WeightsWriter};

/// One ranked candidate: a class index and its softmax probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub class: usize,
    pub prob: f32,
}

/// Turn raw logits into the ranked top-k the UI consumes: softmax, then the k
/// highest. Always exposes alternatives, never a lone argmax.
pub fn rank_logits(logits: &[f32], k: usize) -> Vec<Prediction> {
    let probs = kernel::softmax(logits);
    kernel::top_k(&probs, k)
        .into_iter()
        .map(|(class, prob)| Prediction { class, prob })
        .collect()
}

/// Maps class indices to LaTeX commands (e.g. `12 → "\\alpha"`). The trainer emits
/// this next to the weights as a newline-delimited list.
#[derive(Debug, Clone, Default)]
pub struct Labels(Vec<String>);

impl Labels {
    pub fn from_lines(text: &str) -> Self {
        Labels(
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect(),
        )
    }
    pub fn get(&self, class: usize) -> Option<&str> {
        self.0.get(class).map(String::as_str)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::kernel::*;
    use super::*;

    #[test]
    fn rank_logits_is_ranked_topk() {
        let preds = rank_logits(&[0.0, 5.0, 1.0], 2);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].class, 1); // highest logit first
        assert!(preds[0].prob > preds[1].prob);
    }

    #[test]
    fn labels_map_indices_to_commands() {
        let l = Labels::from_lines("\\alpha\n\\beta\n\n\\gamma\n");
        assert_eq!(l.len(), 3);
        assert_eq!(l.get(0), Some("\\alpha"));
        assert_eq!(l.get(2), Some("\\gamma"));
        assert_eq!(l.get(9), None);
    }

    // Prove the primitives + the weights blob compose into a working forward pass:
    // conv → requant(relu) → maxpool → dense → dequant → softmax → top-k.
    #[test]
    fn primitives_compose_into_a_classifier() {
        let mut w = WeightsWriter::new();
        // conv: 2 out ch over 1 in ch, 3×3. ch0 sums the neighborhood; ch1 negates it.
        let mut cw = Vec::new();
        cw.extend_from_slice(&[1i8; 9]); // out-channel 0: all ones
        cw.extend_from_slice(&[-1i8; 9]); // out-channel 1: all minus-ones
        w.i8("conv.w", &[2, 1, 3, 3], 0.01, &cw);
        w.i32("conv.b", &[2], &[0, 0]);
        // dense over the 2×2×2 = 8 pooled features (CHW: ch0[0..4], ch1[4..8]).
        let mut dw = vec![0i8; 16];
        dw[0..4].fill(1); // class 0 reads channel-0 features
        dw[12..16].fill(1); // class 1 reads channel-1 features
        w.i8("fc.w", &[2, 8], 0.01, &dw);
        w.i32("fc.b", &[2], &[0, 0]);
        let blob = w.finish(); // bind: Weights borrows this, so it must outlive the parse
        let weights = Weights::parse(&blob).unwrap();

        // A strongly positive input: ch0 (sum) large, ch1 (negated) killed by relu.
        let input = vec![100i8; 16]; // 1×4×4
        let conv = weights.get("conv.w").unwrap();
        let (acc, h, wd) = conv2d_i8(
            &input,
            1,
            4,
            4,
            conv.as_i8(),
            2,
            3,
            3,
            &weights.get("conv.b").unwrap().as_i32(),
            1,
            1,
        );
        let q = requantize(&acc, 0.02, true); // relu zeroes the negated channel
        let (pooled, ph, pw) = maxpool2d_i8(&q, 2, h, wd, 2, 2);
        assert_eq!((ph, pw), (2, 2));
        let fc = weights.get("fc.w").unwrap();
        let logits_i = dense_i8(
            &pooled,
            fc.as_i8(),
            2,
            &weights.get("fc.b").unwrap().as_i32(),
        );
        let preds = rank_logits(&dequantize_i32(&logits_i, 0.0001), 2);

        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].class, 0); // positive input ⇒ class 0
        assert!((preds.iter().map(|p| p.prob).sum::<f32>() - 1.0).abs() < 1e-4);
    }
}
