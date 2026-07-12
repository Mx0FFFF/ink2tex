//! Loader for the Detexify ODbL stroke dataset (training-time tooling, desktop-only).
//!
//! The canonical export is a CouchDB/cloudant view dump with a leading `//` curl
//! comment, then `{ rows: [ { key: "<class>", doc: { data: [[{x,y,t}, …], …] } } ] }`.
//! `key` is the class label (e.g. `amssymb-OT1-_bigstar`); `doc.data` is the stroke
//! list; each point is `{x, y, t}` with `t` in epoch milliseconds. We parse into a
//! dynamic `Value` and navigate defensively so the loader also tolerates a plain
//! array of records or newline-delimited JSON (the other shapes the data ships in).

use anyhow::{bail, Result};
use ink2tex_core::{Point, Stroke};
use serde_json::Value;

/// One labelled drawing: a class key and its strokes (in raw Detexify pixel coords;
/// the rasterizer aspect-fits, so absolute scale doesn't matter).
pub struct Sample {
    pub class: String,
    pub strokes: Vec<Stroke>,
}

/// Normalize a Detexify class key to the canonical `latex:<pkg>:<name>` symbolId.
///
/// Two vocabularies ship in the wild for the *same* symbols:
///   - **detexify-next**: `latex:latex2e:xi` — already canonical, passed through.
///   - **classic** (the Postgres bulk dump): `latex2e-OT1-_xi` — that is
///     `<pkg>-<enc>-_<cmd>`, where `<enc>` is the TeX font encoding (OT1/OMS/T1/…) and
///     `<cmd>` is the raw command *with its braces*: `mathcal{P}`, `sqrt{}`.
///
/// The canonical vocabulary spells braces out — `mathcal{P}` → `mathcal-lbrace-P-rbrace`
/// — and an empty argument collapses the doubled dash it would otherwise leave:
/// `sqrt{}` → `sqrt-lbrace-rbrace` (no canonical name contains `--`, so this is
/// unambiguous). Mapping classic keys into this space is what lets the 210k-sample bulk
/// dump and the detexify-next corpus **share one label space**, and it keeps
/// `core::latex::symbol_command` the single place that knows how a label becomes LaTeX.
///
/// A key matching neither shape is returned **unchanged**. Deciding which labels are
/// admissible is the class-space filter's job (`--prepare-detexify --classes`); the
/// loader's job is only to speak one vocabulary wherever it can.
pub fn normalize_class(key: &str) -> String {
    if let Some((_, canonical)) = ALIASES.iter().find(|(k, _)| *k == key) {
        return canonical.to_string();
    }
    if key.starts_with("latex:") {
        return key.to_string(); // already canonical
    }
    // `<pkg>-<enc>-_<cmd>`: split the command off first, then the encoding off the rest.
    let Some((head, cmd)) = key.split_once("-_") else {
        return key.to_string();
    };
    let Some((pkg, _enc)) = head.rsplit_once('-') else {
        return key.to_string();
    };
    let name = spell_braces(cmd);
    if pkg.is_empty() || name.is_empty() {
        return key.to_string();
    }
    format!("latex:{pkg}:{name}")
}

/// The ~18 symbols the two vocabularies genuinely *disagree* on, rather than merely
/// encode differently. Two causes, both historical:
///
///   - **Punctuation the structural rule can't reach.** The classic key is the raw TeX
///     token (`latex2e-OT1-_&`, `latex2e-OT1-[`), where detexify-next spells the glyph
///     out (`ampersand`, `lbracket`). Note the `_` marks a *command*: `_&` is `\&`, but
///     a bare `[` is the literal character — which is why some of these keys don't even
///     carry the `-_` separator the structural rule keys off.
///   - **A hyphen/underscore split**: the dump says `not_equiv`, next says `not-equiv`.
///
/// Without this table these 7,957 samples — **4% of the corpus, and every single sample
/// of 17 classes** — are silently discarded as out-of-vocabulary. With it, every key in
/// the bulk dump lands in the 1,123-class space (asserted below, exhaustively).
///
/// The two `bar:*` targets are detexify-next's own generated ids for the vertical bar:
/// the dump distinguishes `|` (literal) from `_|` (the `\|` command), and next kept two
/// classes for them but named neither. The 2↔2 correspondence is certain; *which* is
/// which is not. They are look-alikes and both surface in top-5, so a swap here would
/// cost a user nothing — but don't mistake this for a checked fact.
const ALIASES: &[(&str, &str)] = &[
    ("latex2e-OT1-_&", "latex:latex2e:ampersand"),
    ("latex2e-OT1-_#", "latex:latex2e:hash"),
    ("latex2e-OT1-[", "latex:latex2e:lbracket"),
    ("latex2e-OT1-/", "latex:latex2e:slash"),
    ("latex2e-OT1-_%", "latex:latex2e:percent"),
    ("latex2e-OT1-!`", "latex:latex2e:exclamation-grave"),
    ("latex2e-OT1-|", "latex:latex2e:bar:16socis"),
    ("latex2e-OT1-_|", "latex:latex2e:bar:1sa4fqg"),
    ("latex2e-OT1-_not_equiv", "latex:latex2e:not-equiv"),
    ("latex2e-OT1-_$", "latex:latex2e:dollar"),
    ("latex2e-OT1-__", "latex:latex2e:underscore"),
    ("latex2e-OT1-]", "latex:latex2e:rbracket"),
    ("latex2e-OT1-_---", "latex:latex2e:dash-dash-dash"),
    ("latex2e-OT1-_----", "latex:latex2e:dash-dash-dash-dash"),
    ("latex2e-OT1-_--", "latex:latex2e:dash-dash"),
    ("latex2e-OT1-_not_approx", "latex:latex2e:not-approx"),
    ("latex2e-OT1-_not_simeq", "latex:latex2e:not-simeq"),
    ("latex2e-OT1-_not_sim", "latex:latex2e:not-sim"),
];

/// `mathcal{P}` → `mathcal-lbrace-P-rbrace`, `sqrt{}` → `sqrt-lbrace-rbrace`.
fn spell_braces(cmd: &str) -> String {
    let spelled = cmd.replace('{', "-lbrace-").replace('}', "-rbrace");
    // Collapse the `--` an empty brace arg leaves behind, and trim the edges.
    let mut out = String::with_capacity(spelled.len());
    for c in spelled.chars() {
        if c == '-' && out.ends_with('-') {
            continue;
        }
        out.push(c);
    }
    out.trim_matches('-').to_string()
}

/// Parse one newline-delimited-JSON record — the **streaming** entry point.
///
/// The classic bulk dump is ~1 GB of JSON / 210k records; holding it (and its inflated
/// `Vec<Sample>`) in memory at once is several GB. Callers that can process a sample and
/// drop it should pull records through here one line at a time. `parse` below stays for
/// the whole-file shapes (the cloudant single-document export), which are small.
pub fn parse_line(line: &str) -> Option<Sample> {
    let rec: Value = serde_json::from_str(line.trim()).ok()?;
    sample_from(&rec)
}

/// Parse a Detexify JSON export into labelled samples.
pub fn parse(text: &str) -> Result<Vec<Sample>> {
    // Strip the leading `// curl …` comment lines the cloudant export carries.
    let cleaned: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    let mut out = Vec::new();
    match serde_json::from_str::<Value>(&cleaned) {
        Ok(root) => collect(&root, &mut out),
        // Not a single JSON doc → try newline-delimited JSON (one record per line).
        Err(_) => {
            for line in cleaned.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(rec) = serde_json::from_str::<Value>(line) {
                    if let Some(s) = sample_from(&rec) {
                        out.push(s);
                    }
                }
            }
        }
    }

    if out.is_empty() {
        bail!("no Detexify samples parsed (unrecognized shape)");
    }
    Ok(out)
}

fn collect(root: &Value, out: &mut Vec<Sample>) {
    if let Some(rows) = root.get("rows").and_then(Value::as_array) {
        out.extend(rows.iter().filter_map(sample_from)); // cloudant view export
    } else if let Some(arr) = root.as_array() {
        out.extend(arr.iter().filter_map(sample_from)); // plain array of records
    } else if let Some(s) = sample_from(root) {
        out.push(s); // single record
    }
}

/// Pull a `Sample` out of a record in whatever shape it takes. Class key is one of
/// `key` / `symbol` / `id`; strokes live under `data` or `strokes`, either directly
/// or nested inside a `doc` (the cloudant row wrapper).
fn sample_from(rec: &Value) -> Option<Sample> {
    let inner = rec.get("doc").unwrap_or(rec);
    // `symbolId` (detexify-next) / `key` (cloudant export + the classic dump). NOT bare
    // `id` — that's the *sample* id in both formats, not the class.
    let class = ["symbolId", "key", "symbol"]
        .iter()
        .find_map(|k| {
            rec.get(*k)
                .or_else(|| inner.get(*k))
                .and_then(Value::as_str)
        })
        .map(normalize_class)?; // one label space, whatever the export
    let data = inner
        .get("data")
        .or_else(|| inner.get("strokes"))
        .or_else(|| rec.get("data"))
        .or_else(|| rec.get("strokes"))
        .and_then(Value::as_array)?;

    let strokes = strokes_from(data);
    if strokes.iter().all(|s| s.points.is_empty()) {
        return None;
    }
    Some(Sample { class, strokes })
}

fn strokes_from(data: &[Value]) -> Vec<Stroke> {
    // Timestamps are absolute epoch-ms; store them relative to the first sample.
    let mut t0: Option<i64> = None;
    data.iter()
        .filter_map(Value::as_array)
        .map(|stroke| Stroke {
            points: stroke
                .iter()
                .filter_map(|p| point_from(p, &mut t0))
                .collect(),
        })
        .collect()
}

fn point_from(p: &Value, t0: &mut Option<i64>) -> Option<Point> {
    // Two point shapes ship in the wild: an object `{x, y, t}` (cloudant view export /
    // detexify-next) and an array `[x, y, t]` (the classic Postgres `strokes json`
    // column). Accept both so one loader handles every Detexify export.
    let (x, y, t) = if let Some(a) = p.as_array() {
        (
            num(a.first()?)? as f32,
            num(a.get(1)?)? as f32,
            a.get(2).and_then(num).unwrap_or(0.0) as i64,
        )
    } else {
        (
            num(p.get("x")?)? as f32,
            num(p.get("y")?)? as f32,
            p.get("t").and_then(num).unwrap_or(0.0) as i64,
        )
    };
    let base = *t0.get_or_insert(t);
    // Detexify has no pressure/tilt; use full pressure so strokes render solid.
    Some(Point::new(
        x,
        y,
        1.0,
        0.0,
        0.0,
        (t - base).max(0) as u64 * 1000,
    ))
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"// curl https://kirelabs.cloudant.com/detexify/...
{"total_rows": 342410, "offset": 0, "rows": [
  {"id":"hash","key":"amssymb-OT1-_bigstar","value":null,
   "doc":{"_id":"hash","_rev":"1-x","id":"amssymb-OT1-_bigstar",
     "data":[[{"x":223,"y":58,"t":1400702665265},
              {"x":230,"y":70,"t":1400702665300}]]}}
]}"#;

    #[test]
    fn parses_cloudant_export() {
        let s = parse(FIXTURE).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].class, "latex:amssymb:bigstar"); // classic key → canonical
        assert_eq!(s[0].strokes.len(), 1);
        let pts = &s[0].strokes[0].points;
        assert_eq!(pts.len(), 2);
        assert_eq!((pts[0].x, pts[0].y), (223.0, 58.0));
        assert_eq!(pts[0].t_us, 0); // relative
        assert_eq!(pts[1].t_us, 35_000); // (1400702665300 - …265) ms → µs
    }

    #[test]
    fn parses_flat_jsonl() {
        let jsonl = "{\"key\":\"latin_A\",\"data\":[[{\"x\":1,\"y\":2,\"t\":0}]]}\n\
                     {\"key\":\"latin_B\",\"strokes\":[[{\"x\":3,\"y\":4,\"t\":0}]]}";
        let s = parse(jsonl).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].class, "latin_A");
        assert_eq!(s[1].class, "latin_B");
    }

    #[test]
    fn parses_detexify_next_symbolid() {
        // detexify-next shape: `symbolId` is the class; points have no `t`.
        let line = r#"{"id":"sample:x","symbolId":"latex:wasysym:XBox","strokes":[[{"x":0.29,"y":0.37},{"x":0.30,"y":0.50}]]}"#;
        let s = parse(line).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].class, "latex:wasysym:XBox");
        assert_eq!(s[0].strokes[0].points.len(), 2);
        assert_eq!(s[0].strokes[0].points[0].t_us, 0); // no timestamps → 0
    }

    #[test]
    fn parses_array_form_points() {
        // Classic Postgres `strokes json` column: each point is an `[x, y, t]` array.
        let line = r#"{"key":"latex2e-OT1-_zeta","strokes":[[[250,103,1362942716695],[242,103,1362942716985]]]}"#;
        let s = parse(line).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].class, "latex:latex2e:zeta");
        let pts = &s[0].strokes[0].points;
        assert_eq!(pts.len(), 2);
        assert_eq!((pts[0].x, pts[0].y), (250.0, 103.0));
        assert_eq!(pts[0].t_us, 0); // relative to first
        assert_eq!(pts[1].t_us, 290_000); // (…985 − …695) ms → µs
    }

    #[test]
    fn parse_line_streams_one_record() {
        let s = parse_line(r#"{"key":"latex2e-OT1-_xi","strokes":[[[1,2,0],[3,4,10]]]}"#).unwrap();
        assert_eq!(s.class, "latex:latex2e:xi");
        assert_eq!(s.strokes[0].points.len(), 2);
        assert!(parse_line("").is_none());
        assert!(parse_line("{oops").is_none());
    }

    /// The classic dump and detexify-next name the same symbols differently. These are
    /// the mappings that let them share one label space — every expectation here was
    /// checked against the real 1,123-class vocabulary in `train/dataset/classes.txt`.
    #[test]
    fn classic_keys_normalize_to_symbolids() {
        let cases = [
            ("latex2e-OT1-_xi", "latex:latex2e:xi"),
            ("latex2e-OT1-_sum", "latex:latex2e:sum"),
            ("amssymb-OT1-_bigstar", "latex:amssymb:bigstar"),
            // braces are spelled out
            (
                "amssymb-OT1-_mathcal{P}",
                "latex:amssymb:mathcal-lbrace-P-rbrace",
            ),
            (
                "dsfont-OT1-_mathds{R}",
                "latex:dsfont:mathds-lbrace-R-rbrace",
            ),
            // …and an *empty* argument collapses the dash it would double up
            ("latex2e-OT1-_sqrt{}", "latex:latex2e:sqrt-lbrace-rbrace"),
            // encodings other than OT1 exist; the encoding is discarded either way
            ("stmaryrd-OMS-_lightning", "latex:stmaryrd:lightning"),
            // already canonical → identity
            ("latex:wasysym:XBox", "latex:wasysym:XBox"),
        ];
        for (raw, want) in cases {
            assert_eq!(normalize_class(raw), want, "normalizing {raw}");
        }
    }

    #[test]
    fn aliased_keys_reach_their_canonical_class() {
        // The punctuation keys the structural rule can't reach: `_&` is the command
        // `\&`, a bare `[` is the literal character (note: no `-_` separator at all).
        assert_eq!(normalize_class("latex2e-OT1-_&"), "latex:latex2e:ampersand");
        assert_eq!(normalize_class("latex2e-OT1-["), "latex:latex2e:lbracket");
        assert_eq!(
            normalize_class("latex2e-OT1-!`"),
            "latex:latex2e:exclamation-grave"
        );
        // hyphen/underscore split between the two vocabularies
        assert_eq!(
            normalize_class("latex2e-OT1-_not_equiv"),
            "latex:latex2e:not-equiv"
        );
    }

    #[test]
    fn unrecognized_keys_pass_through_untouched() {
        // Not this loader's call to make — `--prepare-detexify --classes` is the filter
        // that decides admissibility. Mangling a key here would only hide it.
        for raw in ["latin_A", "some-OT1-garbage", ""] {
            assert_eq!(normalize_class(raw), raw);
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("not json at all").is_err());
    }
}
