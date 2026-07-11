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
    // `symbolId` (detexify-next) / `key` (cloudant export). NOT bare `id` — that's
    // the *sample* id in both formats, not the class.
    let class = ["symbolId", "key", "symbol"]
        .iter()
        .find_map(|k| {
            rec.get(*k)
                .or_else(|| inner.get(*k))
                .and_then(Value::as_str)
        })?
        .to_string();
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
    let x = num(p.get("x")?)? as f32;
    let y = num(p.get("y")?)? as f32;
    let t = p.get("t").and_then(num).unwrap_or(0.0) as i64;
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
        assert_eq!(s[0].class, "amssymb-OT1-_bigstar");
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
    fn rejects_garbage() {
        assert!(parse("not json at all").is_err());
    }
}
