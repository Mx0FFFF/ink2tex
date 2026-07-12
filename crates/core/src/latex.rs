//! Symbol Layout Tree → LaTeX string (DESIGN §4.5). The structure parse is the hard
//! part; emission is a straightforward recursive walk.

use crate::structure::{Base, Slt, Term};

/// Render an SLT as LaTeX.
pub fn to_latex(slt: &Slt) -> String {
    slt.terms.iter().map(term).collect()
}

fn term(t: &Term) -> String {
    let mut s = base(&t.base);
    if let Some(sub) = &t.sub {
        s.push_str(&format!("_{{{}}}", to_latex(sub)));
    }
    if let Some(sup) = &t.sup {
        s.push_str(&format!("^{{{}}}", to_latex(sup)));
    }
    s
}

fn base(b: &Base) -> String {
    match b {
        Base::Symbol(l) => symbol_command(l),
        Base::Frac { num, den } => format!("\\frac{{{}}}{{{}}}", to_latex(num), to_latex(den)),
        Base::Sqrt(r) => format!("\\sqrt{{{}}}", to_latex(r)),
    }
}

/// Map a recognized label to a LaTeX token. A clean label (`x`, `+`, `\alpha`) is the
/// identity; a Detexify `symbolId` (`latex:<pkg>:<name>`) is reduced best-effort to a
/// command (`latex:latex2e:xi` → `\xi`, `latex:amssymb:mathcal-lbrace-R-rbrace` →
/// `\mathcal{R}`).
pub fn symbol_command(label: &str) -> String {
    match label
        .strip_prefix("latex:")
        .and_then(|rest| rest.split_once(':'))
    {
        Some((_pkg, name)) => detexify_name_to_latex(name),
        None => label.to_string(), // already a clean LaTeX token
    }
}

/// The classes detexify-next names by *spelling the glyph out*, and what they actually are.
///
/// `\ampersand` is not a LaTeX command. Neither is `\hash`, `\dollar` or `\bar:16socis`.
/// These are not obscure corners either: the alias table in `detexify.rs` recovered **7,957
/// samples** for exactly these classes, so the model predicts them regularly — and a user
/// who pastes `\ampersand` into a document gets an error, not an `&`.
const SPELLED_OUT: &[(&str, &str)] = &[
    ("ampersand", "\\&"),
    ("hash", "\\#"),
    ("dollar", "\\$"),
    ("percent", "\\%"),
    ("underscore", "\\_"),
    ("slash", "/"),
    ("lbracket", "["),
    ("rbracket", "]"),
    ("exclamation-grave", "!`"),
    ("dash-dash", "--"),
    ("dash-dash-dash", "---"),
    ("dash-dash-dash-dash", "----"),
    ("not-equiv", "\\not\\equiv"),
    ("not-approx", "\\not\\approx"),
    ("not-sim", "\\not\\sim"),
    ("not-simeq", "\\not\\simeq"),
    // detexify-next gave the two vertical bars generated ids and named neither.
    ("bar:16socis", "|"),
    ("bar:1sa4fqg", "\\|"),
];

fn detexify_name_to_latex(name: &str) -> String {
    if let Some((_, tex)) = SPELLED_OUT.iter().find(|(n, _)| *n == name) {
        return tex.to_string();
    }
    // e.g. "mathcal-lbrace-R-rbrace" → \mathcal{R}
    if let Some((cmd, rest)) = name.split_once("-lbrace-") {
        if let Some(arg) = rest.strip_suffix("-rbrace") {
            return format!("\\{cmd}{{{arg}}}");
        }
        // …and an *empty* argument, whose dash collapses: "sqrt-lbrace-rbrace" → \sqrt{}.
        // `\sqrt{}` is a real class with over a thousand samples; without this it emitted
        // the literal `\sqrt-lbrace-rbrace`.
        if rest == "rbrace" {
            return format!("\\{cmd}{{}}");
        }
    }
    // Everything still here is a Detexify symbolId's `name`, and every one of those is a
    // **command** — the single-character ones included. `latex:latex2e:L` is `\L` (Ł), not
    // the letter L; likewise `\O` `\P` `\S` `\l` `\o`. Those are the *only* six single-char
    // classes Detexify has, and treating them as literals (as this used to) both mis-rendered
    // them and put them on a collision course with the real letters HWRT now supplies.
    //
    // The letters and digits themselves never reach here: they are keyed by their literal
    // LaTeX (`x`, `7`, `+`) and `symbol_command` passes those straight through.
    format!("\\{name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The HWRT tokens Detexify cannot express. They are keyed by their literal LaTeX
    /// precisely so that they pass straight through — and so that they cannot collide with
    /// Detexify's `latex:latex2e:L`, which is the *command* `\L` (Ł), not the letter.
    #[test]
    fn the_tokens_hwrt_contributes_render_as_themselves() {
        for (label, want) in [
            ("+", "+"),
            ("-", "-"),
            ("<", "<"),
            (">", ">"),
            ("0", "0"),
            ("7", "7"),
            ("x", "x"),
            ("L", "L"),
        ] {
            assert_eq!(symbol_command(label), want, "token {label}");
        }
        // …and the six that would have collided stay the COMMANDS they always were.
        // Merging the letter `L` onto this key would have poisoned the class with two
        // different symbols — and rendering it as a bare `L` mis-typesets Ł.
        assert_eq!(symbol_command("latex:latex2e:L"), "\\L"); // Ł, not the letter L
        assert_eq!(symbol_command("latex:latex2e:o"), "\\o"); // ø, not the letter o
    }

    /// The spelled-out classes must emit LaTeX that actually compiles. `\ampersand` is not
    /// a command, and these have 7,957 samples behind them, so the model really does say them.
    #[test]
    fn spelled_out_punctuation_emits_real_latex() {
        for (label, want) in [
            ("latex:latex2e:ampersand", "\\&"),
            ("latex:latex2e:hash", "\\#"),
            ("latex:latex2e:dollar", "\\$"),
            ("latex:latex2e:percent", "\\%"),
            ("latex:latex2e:underscore", "\\_"),
            ("latex:latex2e:slash", "/"),
            ("latex:latex2e:lbracket", "["),
            ("latex:latex2e:not-equiv", "\\not\\equiv"),
            ("latex:latex2e:bar:16socis", "|"),
        ] {
            assert_eq!(symbol_command(label), want, "{label} must typeset");
        }
    }

    #[test]
    fn clean_labels_pass_through() {
        assert_eq!(symbol_command("x"), "x");
        assert_eq!(symbol_command("+"), "+");
        assert_eq!(symbol_command("\\alpha"), "\\alpha");
    }

    #[test]
    fn detexify_symbolids_reduce_to_commands() {
        assert_eq!(symbol_command("latex:latex2e:xi"), "\\xi");
        assert_eq!(symbol_command("latex:latex2e:Xi"), "\\Xi");
        assert_eq!(symbol_command("latex:latex2e:sum"), "\\sum");
        assert_eq!(
            symbol_command("latex:amssymb:mathcal-lbrace-R-rbrace"),
            "\\mathcal{R}"
        );
        // An empty brace argument collapses its dash in the class name. Miss this and a
        // class with >1,000 samples emits the literal `\sqrt-lbrace-rbrace`.
        assert_eq!(
            symbol_command("latex:latex2e:sqrt-lbrace-rbrace"),
            "\\sqrt{}"
        );
    }
}
