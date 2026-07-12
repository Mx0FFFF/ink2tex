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

fn detexify_name_to_latex(name: &str) -> String {
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
    // a single latin letter or digit stays literal; anything else is a command name.
    if name.len() == 1 && name.chars().all(|c| c.is_ascii_alphanumeric()) {
        name.to_string()
    } else {
        format!("\\{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
