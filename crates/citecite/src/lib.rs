//! `citecite` — citation-marker `[1] [2]` injector + parser for RAG outputs.
//!
//! Inject markers tied to source ids into model output, then parse them back
//! when post-processing. Round-trippable: `parse(inject(text, cs)) == cs`
//! given the same source mapping.
//!
//! # Example
//!
//! ```
//! use citecite::{inject, parse, strip, Citation, InjectAt};
//!
//! let body = "Anthropic was founded in 2021.";
//! let cited = inject(
//!     body,
//!     &[Citation { idx: 1, source_id: "wikipedia/anthropic".into() }],
//!     InjectAt::End,
//! );
//! assert_eq!(cited, "Anthropic was founded in 2021. [1]");
//!
//! let markers = parse(&cited);
//! assert_eq!(markers.len(), 1);
//! assert_eq!(markers[0].idx, 1);
//!
//! assert_eq!(strip(&cited), "Anthropic was founded in 2021. ");
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

/// One citation to attach to a piece of text. `idx` is the visible bracketed
/// number; `source_id` is the opaque key your RAG layer uses to resolve it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citation {
    /// 1-based index that will appear in the rendered marker `[idx]`.
    pub idx: usize,
    /// Opaque source identifier (URL, doc id, chunk id, etc.). Not rendered.
    pub source_id: String,
}

/// Where to put the markers when injecting.
#[derive(Debug, Clone, Copy)]
pub enum InjectAt {
    /// Append all markers at the end of the text, space-separated.
    End,
    /// Insert at the given byte position. Snaps to the nearest char boundary.
    Position(usize),
}

/// A parsed marker found in the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Marker {
    /// Byte position of the opening `[`.
    pub pos: usize,
    /// The number inside the brackets.
    pub idx: usize,
    /// Length of the rendered marker (e.g., `"[12]".len() == 4`).
    pub len: usize,
}

/// Inject one or more citation markers into `text`.
pub fn inject(text: &str, citations: &[Citation], at: InjectAt) -> String {
    if citations.is_empty() {
        return text.to_string();
    }
    let markers: Vec<String> = citations.iter().map(|c| format!("[{}]", c.idx)).collect();
    let joined = markers.join(" ");
    match at {
        InjectAt::End => {
            // Always separate markers from preceding text by exactly one space.
            if text.is_empty() {
                joined
            } else if text.ends_with(' ') {
                format!("{text}{joined}")
            } else {
                format!("{text} {joined}")
            }
        }
        InjectAt::Position(pos) => {
            let mut p = pos.min(text.len());
            while p > 0 && !text.is_char_boundary(p) {
                p -= 1;
            }
            let (before, after) = text.split_at(p);
            format!("{before}{joined}{after}")
        }
    }
}

/// Parse all `[N]` markers out of `text`. Returns positions in source order.
///
/// Only matches `[<digits>]` with no internal whitespace; `[abc]` is ignored.
pub fn parse(text: &str) -> Vec<Marker> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            i += 1;
            continue;
        }
        // Look for matching `]` within a small window of digits.
        let start = i;
        let mut j = i + 1;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > i + 1 && j < bytes.len() && bytes[j] == b']' {
            // Parse the digits between (i+1, j).
            if let Ok(n) = std::str::from_utf8(&bytes[i + 1..j])
                .unwrap_or("")
                .parse::<usize>()
            {
                out.push(Marker {
                    pos: start,
                    idx: n,
                    len: j + 1 - start,
                });
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Remove every `[N]` marker from `text`. Whitespace around removed markers is
/// not normalized — callers can re-collapse with their own rules.
pub fn strip(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let markers = parse(text);
    if markers.is_empty() {
        return text.to_string();
    }
    let mut cur = 0;
    for m in &markers {
        out.push_str(&text[cur..m.pos]);
        cur = m.pos + m.len;
    }
    out.push_str(&text[cur..]);
    // Trim the bytes safely; we only strip ASCII brackets/digits which are 1 byte each.
    let _ = bytes; // bytes used only for type hint
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cite(idx: usize, src: &str) -> Citation {
        Citation {
            idx,
            source_id: src.to_string(),
        }
    }

    #[test]
    fn inject_end_appends_marker() {
        let out = inject("hello", &[cite(1, "src")], InjectAt::End);
        assert_eq!(out, "hello [1]");
    }

    #[test]
    fn inject_end_multiple_markers_spaced() {
        let out = inject(
            "hello",
            &[cite(1, "a"), cite(2, "b"), cite(3, "c")],
            InjectAt::End,
        );
        assert_eq!(out, "hello [1] [2] [3]");
    }

    #[test]
    fn inject_at_position() {
        let out = inject("hello world", &[cite(7, "src")], InjectAt::Position(5));
        assert_eq!(out, "hello[7] world");
    }

    #[test]
    fn inject_empty_citations_unchanged() {
        let out = inject("hello", &[], InjectAt::End);
        assert_eq!(out, "hello");
    }

    #[test]
    fn parse_finds_markers() {
        let m = parse("foo [1] bar [2] baz");
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].idx, 1);
        assert_eq!(m[0].len, 3);
        assert_eq!(m[1].idx, 2);
    }

    #[test]
    fn parse_ignores_non_numeric_brackets() {
        let m = parse("see [abc] and [123]");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].idx, 123);
    }

    #[test]
    fn parse_handles_multidigit() {
        let m = parse("[42]");
        assert_eq!(m[0].idx, 42);
        assert_eq!(m[0].len, 4);
    }

    #[test]
    fn strip_removes_all_markers() {
        let out = strip("foo [1] bar [12] baz");
        assert_eq!(out, "foo  bar  baz");
    }

    #[test]
    fn round_trip_inject_then_parse() {
        let body = "Anthropic was founded in 2021.";
        let cs = vec![cite(1, "wiki/anthropic"), cite(2, "techcrunch/anth")];
        let cited = inject(body, &cs, InjectAt::End);
        let parsed = parse(&cited);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.iter().map(|m| m.idx).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn inject_position_snaps_to_char_boundary() {
        // "héllo" — 'é' is 2 bytes at position 1..3.
        let out = inject("héllo", &[cite(1, "src")], InjectAt::Position(2));
        // Position 2 is mid-é; should snap back to 1.
        assert_eq!(out, "h[1]éllo");
    }
}
