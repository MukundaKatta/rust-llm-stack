//! `promptbudget` — token-budget-aware text truncation with multiple strategies.
//!
//! Bring-your-own tokenizer (or use the included `CharTokenizer` proxy). All
//! strategies guarantee the result fits within `max_tokens` according to the
//! tokenizer you pass.
//!
//! # Example
//!
//! ```
//! use promptbudget::{fit, CharTokenizer, Strategy, Tokenizer};
//!
//! let long = "a".repeat(1000);  // ~250 tokens at 4 chars/token
//! let tok = CharTokenizer::default();
//! let trimmed = fit(&long, 50, Strategy::Tail, &tok);
//! assert!(tok.count(&trimmed) <= 50);
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

/// Anything that can count and truncate by tokens.
///
/// Implement against your real tokenizer (tiktoken, BPE, sentencepiece, etc.).
/// The crate ships `CharTokenizer` as a zero-dep proxy for sketching.
pub trait Tokenizer {
    /// Number of tokens in `s`.
    fn count(&self, s: &str) -> usize;

    /// Truncate `s` to at most `max_tokens` tokens, taking from the *front*.
    /// Implementations should return the longest prefix that fits.
    fn truncate_head(&self, s: &str, max_tokens: usize) -> String;

    /// Truncate `s` to at most `max_tokens` tokens, taking from the *back*.
    /// Implementations should return the longest suffix that fits.
    fn truncate_tail(&self, s: &str, max_tokens: usize) -> String;
}

/// A coarse "1 token ≈ N chars" proxy. Useful for sketches and tests; not for
/// production accounting against a real model.
#[derive(Debug, Clone, Copy)]
pub struct CharTokenizer {
    /// Characters per token. Default 4 (matches OpenAI's rule-of-thumb).
    pub chars_per_token: usize,
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self { chars_per_token: 4 }
    }
}

impl Tokenizer for CharTokenizer {
    fn count(&self, s: &str) -> usize {
        // Use char count, not byte count, so multibyte text isn't over-counted.
        let chars = s.chars().count();
        chars.div_ceil(self.chars_per_token.max(1))
    }
    fn truncate_head(&self, s: &str, max_tokens: usize) -> String {
        let max_chars = max_tokens.saturating_mul(self.chars_per_token.max(1));
        s.chars().take(max_chars).collect()
    }
    fn truncate_tail(&self, s: &str, max_tokens: usize) -> String {
        let total = s.chars().count();
        let max_chars = max_tokens.saturating_mul(self.chars_per_token.max(1));
        let skip = total.saturating_sub(max_chars);
        s.chars().skip(skip).collect()
    }
}

/// How to drop tokens when text is over budget.
#[derive(Debug, Clone, Copy)]
pub enum Strategy {
    /// Keep the first `max_tokens` tokens. Drops the tail.
    Head,
    /// Keep the last `max_tokens` tokens. Drops the head. Good for chat history.
    Tail,
    /// Keep `head_ratio * max_tokens` from the head and the rest from the tail.
    /// Drops the middle. Useful when the start (instructions) and end (latest
    /// turn) both matter.
    HeadTail {
        /// Fraction of the budget reserved for the head (0.0–1.0).
        head_ratio: f32,
    },
    /// Like `HeadTail` but joins with a visible ellipsis marker so the model
    /// (and humans) can see the cut.
    SmartCut {
        /// Fraction reserved for the head (0.0–1.0).
        head_ratio: f32,
        /// Marker inserted between head and tail. `"\n[...]\n"` is a sane default.
        marker: &'static str,
    },
}

/// Fit `text` into at most `max_tokens` tokens using `strategy` and `tok`.
///
/// If the text already fits, returns it unchanged.
pub fn fit<T: Tokenizer>(text: &str, max_tokens: usize, strategy: Strategy, tok: &T) -> String {
    if tok.count(text) <= max_tokens {
        return text.to_string();
    }
    match strategy {
        Strategy::Head => tok.truncate_head(text, max_tokens),
        Strategy::Tail => tok.truncate_tail(text, max_tokens),
        Strategy::HeadTail { head_ratio } => {
            let (h, t) = split_budget(max_tokens, head_ratio);
            let head = tok.truncate_head(text, h);
            let tail = tok.truncate_tail(text, t);
            format!("{head}{tail}")
        }
        Strategy::SmartCut { head_ratio, marker } => {
            let marker_tokens = tok.count(marker);
            let usable = max_tokens.saturating_sub(marker_tokens);
            let (h, t) = split_budget(usable, head_ratio);
            let head = tok.truncate_head(text, h);
            let tail = tok.truncate_tail(text, t);
            format!("{head}{marker}{tail}")
        }
    }
}

fn split_budget(total: usize, head_ratio: f32) -> (usize, usize) {
    let r = head_ratio.clamp(0.0, 1.0);
    let h = ((total as f32) * r).round() as usize;
    let t = total.saturating_sub(h);
    (h, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_tokenizer_count_known() {
        let t = CharTokenizer::default();
        assert_eq!(t.count(""), 0);
        assert_eq!(t.count("abcd"), 1);
        assert_eq!(t.count("abcde"), 2); // 5 chars / 4 = 1.25 -> 2
    }

    #[test]
    fn fit_passes_through_when_under_budget() {
        let t = CharTokenizer::default();
        let s = "short";
        assert_eq!(fit(s, 100, Strategy::Head, &t), s);
    }

    #[test]
    fn head_keeps_prefix() {
        let t = CharTokenizer::default();
        let s = "0123456789abcdef";
        let out = fit(s, 2, Strategy::Head, &t);
        assert_eq!(out, "01234567"); // 8 chars = 2 tokens
        assert!(t.count(&out) <= 2);
    }

    #[test]
    fn tail_keeps_suffix() {
        let t = CharTokenizer::default();
        let s = "0123456789abcdef";
        let out = fit(s, 2, Strategy::Tail, &t);
        assert_eq!(out, "89abcdef");
        assert!(t.count(&out) <= 2);
    }

    #[test]
    fn head_tail_drops_middle() {
        let t = CharTokenizer::default();
        let s: String = (b'A'..=b'Z').map(|c| c as char).collect();
        // 26 chars ≈ 7 tokens. Budget 4 tokens with 50/50 split = 2/2.
        let out = fit(&s, 4, Strategy::HeadTail { head_ratio: 0.5 }, &t);
        assert!(t.count(&out) <= 4);
        assert!(out.starts_with("ABCDEFGH"));
        assert!(out.ends_with("STUVWXYZ"));
    }

    #[test]
    fn smart_cut_inserts_marker() {
        let t = CharTokenizer::default();
        let s = "X".repeat(200); // ~50 tokens
        let out = fit(
            &s,
            10,
            Strategy::SmartCut {
                head_ratio: 0.5,
                marker: " ... ",
            },
            &t,
        );
        assert!(out.contains(" ... "));
        assert!(t.count(&out) <= 10);
    }

    #[test]
    fn multibyte_text_counted_by_chars_not_bytes() {
        let t = CharTokenizer::default();
        // "héllo" is 5 chars, 6 bytes
        assert_eq!(t.count("héllo"), 2); // 5/4 = 2
    }

    #[test]
    fn zero_budget_is_handled() {
        let t = CharTokenizer::default();
        let out = fit("abc", 0, Strategy::Head, &t);
        assert_eq!(out, "");
    }
}
