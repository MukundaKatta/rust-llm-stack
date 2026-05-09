//! `stopstream` — streaming-safe stop-sequence detector for LLM token streams.
//!
//! The naive solution is `if buffer.contains(stop) { stop_streaming() }`. That
//! works only after the entire stop sequence lands inside one chunk. Real
//! providers stream tokens of arbitrary boundaries, so a stop like `"</answer>"`
//! arrives split across multiple chunks. This crate buffers exactly enough
//! tail to detect the stop without ever emitting a partial match downstream.
//!
//! # Example
//!
//! ```
//! use stopstream::StopDetector;
//!
//! let mut det = StopDetector::new(["</answer>"]);
//! let r1 = det.push("Here is the response </ans");
//! assert!(r1.stopped.is_none());
//! assert_eq!(r1.safe_text, "Here is the response ");
//!
//! let r2 = det.push("wer> trailing");
//! assert_eq!(r2.stopped.as_deref(), Some("</answer>"));
//! // Anything after the stop is dropped.
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

/// What `push` returned for one chunk.
#[derive(Debug, Clone, Default)]
pub struct StopResult {
    /// Text that is safe to forward to the consumer (no partial stop-sequence match).
    pub safe_text: String,
    /// `Some(matched)` if a stop sequence was found in this chunk; emission stops here.
    pub stopped: Option<String>,
}

/// Watches a token stream for any of a set of stop sequences.
///
/// Stop sequences are matched on raw chars; the buffer is byte-aligned to UTF-8
/// boundaries so no multi-byte char is ever split.
pub struct StopDetector {
    sequences: Vec<String>,
    buffer: String,
    stopped: bool,
}

impl StopDetector {
    /// Create a detector watching for any of the given stop sequences.
    ///
    /// Empty input or empty sequences are filtered out (they would match every
    /// position and break the stream immediately).
    pub fn new<I, S>(sequences: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let sequences: Vec<String> = sequences
            .into_iter()
            .map(Into::into)
            .filter(|s| !s.is_empty())
            .collect();
        Self {
            sequences,
            buffer: String::new(),
            stopped: false,
        }
    }

    /// Feed one chunk. Returns the safe-to-emit prefix and (if hit) the stop seq.
    ///
    /// After a stop is hit, every subsequent push returns an empty result.
    pub fn push(&mut self, chunk: &str) -> StopResult {
        if self.stopped {
            return StopResult::default();
        }
        if self.sequences.is_empty() {
            // Nothing to watch for; pass through.
            return StopResult {
                safe_text: chunk.to_string(),
                stopped: None,
            };
        }
        self.buffer.push_str(chunk);

        // Look for any stop sequence anywhere in the buffer.
        if let Some((pos, seq)) = first_match(&self.buffer, &self.sequences) {
            let safe_text = self.buffer[..pos].to_string();
            self.stopped = true;
            self.buffer.clear();
            return StopResult {
                safe_text,
                stopped: Some(seq),
            };
        }

        // No full match yet. Hold back exactly the longest suffix of the buffer
        // that is also a *prefix* of some stop sequence — that's the only thing
        // that could complete a match on the next push. Snap to a UTF-8 char
        // boundary so we never split a multi-byte char.
        let hold = longest_suffix_prefix_overlap(&self.buffer, &self.sequences);
        let mut emit_end = self.buffer.len().saturating_sub(hold);
        while emit_end > 0 && !self.buffer.is_char_boundary(emit_end) {
            emit_end -= 1;
        }
        let safe_text = self.buffer[..emit_end].to_string();
        self.buffer = self.buffer[emit_end..].to_string();
        StopResult {
            safe_text,
            stopped: None,
        }
    }

    /// Flush the buffered tail at end-of-stream. Returns whatever's left if no
    /// stop ever hit. Empty after a stop has been seen.
    pub fn flush(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.buffer)
    }

    /// True after a stop sequence has been observed.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

/// Length (in bytes) of the longest suffix of `buffer` that is also a prefix of
/// any sequence in `needles`. This is the minimum amount we must hold back to
/// avoid emitting a partial stop match.
fn longest_suffix_prefix_overlap(buffer: &str, needles: &[String]) -> usize {
    let mut longest = 0;
    for n in needles {
        let max_k = n.len().min(buffer.len());
        // Walk down from the longest possible overlap to the shortest.
        for k in (1..=max_k).rev() {
            // Cheap byte comparison; stop sequences are usually ASCII anyway.
            if buffer.as_bytes()[buffer.len() - k..] == n.as_bytes()[..k] {
                if k > longest {
                    longest = k;
                }
                break;
            }
        }
    }
    longest
}

fn first_match(haystack: &str, needles: &[String]) -> Option<(usize, String)> {
    let mut best: Option<(usize, &String)> = None;
    for n in needles {
        if let Some(p) = haystack.find(n.as_str()) {
            best = match best {
                None => Some((p, n)),
                Some((bp, _)) if p < bp => Some((p, n)),
                Some(b) => Some(b),
            };
        }
    }
    best.map(|(p, n)| (p, n.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_match_in_one_chunk() {
        let mut d = StopDetector::new(["STOP"]);
        let r = d.push("hello STOP trailing");
        assert_eq!(r.safe_text, "hello ");
        assert_eq!(r.stopped.as_deref(), Some("STOP"));
        assert!(d.is_stopped());
    }

    #[test]
    fn match_split_across_chunks() {
        let mut d = StopDetector::new(["</done>"]);
        let r1 = d.push("alpha </do");
        assert!(r1.stopped.is_none());
        // "alpha " is safe; "</do" is held back.
        assert_eq!(r1.safe_text, "alpha ");

        let r2 = d.push("ne> tail");
        assert_eq!(r2.stopped.as_deref(), Some("</done>"));
        assert_eq!(r2.safe_text, "");
    }

    #[test]
    fn no_partial_overlap_emits_everything() {
        // "hello world" doesn't end with any prefix of "STOP", so we hold nothing.
        let mut d = StopDetector::new(["STOP"]);
        let r1 = d.push("hello world");
        assert!(r1.stopped.is_none());
        assert_eq!(r1.safe_text, "hello world");
        let tail = d.flush();
        assert_eq!(tail, "");
    }

    #[test]
    fn partial_overlap_holds_only_what_could_complete() {
        // "abcSTO" ends with "STO" which is a 3-byte prefix of "STOP".
        // We must hold "STO" and emit "abc".
        let mut d = StopDetector::new(["STOP"]);
        let r1 = d.push("abcSTO");
        assert!(r1.stopped.is_none());
        assert_eq!(r1.safe_text, "abc");
        // Now if "P more" arrives, "STOP" forms and stops fire.
        let r2 = d.push("P more");
        assert_eq!(r2.stopped.as_deref(), Some("STOP"));
        assert_eq!(r2.safe_text, "");
    }

    #[test]
    fn multiple_sequences_first_one_wins() {
        let mut d = StopDetector::new(["</a>", "</b>"]);
        let r = d.push("hi </b> mid </a> end");
        // "</b>" appears first.
        assert_eq!(r.stopped.as_deref(), Some("</b>"));
        assert_eq!(r.safe_text, "hi ");
    }

    #[test]
    fn no_stops_passes_through() {
        let mut d = StopDetector::new::<_, &str>([]);
        let r = d.push("anything goes here");
        assert_eq!(r.safe_text, "anything goes here");
        assert!(r.stopped.is_none());
        assert!(d.flush().is_empty());
    }

    #[test]
    fn empty_sequences_filtered_out() {
        let mut d = StopDetector::new(["", "STOP", ""]);
        let r = d.push("text STOP after");
        assert_eq!(r.stopped.as_deref(), Some("STOP"));
    }

    #[test]
    fn after_stop_pushes_return_empty() {
        let mut d = StopDetector::new(["X"]);
        let _ = d.push("aaXbb");
        let r = d.push("more text");
        assert_eq!(r.safe_text, "");
        assert!(r.stopped.is_none());
    }

    #[test]
    fn multibyte_chars_not_split() {
        // "</done>" is 7 bytes. Push a chunk ending in a 2-byte char near the boundary.
        let mut d = StopDetector::new(["</done>"]);
        let r = d.push("héllo</do");
        // "héllo" is safe; "</do" held. No char split panics.
        assert!(r.stopped.is_none());
        assert!(r.safe_text.starts_with("héllo"));
    }
}
