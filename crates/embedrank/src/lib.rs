//! `embedrank` — batched cosine / dot / L2 distance for f32 embeddings, plus a
//! min-heap top-k selector. Pure-Rust, no BLAS, no allocator surprises.
//!
//! Designed for the hot path of small-to-medium RAG retrieval (up to ~100k
//! candidates × ~768 dims). For larger corpora reach for a vector index.
//!
//! # Example
//!
//! ```
//! use embedrank::{top_k_cosine};
//!
//! let query = vec![1.0_f32, 0.0, 0.0];
//! let candidates = vec![
//!     vec![1.0_f32, 0.0, 0.0],   // identical, score 1.0
//!     vec![0.0_f32, 1.0, 0.0],   // orthogonal, score 0.0
//!     vec![0.7_f32, 0.7, 0.0],   // ~45 degrees, score ~0.707
//! ];
//! let top = top_k_cosine(&query, &candidates, 2);
//! assert_eq!(top[0].1, 0); // best match is index 0
//! assert_eq!(top[1].1, 2); // second is index 2
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Cosine similarity in `[-1, 1]`. Returns `0.0` if either input has zero norm.
///
/// # Panics
///
/// Panics in debug builds if `a.len() != b.len()`. Release builds will produce
/// a partial product over the shorter slice; pass equal-length inputs.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine: inputs must share length");
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Plain dot product. Faster than cosine when your vectors are already L2-normalized.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: inputs must share length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared L2 (Euclidean) distance. Skip the sqrt for ranking — order is preserved.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2_squared: inputs must share length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Full Euclidean distance.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Normalize `v` to unit L2 in place. No-op if `v` has zero norm.
pub fn normalize_inplace(v: &mut [f32]) {
    let mut s = 0.0_f32;
    for x in v.iter() {
        s += x * x;
    }
    let n = s.sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// A bounded min-heap that keeps the top-k highest-scoring (score, index) pairs.
///
/// Push items as you produce them; only the best `k` are retained. `O(log k)`
/// per push. `into_sorted` returns the survivors in descending order of score.
pub struct TopK {
    k: usize,
    heap: BinaryHeap<MinScored>,
}

#[derive(Debug, Clone, Copy)]
struct MinScored {
    score: f32,
    idx: usize,
}

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.idx == other.idx
    }
}
impl Eq for MinScored {}
impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. To make the *smallest* score sit at the
        // top (so it's the cheapest to evict), reverse the comparison.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl TopK {
    /// Create a top-k selector. `k = 0` produces a selector that retains nothing.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Insert a `(score, idx)` pair. Higher scores win.
    pub fn push(&mut self, score: f32, idx: usize) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(MinScored { score, idx });
        } else if let Some(top) = self.heap.peek() {
            // top is the smallest score in the heap (because of reversed Ord).
            if score > top.score {
                self.heap.pop();
                self.heap.push(MinScored { score, idx });
            }
        }
    }

    /// Number of items currently retained.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// True if no items have been pushed (or `k` was zero).
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Consume the selector and return `(score, idx)` pairs sorted by score descending.
    pub fn into_sorted(self) -> Vec<(f32, usize)> {
        let mut out: Vec<(f32, usize)> = self.heap.into_iter().map(|m| (m.score, m.idx)).collect();
        out.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        out
    }
}

/// Convenience: top-k cosine matches between `query` and a slice of candidates.
///
/// Returns up to `k` `(score, candidate_index)` pairs in descending score order.
pub fn top_k_cosine(query: &[f32], candidates: &[Vec<f32>], k: usize) -> Vec<(f32, usize)> {
    let mut tk = TopK::new(k);
    for (i, c) in candidates.iter().enumerate() {
        tk.push(cosine(query, c), i);
    }
    tk.into_sorted()
}

/// Convenience: top-k by plain dot product (use when vectors are pre-normalized).
pub fn top_k_dot(query: &[f32], candidates: &[Vec<f32>], k: usize) -> Vec<(f32, usize)> {
    let mut tk = TopK::new(k);
    for (i, c) in candidates.iter().enumerate() {
        tk.push(dot(query, c), i);
    }
    tk.into_sorted()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn cosine_opposite_is_minus_one() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        assert!((cosine(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm_is_zero() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn dot_works() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert_eq!(dot(&a, &b), 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn l2_known_value() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        assert!((l2(&a, &b) - 5.0).abs() < 1e-6);
        assert!((l2_squared(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_unit_vector() {
        let mut v = vec![3.0_f32, 4.0];
        normalize_inplace(&mut v);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn topk_picks_highest_n() {
        let mut t = TopK::new(3);
        for (i, &s) in [0.1_f32, 0.9, 0.5, 0.3, 0.8, 0.2].iter().enumerate() {
            t.push(s, i);
        }
        let out = t.into_sorted();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].1, 1); // 0.9
        assert_eq!(out[1].1, 4); // 0.8
        assert_eq!(out[2].1, 2); // 0.5
    }

    #[test]
    fn topk_zero_k_keeps_nothing() {
        let mut t = TopK::new(0);
        t.push(1.0, 0);
        t.push(2.0, 1);
        assert!(t.is_empty());
        assert!(t.into_sorted().is_empty());
    }

    #[test]
    fn topk_fewer_pushes_than_k() {
        let mut t = TopK::new(5);
        t.push(0.7, 0);
        t.push(0.4, 1);
        let out = t.into_sorted();
        assert_eq!(out, vec![(0.7, 0), (0.4, 1)]);
    }

    #[test]
    fn top_k_cosine_returns_descending() {
        let q = vec![1.0_f32, 0.0, 0.0];
        let cs = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 1.0, 0.0],
            vec![0.7_f32, 0.7, 0.0],
        ];
        let top = top_k_cosine(&q, &cs, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].1, 0);
        assert_eq!(top[1].1, 2);
    }

    #[test]
    fn top_k_cosine_k_larger_than_candidates_returns_all() {
        let q = vec![1.0_f32, 0.0];
        let cs = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let top = top_k_cosine(&q, &cs, 10);
        assert_eq!(top.len(), 2);
    }
}
