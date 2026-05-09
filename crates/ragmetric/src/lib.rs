//! `ragmetric` — IR metrics for RAG retrieval evaluation.
//!
//! Recall@k, Hit@k, MRR, and NDCG@k. Pure data ops, no model dependencies. All
//! metrics take retrieved doc IDs and the (binary) relevance set; identifiers
//! are arbitrary strings (typically chunk IDs or URLs).
//!
//! # Example
//!
//! ```
//! use ragmetric::{recall_at_k, mrr, ndcg_at_k};
//!
//! let retrieved = vec!["a".to_string(), "b".to_string(), "c".to_string()];
//! let relevant  = vec!["b".to_string(), "d".to_string()];
//!
//! assert_eq!(recall_at_k(&retrieved, &relevant, 3), 0.5);
//! assert!((mrr(&retrieved, &relevant) - 0.5).abs() < 1e-9);
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

use std::collections::HashSet;

/// Recall@k: fraction of the relevant set that appears in the top `k` retrieved.
///
/// `0.0` if `relevant` is empty (vacuous; no answer to find).
pub fn recall_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let rel: HashSet<&String> = relevant.iter().collect();
    let take = retrieved.len().min(k);
    let hits = retrieved[..take].iter().filter(|d| rel.contains(d)).count();
    hits as f64 / relevant.len() as f64
}

/// Hit@k: 1.0 if any relevant doc appears in the top `k`, else 0.0.
pub fn hit_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let rel: HashSet<&String> = relevant.iter().collect();
    let take = retrieved.len().min(k);
    if retrieved[..take].iter().any(|d| rel.contains(d)) {
        1.0
    } else {
        0.0
    }
}

/// Mean Reciprocal Rank for a single query.
///
/// MRR = `1 / rank_of_first_relevant`, or `0.0` if no relevant doc was retrieved.
/// Ranks are 1-based.
pub fn mrr(retrieved: &[String], relevant: &[String]) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let rel: HashSet<&String> = relevant.iter().collect();
    for (i, d) in retrieved.iter().enumerate() {
        if rel.contains(d) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// NDCG@k with **binary relevance** (gain = 1 if relevant, else 0).
///
/// DCG_k = sum_{i=1..k} rel_i / log2(i + 1)
/// IDCG_k = DCG_k of the optimal ordering.
/// Returns `DCG_k / IDCG_k` (or `0.0` if the ideal DCG is zero).
pub fn ndcg_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }
    let rel: HashSet<&String> = relevant.iter().collect();
    let take = retrieved.len().min(k);

    let mut dcg = 0.0_f64;
    for (i, d) in retrieved[..take].iter().enumerate() {
        if rel.contains(d) {
            dcg += 1.0 / ((i as f64 + 2.0).log2()); // log2(i+1+1)
        }
    }

    // IDCG: as many 1s as possible at the top, capped by relevant count and k.
    let n_ideal = relevant.len().min(k);
    let mut idcg = 0.0_f64;
    for i in 0..n_ideal {
        idcg += 1.0 / ((i as f64 + 2.0).log2());
    }
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Aggregated mean of each metric across a batch of queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggregateMetrics {
    /// Mean Recall@k.
    pub mean_recall_at_k: f64,
    /// Mean Hit@k.
    pub mean_hit_at_k: f64,
    /// Mean Reciprocal Rank.
    pub mean_mrr: f64,
    /// Mean NDCG@k.
    pub mean_ndcg_at_k: f64,
    /// Number of queries evaluated.
    pub n_queries: usize,
}

/// Run all four metrics across a batch of `(retrieved, relevant)` pairs and
/// return the per-metric mean.
///
/// Empty input returns zeros across the board with `n_queries = 0`.
pub fn evaluate_batch(queries: &[(Vec<String>, Vec<String>)], k: usize) -> AggregateMetrics {
    if queries.is_empty() {
        return AggregateMetrics {
            mean_recall_at_k: 0.0,
            mean_hit_at_k: 0.0,
            mean_mrr: 0.0,
            mean_ndcg_at_k: 0.0,
            n_queries: 0,
        };
    }
    let n = queries.len() as f64;
    let mut sr = 0.0_f64;
    let mut sh = 0.0_f64;
    let mut sm = 0.0_f64;
    let mut sn = 0.0_f64;
    for (retrieved, relevant) in queries {
        sr += recall_at_k(retrieved, relevant, k);
        sh += hit_at_k(retrieved, relevant, k);
        sm += mrr(retrieved, relevant);
        sn += ndcg_at_k(retrieved, relevant, k);
    }
    AggregateMetrics {
        mean_recall_at_k: sr / n,
        mean_hit_at_k: sh / n,
        mean_mrr: sm / n,
        mean_ndcg_at_k: sn / n,
        n_queries: queries.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn recall_at_k_basic() {
        let r = s(&["a", "b", "c"]);
        let g = s(&["b", "d"]);
        assert_eq!(recall_at_k(&r, &g, 3), 0.5); // 1 of 2 relevant found
        assert_eq!(recall_at_k(&r, &g, 1), 0.0); // top-1 is 'a', not relevant
        assert_eq!(recall_at_k(&r, &g, 2), 0.5); // top-2 includes 'b'
    }

    #[test]
    fn recall_at_k_perfect() {
        let r = s(&["a", "b"]);
        let g = s(&["a", "b"]);
        assert_eq!(recall_at_k(&r, &g, 2), 1.0);
    }

    #[test]
    fn recall_at_k_empty_relevant_is_zero() {
        let r = s(&["a"]);
        assert_eq!(recall_at_k(&r, &[], 1), 0.0);
    }

    #[test]
    fn hit_at_k_basic() {
        let r = s(&["a", "b", "c"]);
        let g = s(&["b"]);
        assert_eq!(hit_at_k(&r, &g, 3), 1.0);
        assert_eq!(hit_at_k(&r, &g, 1), 0.0);
        let r2 = s(&["x", "y", "z"]);
        assert_eq!(hit_at_k(&r2, &g, 3), 0.0);
    }

    #[test]
    fn mrr_first_position() {
        let r = s(&["a", "b", "c"]);
        let g = s(&["a"]);
        assert!((mrr(&r, &g) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_third_position() {
        let r = s(&["x", "y", "z"]);
        let g = s(&["z"]);
        assert!((mrr(&r, &g) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_no_match() {
        let r = s(&["x", "y"]);
        let g = s(&["a"]);
        assert_eq!(mrr(&r, &g), 0.0);
    }

    #[test]
    fn ndcg_perfect_ordering() {
        let r = s(&["a", "b", "c"]);
        let g = s(&["a", "b"]);
        assert!((ndcg_at_k(&r, &g, 3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_irrelevant_at_top() {
        // retrieved: x, a -> a is relevant at rank 2.
        let r = s(&["x", "a"]);
        let g = s(&["a"]);
        // DCG = 1/log2(3) ~= 0.6309
        // IDCG = 1/log2(2) = 1.0
        let v = ndcg_at_k(&r, &g, 2);
        assert!((v - (1.0 / 3.0_f64.log2())).abs() < 1e-9);
    }

    #[test]
    fn ndcg_empty_relevant_zero() {
        assert_eq!(ndcg_at_k(&s(&["a"]), &[], 1), 0.0);
    }

    #[test]
    fn evaluate_batch_means() {
        let queries = vec![
            (s(&["a", "b"]), s(&["a"])),         // perfect
            (s(&["x", "y"]), s(&["y"])),         // 2nd position
            (s(&["m", "n"]), s(&["unrelated"])), // miss
        ];
        let agg = evaluate_batch(&queries, 2);
        assert_eq!(agg.n_queries, 3);
        // Hit@2 = (1 + 1 + 0) / 3 = 0.6667
        assert!((agg.mean_hit_at_k - 2.0 / 3.0).abs() < 1e-9);
        // MRR = (1 + 0.5 + 0) / 3 = 0.5
        assert!((agg.mean_mrr - 0.5).abs() < 1e-9);
    }

    #[test]
    fn evaluate_batch_empty() {
        let agg = evaluate_batch(&[], 5);
        assert_eq!(agg.n_queries, 0);
        assert_eq!(agg.mean_hit_at_k, 0.0);
    }
}
