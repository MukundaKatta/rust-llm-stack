# ragmetric

IR metrics for RAG retrieval evaluation: Recall@k, Hit@k, MRR, NDCG@k.

Pure data operations, no model dependencies. Works on `Vec<String>` of
retrieved doc IDs and the relevant set.

## Install

```toml
[dependencies]
ragmetric = "0.1"
```

## Use

```rust
use ragmetric::{recall_at_k, mrr, ndcg_at_k, evaluate_batch};

let retrieved = vec!["a".into(), "b".into(), "c".into()];
let relevant  = vec!["b".into(), "d".into()];

let r = recall_at_k(&retrieved, &relevant, 3);  // 0.5
let m = mrr(&retrieved, &relevant);              // 0.5 (b at rank 2)
let n = ndcg_at_k(&retrieved, &relevant, 3);     // ~0.387

// Aggregate across many queries:
let queries = vec![
    (vec!["a".into(), "b".into()], vec!["a".into()]),
    (vec!["x".into(), "y".into()], vec!["y".into()]),
];
let agg = evaluate_batch(&queries, 2);
println!("mean MRR: {:.3}", agg.mean_mrr);
```

## Metrics

| Metric | Definition |
|---|---|
| `recall_at_k` | Fraction of relevant items in the top k |
| `hit_at_k` | 1 if any relevant item is in top k, else 0 |
| `mrr` | 1 / (rank of first relevant), 0 if none retrieved |
| `ndcg_at_k` | Normalized DCG@k with binary relevance, log2 discount |

All metrics return `0.0` for an empty relevant set (vacuous query).

License: MIT OR Apache-2.0.
