# embedrank

Batched cosine / dot / L2 distance for f32 embeddings, with a heap-based top-k
selector. No BLAS dependency, no allocator surprises.

For the hot path of small-to-medium RAG retrieval (up to ~100k candidates ×
~768 dims). Beyond that, reach for a real vector index.

## Install

```toml
[dependencies]
embedrank = "0.1"
```

## Use

```rust
use embedrank::{cosine, top_k_cosine, normalize_inplace};

let query = vec![1.0_f32, 0.0, 0.0];
let candidates = vec![
    vec![1.0_f32, 0.0, 0.0],
    vec![0.0_f32, 1.0, 0.0],
    vec![0.7_f32, 0.7, 0.0],
];

// Single pair:
let s = cosine(&query, &candidates[0]);  // 1.0

// Top-k retrieval:
let top = top_k_cosine(&query, &candidates, 2);
// top == vec![(1.0, 0), (~0.707, 2)]

// Normalize once, then use plain dot for speed:
let mut q = query.clone();
normalize_inplace(&mut q);
```

## API

| Function | Returns |
|---|---|
| `cosine(a, b) -> f32` | Cosine similarity in `[-1, 1]` |
| `dot(a, b) -> f32` | Plain dot product |
| `l2(a, b) -> f32` | Euclidean distance |
| `l2_squared(a, b) -> f32` | Squared L2, skip the sqrt for ranking |
| `normalize_inplace(&mut v)` | L2-normalize in place |
| `TopK::new(k)` + `push` + `into_sorted` | Heap-based top-k selector |
| `top_k_cosine(query, candidates, k)` | Convenience wrapper |
| `top_k_dot(query, candidates, k)` | Same, for pre-normalized vectors |

License: MIT OR Apache-2.0.
