# rust-llm-stack

Five small, focused Rust crates for the LLM / RAG / agent niche. Each one
solves a single problem well, ships independently on crates.io, and has zero
weird transitive deps.

| Crate | Purpose | Lines |
|---|---|---|
| [`embedrank`](./crates/embedrank/) | Cosine / dot / L2 distance for f32 embeddings + heap top-k | ~200 |
| [`promptbudget`](./crates/promptbudget/) | Token-budget-aware text truncation, BYO tokenizer | ~210 |
| [`stopstream`](./crates/stopstream/) | Streaming-safe stop-sequence detector with partial-match handling | ~210 |
| [`citecite`](./crates/citecite/) | Citation-marker `[1] [2]` injector + parser for RAG outputs | ~230 |
| [`ragmetric`](./crates/ragmetric/) | IR metrics: recall@k, hit@k, MRR, NDCG@k | ~250 |

Each crate is its own publishable unit. Pick the one you need:

```bash
cargo add embedrank promptbudget stopstream citecite ragmetric
```

## Why a single workspace?

These crates share a problem space (one piece of an LLM/RAG pipeline each), an
author, and dual MIT / Apache-2.0 licensing. Putting them in one repo gives you
shared CI, one README to find them all, and the option to depend on multiple
without juggling git remotes.

## Status

v0.1.0 alpha. APIs may change in 0.x. Each crate has its own tests and docs.

## License

MIT OR Apache-2.0.
