# promptbudget

Token-budget-aware text truncation with multiple strategies. Bring your own
tokenizer; the crate ships a `CharTokenizer` proxy for sketches and tests.

## Install

```toml
[dependencies]
promptbudget = "0.1"
```

## Use

```rust
use promptbudget::{fit, CharTokenizer, Strategy};

let conversation = "...long chat history...";
let tok = CharTokenizer::default();  // or wire up tiktoken, BPE, etc.

// Keep latest turns:
let head_dropped = fit(conversation, 4096, Strategy::Tail, &tok);

// Keep system prompt + latest turns:
let middle_dropped = fit(
    conversation,
    4096,
    Strategy::SmartCut { head_ratio: 0.2, marker: "\n[...]\n" },
    &tok,
);
```

## Strategies

| Strategy | Drops | Good for |
|---|---|---|
| `Head` | Tail | Long source documents you want to summarize from the top |
| `Tail` | Head | Chat history where latest matters most |
| `HeadTail { head_ratio }` | Middle | Instructions + latest turn both matter |
| `SmartCut { head_ratio, marker }` | Middle, with visible cut marker | Same as HeadTail but model knows truncation happened |

## Bring your own tokenizer

Implement the `Tokenizer` trait against your real tokenizer:

```rust
use promptbudget::Tokenizer;

struct TiktokenAdapter { /* ... */ }
impl Tokenizer for TiktokenAdapter {
    fn count(&self, s: &str) -> usize { /* ... */ }
    fn truncate_head(&self, s: &str, max_tokens: usize) -> String { /* ... */ }
    fn truncate_tail(&self, s: &str, max_tokens: usize) -> String { /* ... */ }
}
```

License: MIT OR Apache-2.0.
