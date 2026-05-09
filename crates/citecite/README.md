# citecite

Citation-marker `[1] [2]` injector + parser for RAG outputs. Round-trippable:
inject markers tied to source ids into model output, then parse them back when
post-processing.

## Install

```toml
[dependencies]
citecite = "0.1"
```

## Use

```rust
use citecite::{inject, parse, strip, Citation, InjectAt};

// Inject markers tied to source IDs:
let body = "Anthropic was founded in 2021.";
let cited = inject(
    body,
    &[Citation { idx: 1, source_id: "wikipedia/anthropic".into() }],
    InjectAt::End,
);
// cited == "Anthropic was founded in 2021. [1]"

// Parse markers out:
let markers = parse(&cited);
// markers == [Marker { pos: 30, idx: 1, len: 3 }]

// Or strip them entirely:
let plain = strip(&cited);
// plain == "Anthropic was founded in 2021. "
```

## API

| Function | Purpose |
|---|---|
| `inject(text, citations, InjectAt::End)` | Append `[1] [2] ...` to the end |
| `inject(text, citations, InjectAt::Position(p))` | Insert at byte position (char-boundary safe) |
| `parse(text) -> Vec<Marker>` | Find every `[N]` in source order |
| `strip(text) -> String` | Remove every `[N]` |

License: MIT OR Apache-2.0.
