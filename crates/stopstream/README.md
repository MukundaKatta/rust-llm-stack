# stopstream

Streaming-safe stop-sequence detector for LLM token streams. Buffers exactly
enough tail to detect partial matches across chunk boundaries without ever
emitting a partial match downstream.

The naive `if buffer.contains(stop) { ... }` works only when the stop sequence
lands fully inside one chunk. Real providers stream arbitrary byte boundaries,
so `</answer>` typically arrives as `</ans` then `wer>` and the naive code
emits `</ans` to the user before noticing.

## Install

```toml
[dependencies]
stopstream = "0.1"
```

## Use

```rust
use stopstream::StopDetector;

let mut det = StopDetector::new(["</answer>", "<|endoftext|>"]);

for chunk in stream {
    let r = det.push(&chunk);
    if !r.safe_text.is_empty() {
        emit_to_user(&r.safe_text);
    }
    if r.stopped.is_some() {
        break;
    }
}
let tail = det.flush();
if !tail.is_empty() {
    emit_to_user(&tail);
}
```

## Guarantees

- Never emits a partial stop match. Holds back at most `(longest_stop_len - 1)`
  bytes (snapped to a UTF-8 char boundary).
- After a stop fires, every subsequent `push` returns an empty result.
- Multi-byte UTF-8 chars are never split.
- Multiple stop sequences supported; the leftmost one in the buffer wins.

License: MIT OR Apache-2.0.
