# Chunking Strategies

| Strategy | When to use | Tradeoff |
|---|---|---|
| **Fixed** | Homogeneous text (transcripts, logs) | Simple, predictable. Loses semantic boundaries. |
| **Recursive** | Most prose / docs (default) | Splits on `\n\n`, `\n`, `. `, space in that order. Keeps structure. |
| **Semantic** | High-quality knowledge bases | Most expensive. Needs an embedder. Splits on semantic drift between sentences. |

## Parameters that matter

- **size** — target chunk length in chars. Too small: loses context; too big: dilutes relevance. 600–1200 chars is a good range for most models.
- **overlap** — bridges information spanning a boundary. 10–20% of `size` is typical.
- **breakpoint_percentile** (semantic only) — 0.6-0.85; higher = fewer, larger chunks.

## Measuring chunking quality

Run the golden-set eval against each strategy and compare `context_precision` and `answer_relevance`. Usually recursive beats fixed by ~5-15%; semantic beats recursive by ~2-10% at ~3-5x the cost. Break-even on small corpora is typically never reached — stick with recursive.
