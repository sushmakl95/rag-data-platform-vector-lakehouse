# Evaluation Methodology

## Metrics implemented

| Metric | Definition | Production swap |
|---|---|---|
| `context_precision` | fraction of retrieved chunks that overlap with ground-truth snippets | LLM judge: "does this chunk support the answer?" |
| `context_recall` | fraction of ground-truth snippets covered by any retrieved chunk | LLM judge: "is any ground-truth fact missing?" |
| `answer_relevance` | token overlap between answer and question | LLM judge: "is the answer on-topic?" |
| `faithfulness` | token overlap between answer and retrieved contexts | LLM judge: "is every claim supported?" |
| `exact_match` | normalised equality to ground truth | BLEU / ROUGE for freeform |

## Golden set curation

- 20-200 examples typically.
- Cover the **long tail**: edge cases, ambiguous phrasings, cross-doc synthesis.
- Version the YAML; treat schema changes as breaking.

## CI gating

Thresholds in `configs/rag_config.yaml` block release if any metric drops below target on the golden set. Regression triage flow:

1. Read the failing case's retrieved contexts.
2. Check if it's a retrieval failure (chunk missing) or generation failure (chunk present, answer wrong).
3. Retrieval fail → adjust chunking / reranker / filters. Generation fail → adjust prompt.
