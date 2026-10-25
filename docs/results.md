# Results (2026-10-25)

## test (7,814 reviews, ['2026-04-23', '2026-06-19'])

| Model | Aspect macro-F1 | Sentiment macro-F1 | Urgency macro-F1 | Kritis recall | Needs review | Throughput | p50 latency | Size |
|---|---|---|---|---|---|---|---|---|
| Rules (urgency only) | – | – | 0.637 | 0.6364 | – | – | – | – |
| TF-IDF + LogReg | 0.238 | 0.511 | 0.555 | 0.3636 | 74.2% | 2383.6/s | 6.6 ms | 0.4 MB |
| IndoBERT multi-task | 0.612 | 0.687 | 0.853 | 1.0 | 21.3% | 7.4/s | 106.48 ms | 498.6 MB |
Urgency is scored on the 122 reviews with an urgency label (11 kritis), so one kritis review is worth 9% of kritis recall.

> **Urgency labels: silver: not human-verified.** train/val: keyword rules and LLM second opinion agree; test period: LLM second opinion. See `docs/label_provenance.json`.

