# Cross-Encoder Relevance Threshold Calibration — V1

## Objective

Evaluate whether cross-encoder reranking improves retrieval ranking, context precision, and unsupported-query rejection while preserving complete evidence recall.

## Experimental Configuration

- Dataset: Calibration split
- Total queries: 18
- Answerable queries: 14
- Unsupported queries: 4
- Corpus size: 6 chunks
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Device: CPU
- Batch size: 8
- Maximum sequence length: 512
- Candidate K: 5
- Final K: 2
- Dense similarity threshold: 0.35
- RRF K: 60
- Threshold grid: -12 to 10, step 0.25
- Exact observed score boundaries included

## Experiment Results

A total of 668 configurations were evaluated, including two retrieval baselines and 666 reranked threshold configurations.

- Recall-preserving configurations: 283
- Quality-target-meeting configurations: 0
- Pareto-optimal configurations: 42
- Cached query-passage pairs: 198
- Recorded calibration runtime: 2.2187 seconds

Historical dense and hybrid baseline quality metrics were reproduced.

| Metric | Dense baseline | Best dense reranked |
|---|---:|---:|
| Precision@2 | 0.642857 | 0.642857 |
| Recall@2 | 1.000000 | 1.000000 |
| Complete Recall@2 | 1.000000 | 1.000000 |
| MRR | 0.964286 | 1.000000 |
| Returned-chunk micro precision | 0.857143 | 0.857143 |
| Global returned-chunk precision | 0.750000 | 0.782609 |
| False acceptance rate | 0.500000 | 0.500000 |
| False rejection rate | 0.000000 | 0.000000 |

## Selected Experimental Configuration

- Strategy: Dense retrieval followed by cross-encoder reranking
- Dense minimum similarity: 0.35
- Candidate K: 5
- Final K: 2
- Reranker minimum score: -9.0

The automatically selected threshold was -10.178775787353514. A rounded threshold of -9.0 was selected for subsequent benchmarking because it produces the same observed retrieval outcomes and avoids an immediate floating-point score boundary.

## Failure Analysis

The cross-encoder did not adequately separate relevant evidence from unsupported but topically related passages.

Representative scores:

- Minimum password length, relevant: -8.3918
- Accommodation limit, relevant: -4.9159
- Maternity leave, unsupported: -3.9966
- Password rotation, unsupported: 1.1529

No global threshold could achieve the target false acceptance rate while preserving complete evidence recall.

At a false acceptance rate of 25%, the best achievable complete recall was 85.71%, with two answerable questions rejected.

## Engineering Decision

Retain dense semantic retrieval as the default production strategy.

Preserve cross-encoder reranking as an experimental ranking component. Do not deploy it as a mandatory answerability gate.

The selected dense reranked configuration will undergo comparative latency benchmarking before any integration decision.

Further threshold optimization on the current corpus is deprioritized. Semantic chunking will be investigated as a potential method for improving relevance discrimination.

## Limitations

- Small synthetic corpus and calibration dataset.
- Only four unsupported calibration questions.
- Calibration questions were previously examined.
- The existing test split is partially exposed.
- Cross-encoder relevance scores are not calibrated probabilities or entailment scores.
- Nonempty retrieval is only a proxy for answerability.
- Cached threshold evaluation does not measure live request latency.
- Hybrid reranking was evaluated with a maximum of five scored candidates, not the complete dense-sparse union.
- Equivalent thresholds are counted as separate configurations.
- Chunk-length effects remain a hypothesis requiring controlled experimentation.

## Reproducibility

The JSON report preserves the full threshold sweep, cached logits, original retrieval scores, per-query outcomes, Pareto frontier, corpus checksum, dataset checksum, and BM25 implementation checksum.

Corpus SHA-256:

edb953f359f5f2adc4b86ba51c71bc8762789b9a7329e2878861e332c1332072

## Conclusion

Cross-encoder reranking achieved a 3.70% relative MRR improvement and a 3.26 percentage-point improvement in global returned-chunk precision without losing labelled evidence. However, it failed to improve false acceptance over the dense baseline.

The experiment demonstrates partial ranking improvement but does not satisfy the complete retrieval quality acceptance criteria.