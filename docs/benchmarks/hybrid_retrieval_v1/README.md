# Hybrid Retrieval V1 — Diagnostic Experiment

## Objective

Evaluate whether combining dense semantic retrieval and BM25 lexical retrieval using Reciprocal Rank Fusion improves evidence retrieval without introducing significant computational overhead.

## Configuration

- Dense model: all-MiniLM-L6-v2
- Dense threshold: 0.35
- Sparse retriever: BM25
- Candidate pool: 5 per retriever
- Final top-k: 2
- RRF smoothing constant: 60
- Dense and sparse weights: 1.0 each
- Warm-up requests: 5
- Diagnostic queries: 14
- Corpus size: 6 chunks

## Results

| Metric | Dense | BM25 | Hybrid |
|---|---:|---:|---:|
| Diagnostic Hit@2 | 80% | 90% | 90% |
| Mean latency (ms) | 4.2261 | 0.0582 | 4.2965 |
| Multi-part complete recall | 100% | 100% | 100% |
| Unsupported-query context acceptance | 75% | 100% | 100% |

## Key Findings

Hybrid retrieval recovered relevant numerical evidence for the query "12 characters", which thresholded dense retrieval failed to retrieve.

Both retrieval methods failed to resolve the abbreviation "MFA".

Hybrid retrieval preserved complete recall for both multi-document diagnostic questions.

The additional mean retrieval latency was approximately 0.0704 ms, representing a 1.67% overhead relative to dense retrieval.

However, hybrid retrieval reintroduced irrelevant context for an out-of-domain weather query.

Identical RRF scores were observed for answerable and unsupported questions, demonstrating that ranking agreement cannot establish evidence sufficiency.

A zero-score BM25 candidate was also observed and requires investigation.

## Engineering Decision

Retain hybrid retrieval as an experimental retrieval strategy. Do not replace the default dense retriever until systematic calibration, reranking and downstream generation evaluation are completed.

## Limitations

The experiment uses a small synthetic corpus and a manually assessed diagnostic dataset. Results are not representative of production-scale retrieval performance. The combined latency is calculated from component timings rather than direct end-to-end hybrid search measurements.

## Test Results

498 passed, 14 skipped, 5 warnings.