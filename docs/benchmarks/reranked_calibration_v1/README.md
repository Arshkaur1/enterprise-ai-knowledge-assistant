# Cross-Encoder Comparative Retrieval Benchmark V1

## Objective

Evaluate whether cross-encoder reranking improves retrieval quality sufficiently to justify its additional inference cost.

## Experimental Configuration

- Dataset: Calibration split
- Queries: 18 (14 answerable, 4 unsupported)
- Retrieval configurations: 7
- Measured requests: 1,260
- Warm-up requests: 5 per configuration
- Repetitions: 10 per query
- Final retrieval K: 2
- Cross-encoder candidate K: 5
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Device: CPU
- Batch size: 8
- Maximum sequence length: 512
- Benchmark runtime: 16.89 seconds

## Results

| Strategy | Recall@2 | MRR | Micro precision | FAR | Mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 1.000 | 0.9643 | 0.8571 | 0.50 | 3.075 | 3.704 |
| BM25 | 1.000 | 1.0000 | 0.6429 | 1.00 | 0.037 | 0.069 |
| Hybrid | 1.000 | 1.0000 | 0.6429 | 1.00 | 3.099 | 3.702 |
| Dense CE unfiltered | 1.000 | 1.0000 | 0.8571 | 0.50 | 12.091 | 21.331 |
| Dense CE calibrated | 1.000 | 1.0000 | 0.8571 | 0.50 | 11.645 | 20.425 |
| Expanded dense CE | 1.000 | 1.0000 | 0.8182 | 0.50 | 32.331 | 41.564 |
| Hybrid CE | 1.000 | 1.0000 | 0.8182 | 0.50 | 26.950 | 31.065 |

All configurations achieved complete recall of 1.0, zero false rejection and 100% ordered chunk-ID determinism.

## Key Findings

Calibrated dense reranking improved MRR from 0.9643 to 1.0 and global returned-chunk precision from 75.00% to 78.26%.

The mean retrieval latency increased by 8.57 ms, representing a 3.79x latency multiplier relative to dense retrieval.

The false acceptance rate remained unchanged at 50%. Unsupported maternity leave and password rotation questions continued to retrieve evidence despite cross-encoder threshold filtering.

Candidate expansion increased computational cost without improving recall.

No configuration satisfied all predefined acceptance criteria.

## Recommended Configuration

The preferred experimental reranking configuration is:

- Dense minimum score: 0.35
- Candidate K: 5
- Cross-encoder minimum logit: -9.0
- Final K: 2

Dense retrieval remains the default strategy.

## Decision

Partial acceptance of cross-encoder reranking.

Retain the implementation as an optional experimental retrieval strategy. Do not use cross-encoder relevance scores as a standalone guarantee of evidence sufficiency.

## Limitations

- Small synthetic corpus and calibration dataset.
- Thresholds evaluated on previously exposed calibration queries.
- Sequential rather than randomized benchmark execution.
- Warm-model inference latency excludes initialization.
- Hybrid reranking considers the top five fused candidates, not the complete candidate union.
- Candidate diagnostics are collected separately from timed requests.
- Determinism measures ordered chunk IDs rather than exact floating-point scores.
- Retrieval answerability is approximated by nonempty retrieval results.
- Downstream generation quality was not evaluated.
- Independent held-out evaluation remains necessary.

## Reproducibility

Corpus SHA-256:

`edb953f359f5f2adc4b86ba51c71bc8762789b9a7329e2878861e332c1332072`

Dataset SHA-256:

`162b810c5730da98215c1c0f2caef40ee8182facc99e2c0f868c0a9c05e8643b`

Run:

`HF_HUB_OFFLINE=1 python scripts/benchmark_reranked_retrieval.py`

The accompanying JSON and CSV reports preserve the measured results.