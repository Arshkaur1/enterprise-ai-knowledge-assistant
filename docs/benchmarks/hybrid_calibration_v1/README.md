
# Hybrid Retrieval Calibration Benchmark — V1

## Objective

Evaluate whether BM25 lexical retrieval and weighted
Reciprocal Rank Fusion improve dense semantic retrieval.

## Experimental Setup

- Dataset: Calibration split
- Total queries: 18
- Answerable queries: 14
- Unanswerable queries: 4
- Corpus: 6 document chunks
- Embedding model: all-MiniLM-L6-v2
- Retrieval top-k: 2
- Candidate pool: 5 per retriever for hybrid retrieval
- Dense similarity threshold: 0.35
- Configurations: 7
- Warm-up requests: 5 per configuration
- Repetitions: 10 per query
- Total measured requests: 1,260

## Results

| Configuration | P@2 | R@2 | MRR | FAR | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.643 | 1.000 | 0.964 | 0.500 | 3.174 | 3.869 |
| BM25 | 0.643 | 1.000 | 1.000 | 1.000 | 0.037 | 0.067 |
| Hybrid Equal | 0.643 | 1.000 | 1.000 | 1.000 | 3.618 | 5.111 |
| Hybrid Dense Dominant | 0.643 | 1.000 | 0.964 | 1.000 | 2.848 | 2.992 |
| Hybrid Sparse Dominant | 0.643 | 1.000 | 1.000 | 1.000 | 2.852 | 3.031 |
| Hybrid RRF 10 | 0.643 | 1.000 | 1.000 | 1.000 | 3.513 | 4.407 |
| Hybrid RRF 100 | 0.643 | 1.000 | 1.000 | 1.000 | 3.309 | 3.823 |

## Key Findings

- All configurations achieved complete retrieval recall.
- BM25 and most hybrid configurations improved MRR
  from 0.9643 to 1.0.
- BM25 and all hybrid configurations increased
  retrieval-level false acceptance from 50% to 100%.
- Changing RRF weights or smoothing constants did not
  resolve false acceptance.
- BM25 retrieval was computationally inexpensive.
- No unique hybrid ranking advantage over BM25 was
  demonstrated on this calibration dataset.

## Engineering Decision

Retain dense retrieval as the default configuration.

Preserve BM25 and hybrid retrieval as experimental
candidate-generation strategies.

Evaluate cross-encoder reranking and relevance
thresholding before promoting hybrid retrieval.

## Limitations

- Small synthetic corpus.
- Previously examined calibration queries.
- Nonempty retrieval is only an answerability proxy.
- No end-to-end generation evaluation in this benchmark.
- Sequential latency measurements may have order effects.
- Fixed-denominator Precision@2 has a dataset-specific
  ceiling of approximately 64.29%.
- Dense and sparse corpus checksum equivalence requires
  explicit verification.
- BM25 score filtering policy must be documented.
- Results cannot establish production-scale performance.

## Validation

- 503 tests passed.
- 14 integration tests skipped.
- 5 dependency deprecation warnings.
- Ruff validation requires confirmation.
open 
# Hybrid Retrieval Calibration Benchmark — V1

## Objective

Evaluate whether BM25 lexical retrieval and weighted
Reciprocal Rank Fusion improve dense semantic retrieval.

## Experimental Setup

- Dataset: Calibration split
- Total queries: 18
- Answerable queries: 14
- Unanswerable queries: 4
- Corpus: 6 document chunks
- Embedding model: all-MiniLM-L6-v2
- Retrieval top-k: 2
- Candidate pool: 5 per retriever for hybrid retrieval
- Dense similarity threshold: 0.35
- Configurations: 7
- Warm-up requests: 5 per configuration
- Repetitions: 10 per query
- Total measured requests: 1,260

## Results

| Configuration | P@2 | R@2 | MRR | FAR | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.643 | 1.000 | 0.964 | 0.500 | 3.174 | 3.869 |
| BM25 | 0.643 | 1.000 | 1.000 | 1.000 | 0.037 | 0.067 |
| Hybrid Equal | 0.643 | 1.000 | 1.000 | 1.000 | 3.618 | 5.111 |
| Hybrid Dense Dominant | 0.643 | 1.000 | 0.964 | 1.000 | 2.848 | 2.992 |
| Hybrid Sparse Dominant | 0.643 | 1.000 | 1.000 | 1.000 | 2.852 | 3.031 |
| Hybrid RRF 10 | 0.643 | 1.000 | 1.000 | 1.000 | 3.513 | 4.407 |
| Hybrid RRF 100 | 0.643 | 1.000 | 1.000 | 1.000 | 3.309 | 3.823 |

## Key Findings

- All configurations achieved complete retrieval recall.
- BM25 and most hybrid configurations improved MRR
  from 0.9643 to 1.0.
- BM25 and all hybrid configurations increased
  retrieval-level false acceptance from 50% to 100%.
- Changing RRF weights or smoothing constants did not
  resolve false acceptance.
- BM25 retrieval was computationally inexpensive.
- No unique hybrid ranking advantage over BM25 was
  demonstrated on this calibration dataset.

## Engineering Decision

Retain dense retrieval as the default configuration.

Preserve BM25 and hybrid retrieval as experimental
candidate-generation strategies.

Evaluate cross-encoder reranking and relevance
thresholding before promoting hybrid retrieval.

## Limitations

- Small synthetic corpus.
- Previously examined calibration queries.
- Nonempty retrieval is only an answerability proxy.
- No end-to-end generation evaluation in this benchmark.
- Sequential latency measurements may have order effects.
- Fixed-denominator Precision@2 has a dataset-specific
  ceiling of approximately 64.29%.
- Dense and sparse corpus checksum equivalence requires
  explicit verification.
- BM25 score filtering policy must be documented.
- Results cannot establish production-scale performance.

## Validation

- 503 tests passed.
- 14 integration tests skipped.
- 5 dependency deprecation warnings.
- Ruff validation requires confirmation.


## BM25 Score Filtering Policy

The BM25 retriever uses BM25Okapi with the following
configuration:

- k1: 1.5
- b: 0.75
- Default minimum score: None
- Non-finite scores: Excluded
- Configured minimum score: Inclusive
- Explicit positive-score requirement: None

Zero and negative BM25 scores are not automatically
excluded by the score filtering conditions.

BM25 scores are corpus-dependent lexical relevance
scores and must not be interpreted as calibrated
probabilities or compared directly with cosine
similarity, RRF scores or cross-encoder logits.

The existing filtering policy is retained during
cross-encoder calibration to ensure experimental
consistency.

### Lexical Candidate Eligibility

A document chunk must share at least one token with the
tokenized query to qualify as a BM25 retrieval candidate.

Candidates with non-finite scores are excluded.

Zero and negative scores remain eligible when lexical
overlap exists and no explicit minimum score excludes them.

Candidates are sorted by descending BM25 score, with
chunk ID used as a deterministic tie-breaker.### Lexical Candidate Eligibility

A document chunk must share at least one token with the
tokenized query to qualify as a BM25 retrieval candidate.

Candidates with non-finite scores are excluded.

Zero and negative scores remain eligible when lexical
overlap exists and no explicit minimum score excludes them.

Candidates are sorted by descending BM25 score, with
chunk ID used as a deterministic tie-breaker.