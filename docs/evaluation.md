# Semantic Retrieval Evaluation

## Objective

Evaluate the retrieval accuracy, computational efficiency and answerability classification performance of the Enterprise AI Knowledge Assistant.

## Experimental Setup

- Embedding model: Sentence Transformers all-MiniLM-L6-v2
- Embedding dimensions: 384
- Vector index: FAISS IndexFlatIP
- Similarity metric: Cosine similarity
- Hardware: Apple Silicon, CPU inference, 16 GB RAM
- Dataset: Three synthetic enterprise documents containing six indexed chunks
- Evaluation questions: 30
- Calibration questions: 18
- Held-out questions: 12
- Warm-up iterations: 5
- Repetitions per query: 10
- Total timed retrieval operations: 300


## Validated Retrieval Configuration

The selected retrieval configuration is:

- Top-K: 2
- Minimum cosine similarity: 0.35
- Embedding model: all-MiniLM-L6-v2
- Embedding dimension: 384
- Index type: FAISS IndexFlatIP

Validation on 12 held-out questions produced:

| Metric | Result |
|---|---:|
| Precision@2 | 68.75% |
| Recall@2 | 100% |
| MRR | 1.0 |
| Hit Rate@2 | 100% |
| Complete Recall Rate | 100% |
| Answerability Accuracy | 91.67% |
| Answerability F1 | 94.12% |
| False Acceptance Rate | 25% |
| False Rejection Rate | 0% |

Compared with unfiltered Top-5 retrieval, the selected configuration increased Precision@K from 27.50% to 68.75%, representing a 150% relative improvement, while maintaining complete recall on the evaluated answerable questions.

### Error Analysis

The query "What is the company's mileage reimbursement rate?" was incorrectly accepted with a cosine similarity of 0.4258.

Although the retrieved finance policy contains information about business travel and expense reimbursement, it does not specify a mileage reimbursement rate.

This demonstrates that semantic similarity alone cannot establish answerability. The next development phase will introduce evidence-grounded generation, abstention mechanisms and citation validation.

### Evaluation Limitations

These results are based on a small synthetic corpus of three documents, six chunks and 30 manually labelled questions. They should not be interpreted as production-scale retrieval accuracy or evidence of generalization to unseen enterprise documents.

## Computational Performance

| Metric | Result |
|---|---:|
| Mean retrieval latency | 3.81 ms |
| P50 latency | 3.76 ms |
| P95 latency | 4.67 ms |
| Maximum latency | 8.54 ms |
| Estimated sequential throughput | 262.68 queries/sec |

Measurements include query embedding and vector retrieval but exclude model initialization, index loading, network overhead and LLM generation.

## Threshold Calibration

Thirteen similarity thresholds between 0.00 and 0.60 were evaluated using calibration questions. The threshold of 0.35 was selected by maximizing F1 score, with lower false acceptance used as a tie-breaking criterion.

Held-out answerability classification achieved:

- Accuracy: 91.67%
- Precision: 88.89%
- Recall: 100%
- F1 score: 94.12%
- False acceptance rate: 25%
- False rejection rate: 0%

## Proposed Retrieval Configuration

- Top-K: 2
- Minimum similarity: 0.35

The combined filtered configuration requires final verification.

## Limitations

The evaluation corpus contains only three synthetic documents and six chunks. The manually constructed questions are not independently sampled, and some questions share related concepts across evaluation splits.

Similarity-based answerability classification does not guarantee that retrieved passages contain sufficient evidence to answer a question.

Additional limitations include PDF extraction artifacts, fixed-character chunking, lack of large-scale retrieval experiments and the absence of concurrent load testing.

Future experiments will investigate improved chunking, hybrid retrieval, reranking, evidence validation and grounded LLM generation.

## Reproducibility

```bash
python scripts/generate_sample_docs.py
python -m knowledge_assistant.cli
python -m knowledge_assistant.indexing
python scripts/generate_evaluation_dataset.py
HF_HUB_OFFLINE=1 python scripts/benchmark_retrieval.py
HF_HUB_OFFLINE=1 python scripts/validate_retrieval_config.py
pytest -q
```

Offline execution requires the embedding model to be cached locally.

## Quality Assurance

The project currently contains 64 passing automated tests covering document ingestion, preprocessing, chunking, vector storage, indexing, retrieval, evaluation metrics and LLM client functionality.