# Systematic RAG Generation Baseline v1

## Experimental Configuration

- Model: Qwen 2.5 1.5B
- Embedding model: all-MiniLM-L6-v2
- Prompt version: 1.0
- Top K: 2
- Similarity threshold: 0.35
- Temperature: 0.0
- Evaluation split: Calibration
- Evaluation questions: 18
- Answerable questions: 14
- Unanswerable questions: 4

## Baseline Results

| Metric | Result |
|---|---:|
| Answerability accuracy | 88.89% |
| Answerability precision | 87.50% |
| Answerability recall | 100% |
| Answerability F1 | 93.33% |
| False acceptance rate | 50.00% |
| False rejection rate | 0.00% |
| Macro citation precision | 89.29% |
| Macro citation recall | 96.43% |
| Macro citation F1 | 90.48% |
| Operational success rate | 100% |
| Mean retrieval latency | 13.73 ms |
| Mean generation latency | 690.58 ms |
| Mean end-to-end latency | 627.75 ms |
| P95 end-to-end latency | 999.92 ms |
| Total reported tokens | 7,792 |

## Key Findings

- All answerable questions received answers.
- Both out-of-domain questions were correctly rejected.
- Both in-domain unsupported questions were incorrectly accepted.
- Generation did not improve answerability classification compared with the retrieval-only calibration baseline.
- All generated responses produced validation warnings.
- No operational errors or context truncation events occurred.
- LLM inference accounted for most of the measured processing time.

## Limitations

Answerability classification does not establish factual correctness. Citation overlap does not establish claim-level entailment. Manual answer quality assessment remains pending.

The evaluation uses a small synthetic corpus and a partially exposed dataset. Results should not be interpreted as production-level reliability.

## Next Steps

- Complete manual answer quality assessment.
- Optimize prompt instructions.
- Introduce evidence sufficiency verification.
- Improve citation precision and recall.
- Compare optimized configurations against the unchanged baseline.
- Evaluate the final configuration using fresh held-out questions.