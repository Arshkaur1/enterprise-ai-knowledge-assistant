# Post-Generation Grounding Verification V1.0

## Objective

Evaluate whether post-generation grounding verification improves answer reliability compared with standalone pre-generation evidence sufficiency verification and the existing generation-only RAG architecture.

## Experimental Configuration

- Generation model: Qwen 2.5 1.5B
- Verification model: Qwen 2.5 1.5B
- Generation prompt version: 1.0
- Grounding verifier version: 1.0
- Retrieval top-k: 2
- Minimum similarity score: 0.35
- Temperature: 0.0
- Evaluation questions: 9
- Warm-up request excluded from reported results

## Results

| Metric | Generation Only | Pre-Generation V1.1 | Post-Generation V1.0 |
|---|---:|---:|---:|
| Accuracy | 66.67% | 66.67% | 33.33% |
| True positives | 4 | 1 | 1 |
| True negatives | 2 | 5 | 2 |
| False positives | 3 | 0 | 1 |
| False negatives | 0 | 3 | 0 |
| Operational errors | 0 | 0 | 5 |
| Operational success | 100% | 100% | 44.44% |
| Reported tokens | 3,791 | 6,583 | 7,896 |

## Key Findings

1. All four answerable questions were answered correctly by the generator before verification.
2. Three correct answers were subsequently blocked by quotation validation failures.
3. Five verification responses failed deterministic quotation authenticity checks.
4. The verifier incorrectly approved a maternity leave hallucination using authentic but semantically irrelevant annual leave evidence.
5. Two unsupported generated answers were blocked through invalid-response handling rather than correct semantic abstention.
6. Post-generation verification increased reported token consumption by 108.28% relative to generation alone.
7. Mean verification latency was 1,740.35 milliseconds per verification request.

## Root Cause Analysis

The observed failures indicate limitations in exact quotation extraction and semantic grounding.

PDF extraction artifacts may contribute to quotation mismatches, but this cannot be confirmed because rejected raw quotations were not preserved.

Using the same small language model for generation and verification also creates correlated error risks.

## Engineering Decision

Do not integrate the current post-generation verifier into the default RAG pipeline.

Retain the grounding verification infrastructure and deterministic quotation validator for future experiments with alternative models.

Preserve the original generation-only architecture as the operational baseline.

## Future Improvements

- Evaluate stronger or independent verification models.
- Investigate natural language inference-based grounding.
- Improve PDF extraction and chunking quality.
- Preserve rejected verifier responses in controlled diagnostic artifacts.
- Evaluate claim-level factual support and citation entailment.
- Validate the final architecture using an independent held-out dataset.

## Limitations

The experiment uses a small, previously examined diagnostic dataset. Results are not statistically generalizable.

The generation-only baseline was reconstructed from candidate answers in the same diagnostic run. The pre-generation verifier performs a different task and its computational cost is not directly equivalent.

Operational errors are counted as incorrect outcomes and are not treated as successful abstentions.