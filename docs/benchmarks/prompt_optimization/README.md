# Controlled Prompt Optimization Experiment

## Objective

Evaluate whether explicit evidence sufficiency instructions, attribute matching, citation selectivity and few-shot examples improve RAG performance.

## Experimental Design

Two prompt versions were evaluated using the same 18-question calibration dataset, Qwen 2.5 1.5B model, embedding model, retrieval configuration and response validation pipeline.

- Control: Prompt V1.0
- Treatment: Prompt V2.0
- Top K: 2
- Minimum similarity score: 0.35
- Temperature: 0.0
- Context budget: 4,000 characters

## Results

| Metric | V1 | V2 |
|---|---:|---:|
| Answerability accuracy | 88.89% | 88.89% |
| Answerability F1 | 93.33% | 93.33% |
| False acceptance rate | 50.00% | 50.00% |
| False rejection rate | 0.00% | 0.00% |
| Citation precision | 89.29% | 92.86% |
| Citation recall | 96.43% | 78.57% |
| Citation F1 | 90.48% | 83.33% |
| Generation warning rate | 100.00% | 12.50% |
| Operational success rate | 100% | 100% |
| Mean end-to-end latency | 666.65 ms | 666.81 ms |
| P95 latency | 970.79 ms | 1,027.34 ms |
| Total reported tokens | 7,792 | 13,808 |

## Findings

Prompt V2 reduced the number of responses requiring validation warnings by 87.5% and improved citation precision by 3.57 percentage points.

However, citation recall decreased by 17.86 percentage points, and total reported token consumption increased by 77.21%.

Neither prompt version correctly rejected the two in-domain unsupported policy questions.

The results indicate that the tested prompt modifications alone were insufficient to resolve the evidence sufficiency problem.

## Engineering Decision

Retain Prompt V1 as the default configuration pending detailed response-level assessment.

Preserve Prompt V2 for reproducibility and future experimentation.

Investigate a dedicated evidence sufficiency verification layer to improve unsupported-question rejection.

## Limitations

- Small synthetic evaluation dataset.
- Calibration results are not independent held-out performance estimates.
- Single benchmark run per prompt version in the controlled comparison.
- Citation overlap does not establish claim-level factual entailment.
- Manual factual correctness assessment for V2 remains pending.
- Latency differences may reflect caching and runtime variability.

## Conclusion

Prompt optimization improved output compliance but did not improve answerability classification. Architectural evidence verification is the next experimental direction.