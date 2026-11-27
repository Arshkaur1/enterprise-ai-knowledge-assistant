# Evidence Sufficiency Verifier — Experiment V1

## Objective

Evaluate whether an independent pre-generation evidence verifier can improve unsupported-question rejection in a local RAG pipeline.

## Configuration

- Model: Qwen 2.5 1.5B
- Inference engine: Ollama
- Verifier version: 1.0
- Retrieval top-k: 2
- Minimum similarity threshold: 0.35
- Temperature: 0.0
- Verification output budget: 512 tokens
- Diagnostic dataset: 9 questions

## Results

| Metric | Result |
|---|---:|
| Accuracy | 66.67% |
| Precision | 100% |
| Recall | 25% |
| F1-score | 40% |
| False acceptance rate | 0% |
| False rejection rate | 75% |
| Operational success | 100% |
| Mean verification latency | 978.87 ms |
| Total reported tokens | 6,583 |

## Key Findings

- All five unsupported or partially supported questions were rejected.
- Three of four answerable questions were incorrectly rejected despite sufficient retrieved evidence.
- The only extracted quotation was authentic but did not support the requested fact.
- Partial evidence recognition was unsuccessful.
- Valid structured output did not guarantee semantic correctness.

## Engineering Decision

Do not enable the current verifier as a mandatory pre-generation gate.

Preserve the implementation as an experimental component. Evaluate deterministic quotation validation and post-generation answer grounding verification as alternative approaches.

## Limitations

The diagnostic dataset is small and includes previously examined questions. Results do not establish generalization performance. The underlying cause of the verifier's conservative behavior has not been isolated. Integrated pipeline latency has not been measured.