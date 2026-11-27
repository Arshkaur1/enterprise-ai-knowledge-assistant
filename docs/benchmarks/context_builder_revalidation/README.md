
# ContextBuilder Regression Revalidation

## Objective

Determine whether fixing the mutable source_texts
dictionary in ContextBuilder changes the observed
evidence and grounding verification outcomes.

## Model

- LLM: qwen2.5:1.5b
- Evidence verifier: V1.1
- Grounding verifier: V1.0
- Generation prompt: V1.0

## Results

| Metric | Before Fix | After Fix |
|---|---:|---:|
| Evidence accuracy | 66.67% | 66.67% |
| Evidence recall | 25% | 25% |
| Evidence FAR | 0% | 0% |
| Evidence FRR | 75% | 75% |
| Evidence tokens | 6583 | 6583 |
| Grounding accuracy | 33.33% | 33.33% |
| Grounding operational errors | 5 | 5 |
| Grounding operational success | 44.44% | 44.44% |
| Grounding combined tokens | 7896 | 7896 |

## Findings

- All query-level classification outcomes remained
  unchanged.
- Evidence verification continued to reject three
  answerable questions.
- Grounding verification continued to produce five
  quotation validation errors.
- The maternity leave hallucination was incorrectly
  classified as supported.
- Authentic quotations did not guarantee semantic
  grounding.
- Evidence verification exhibited a substantial
  first-request latency spike.

## Engineering Decision

Retain the ContextBuilder regression fix and
deterministic quotation validation.

Do not enable either LLM-based verifier as a
mandatory production gate.

Proceed with cross-encoder reranking experiments.

Revisit semantic verification using stronger models
or natural language inference.

## Limitations

- Small, previously examined diagnostic dataset.
- No standardized cold-start latency measurement.
- Rejected raw quotations were not preserved.
- Report metadata does not identify the Git commit.
- Results do not establish generalization.
