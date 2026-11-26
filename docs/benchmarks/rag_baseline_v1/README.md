# RAG Generation Baseline v1

## Environment

- Platform: macOS Apple Silicon
- Embeddings: all-MiniLM-L6-v2 (CPU)
- Vector search: FAISS IndexFlatIP
- LLM: Qwen 2.5 1.5B
- Inference: Ollama GPU-accelerated
- Prompt version: 1.0
- Context window: 4096 tokens

## Configuration

- Top K: 2
- Minimum similarity: 0.35
- Maximum context: 4000 characters
- Temperature: 0.0
- Maximum generation: 256 tokens

## Results

- Automated tests: 203 passed
- Smoke-test questions: 8
- Correct answers: 5
- Correct abstentions: 2
- Incorrect answers: 1
- Manually assessed accuracy: 87.5%
- Answerability F1-score: 90.91%
- Mean retrieval latency: 27.49 ms
- Mean generation latency: 831.86 ms
- Mean end-to-end latency: 755.54 ms
- Total reported tokens: 3478
- Generation errors: 0
- Invalid responses: 0

## Known limitations

- Incorrect answer to password rotation frequency.
- Over-citation of retrieved chunks.
- All seven generated responses required normalization.
- Character-based context budgeting.
- PDF extraction and chunk-boundary artifacts.
- Small synthetic evaluation sample.
- No automated factual entailment verification.

## Conclusion

The baseline demonstrates functional end-to-end RAG
orchestration with citation resolution, early abstention,
structured generation and latency tracking.

Further evaluation is required before making claims
about generalization or production reliability.