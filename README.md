
# Enterprise AI Knowledge Assistant

An enterprise document intelligence system powered by
Retrieval-Augmented Generation (RAG) and local LLMs.

## Objective

Build an intelligent document question-answering system
that retrieves relevant information from enterprise
documents and generates grounded answers with citations.

## Technology Stack

- Python
- Ollama
- Hugging Face
- Sentence Transformers
- FAISS
- FastAPI
- PyTorch
- Docker

## Architecture

Document Ingestion
    ↓
Text Processing
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Vector Database
    ↓
Semantic Retrieval
    ↓
LLM Generation
    ↓
Grounded Answers

## Current Progress

- [x] Project environment setup
- [x] Repository structure
- [x] Local LLM integration
- [x] Initial text chunking
- [x] Unit tests
- [ ] Document ingestion pipeline
- [ ] Embedding generation
- [ ] Vector database
- [ ] RAG implementation
- [ ] Model evaluation
- [ ] API development
- [ ] Deployment

## Running Locally

Install dependencies:

pip install -e ".[dev,llm]"

Run the application:

python -m knowledge_assistant.cli \
  --prompt "Explain RAG"

## Retrieval-Augmented Generation

The Enterprise AI Knowledge Assistant implements a modular, locally hosted RAG pipeline for answering questions from enterprise policy documents.

### Features

- Semantic retrieval using Sentence Transformers and FAISS.
- Configurable similarity thresholds and context budgets.
- Evidence-grounded prompt construction.
- Local LLM inference using Qwen 2.5 through Ollama.
- Structured JSON generation and response validation.
- Document citations with PDF page references.
- Early abstention when no evidence is retrieved.
- Retrieval and generation latency tracking.
- Interactive and programmatic command-line interfaces.

### Usage

Interactive question answering:

```bash
knowledge-assistant
```

Single-question inference:

```bash
knowledge-assistant "What is the annual leave policy?"
```

Structured JSON output:

```bash
knowledge-assistant "What is the expense submission deadline?" --json
```

### Testing

Run the standard automated test suite:

```bash
pytest -q
```

Run real-model integration tests:

```bash
RUN_INTEGRATION_TESTS=1 HF_HUB_OFFLINE=1 pytest -q
```

### Preliminary RAG Baseline

| Metric | Result |
|---|---:|
| Smoke-test questions | 8 |
| Correct answers | 5 |
| Correct abstentions | 2 |
| Incorrect answers | 1 |
| Manually assessed accuracy | 87.50% |
| Answerability F1-score | 90.91% |
| Mean end-to-end latency | 755.54 ms |
| Generation errors | 0 |
| Invalid responses | 0 |

These results represent a preliminary baseline on a small synthetic enterprise policy corpus and should not be interpreted as production-level performance.

### Known Limitations

- Topically relevant evidence may not contain the requested information.
- Citation validation does not guarantee factual entailment.
- Small language models may produce irrelevant answers instead of abstaining.
- Context budgeting is currently character-based.
- Conversational memory is not yet supported.

Systematic evaluation and optimization will address these limitations in subsequent development phases.