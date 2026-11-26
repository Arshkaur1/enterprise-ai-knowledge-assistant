
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
