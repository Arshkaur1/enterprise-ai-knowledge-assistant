"""REST API for the Enterprise AI Knowledge Assistant."""

import logging
import os
from functools import lru_cache
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.rag import RAGEngine
from knowledge_assistant.retrieval_config import RetrievalConfig
from knowledge_assistant.retrieval_interface import RetrievalStrategy

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description=(
        "Local retrieval-augmented generation for enterprise document intelligence."
    ),
    version="1.0.0",
)

inference_lock = Lock()


class QuestionRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
    )
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    top_k: int = Field(default=2, ge=1, le=5)


@lru_cache(maxsize=2)
def get_engine(
    strategy: RetrievalStrategy,
    top_k: int,
) -> RAGEngine:
    config = RAGConfig(
        top_k=top_k,
        retrieval=RetrievalConfig(
            strategy=strategy,
        ),
    )

    return RAGEngine.from_configured_artifacts(
        artifact_dir=os.getenv(
            "ARTIFACT_DIR",
            "artifacts",
        ),
        config=config,
    )


@app.get("/")
def root():
    return {
        "service": "Enterprise AI Knowledge Assistant",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/strategies")
def strategies():
    return [
        {
            "name": strategy.value,
            "score_type": RetrievalConfig(
                strategy=strategy,
            ).score_type.value,
        }
        for strategy in RetrievalStrategy
    ]


@app.post("/ask")
def ask(request: QuestionRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        with inference_lock:
            engine = get_engine(
                request.strategy,
                request.top_k,
            )

            response = engine.ask(query)

        result = response.to_dict()

        result["retrieval_strategy"] = request.strategy.value

        result["retrieval_score_type"] = engine.config.retrieval.score_type.value

        return result

    except Exception as exc:
        logger.exception("Question processing failed.")

        raise HTTPException(
            status_code=500,
            detail="Question processing failed.",
        ) from exc
