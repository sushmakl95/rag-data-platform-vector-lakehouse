"""FastAPI server exposing /ask, /ingest, /eval.

Uses in-memory vector store + hash embeddings by default so the server runs
anywhere without configuration. Production overrides come from environment.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.chunking.recursive import recursive_chunks
from src.rag.embeddings.hash_provider import HashEmbeddingProvider
from src.rag.generation.llm_providers import EchoLLM
from src.rag.generation.prompt_templates import build_messages
from src.rag.retrieval.reranker import keyword_overlap_score
from src.rag.retrieval.retriever import HybridRetriever, RetrieverConfig
from src.rag.types import EmbeddedChunk
from src.rag.vector_stores.inmemory_store import InMemoryVectorStore

app = FastAPI(title="RAG Data Platform", version="1.0.0")

EMBEDDER = HashEmbeddingProvider(dim=384)
STORE = InMemoryVectorStore()
LLM = EchoLLM(max_chars=500)
RETRIEVER = HybridRetriever(
    embedder=EMBEDDER,
    vector_store=STORE,
    rerank_score_fn=keyword_overlap_score,
    config=RetrieverConfig(top_k_vector=20, top_k_bm25=20, top_k_rerank=3, use_rerank=True),
)


class IngestRequest(BaseModel):
    source: str
    text: str
    chunk_size: int = Field(default=800, ge=50, le=4000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    chunks: int
    total_in_store: int


class AskRequest(BaseModel):
    question: str
    filters: dict[str, Any] | None = None
    top_k: int = Field(default=3, ge=1, le=20)


class RetrievedChunk(BaseModel):
    id: str
    text: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    contexts: list[RetrievedChunk]


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "stored_chunks": STORE.count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")
    chunks = recursive_chunks(req.text, source=req.source, size=req.chunk_size)
    vectors = EMBEDDER.embed([c.text for c in chunks])
    STORE.add([EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vectors, strict=True)])
    return IngestResponse(chunks=len(chunks), total_in_store=STORE.count())


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if STORE.count() == 0:
        raise HTTPException(status_code=409, detail="no documents ingested")
    RETRIEVER.config.top_k_rerank = req.top_k
    hits = RETRIEVER.retrieve(req.question, filters=req.filters)
    contexts = [h.chunk.text for h in hits]
    messages = build_messages(req.question, contexts)
    answer = LLM.complete(messages)
    return AskResponse(
        answer=answer,
        contexts=[
            RetrievedChunk(id=h.chunk.id, text=h.chunk.text, source=h.chunk.source, score=h.score)
            for h in hits
        ],
    )
