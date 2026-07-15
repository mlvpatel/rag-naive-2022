"""FastAPI service for RagFlow, the naive RAG baseline (2022).

Endpoints are intentionally simple: ask a question, manage documents, check
health. The retrieval behind /v1/chat is a single dense search, no reranking.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api import memory
from src.api.security import limiter, sanitize
from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.rag_chain import answer_question
from src.embeddings.vectorstore import delete_document, index_document

log = get_logger("ragflow.api")

_CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.init_db()
    log.info("RagFlow started, database ready")
    yield


app = FastAPI(
    title="RagFlow",
    description="Naive RAG baseline, 2022 generation",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[str]


class DeleteRequest(BaseModel):
    file_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
def chat(request: Request, body: ChatRequest):
    question = sanitize(body.question)
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    session_id = body.session_id or uuid.uuid4().hex
    result = answer_question(question)
    memory.log_turn(session_id, question, result["answer"])
    return ChatResponse(
        answer=result["answer"], session_id=session_id, sources=result["sources"]
    )


def _copy_capped(src, dst, limit: int) -> None:
    """Stream an upload to disk, stopping the moment it passes the cap.

    Trusting Content-Length would trust the client, and measuring after the copy
    means the disk already took the hit, so count while writing.
    """
    total = 0
    while chunk := src.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="File too large")
        dst.write(chunk)


# Embedding a document costs far more than answering one question, so this is
# the endpoint that most needs a limit, not the one that least needs it.
@app.post("/v1/upload-doc")
@limiter.limit("6/minute")
def upload_doc(request: Request, file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as out:
            _copy_capped(file.file, out, get_settings().max_upload_mb * 1024 * 1024)
        file_id = uuid.uuid4().hex
        chunks = index_document(tmp_path, file_id, file.filename or "document")
        memory.register_document(file_id, file.filename or "document", chunks)
    finally:
        os.unlink(tmp_path)
    return {"file_id": file_id, "filename": file.filename, "chunks": chunks}


@app.get("/v1/list-docs")
def list_docs():
    return {"documents": memory.list_documents()}


@app.post("/v1/delete-doc")
def delete_doc(body: DeleteRequest):
    delete_document(body.file_id)
    memory.remove_document(body.file_id)
    return {"deleted": body.file_id}
