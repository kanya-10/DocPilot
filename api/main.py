"""
FastAPI app exposing DocPilot as an HTTP service.

Run locally:
    uvicorn api.main:app --reload

POST /ask {"question": "..."} -> {"answer": ..., "sources": [...], "tools_used": [...], "confidence": ...}
"""

import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.core import DocPilotAgent
from observability.logger import log_event

app = FastAPI(title="DocPilot", description="AI agent for LangChain docs, issues, and releases")
agent = DocPilotAgent()

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


# Simple in-memory rate limiter: max N requests per IP per minute.
# For real production traffic you'd move this to a reverse proxy / Redis,
# but this is enough to demonstrate the pattern and stop naive abuse.
_RATE_LIMIT = 20
_WINDOW_SECONDS = 60
_request_log: dict[str, list[float]] = {}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    tools_used: list[str]
    confidence: str
    latency_ms: int


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    history = [t for t in _request_log.get(client_ip, []) if now - t < _WINDOW_SECONDS]
    if len(history) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    history.append(now)
    _request_log[client_ip] = history


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, client_ip: str = "anonymous"):
    _check_rate_limit(client_ip)

    start = time.time()
    try:
        result = await agent.answer(request.question)
    except Exception:
        tb = traceback.format_exc()
        log_event("error", question=request.question, traceback=tb)
        raise HTTPException(status_code=500, detail="Something went wrong answering your question.")

    latency_ms = int((time.time() - start) * 1000)

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        tools_used=result.tools_used,
        confidence=result.confidence,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}