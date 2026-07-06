# DocPilot

An AI agent that answers developer questions about **LangChain** — combining
retrieval over the official docs with live tool calls to GitHub issues and
PyPI, so it doesn't just repeat stale documentation.

## The problem

Debugging a library issue usually means juggling three sources: the official
docs (which lag behind releases), GitHub issues (where someone already hit
your exact bug), and the currently installed package version (which may not
match what the docs describe). DocPilot collapses that into one agent that
knows when to read, when to search, and when to say "I don't know."

## How it works

```
Question
   │
   ▼
Retrieve relevant chunks from LangChain docs (RAG)
   │
   ▼
LLM decides: answer from docs / call a tool / both
   │              │
   │              ├─ search_github_issues (MCP tool)
   │              └─ get_latest_pypi_version (MCP tool)
   ▼
Answer with cited sources + confidence flag
```

- **RAG**: docs are scraped, chunked, embedded, and stored in a vector store
  behind a swappable interface — **Pinecone** in production,
  **FAISS** for local dev/tests (no API key needed to run the test suite).
- **Tools via MCP**: GitHub issue search and PyPI version lookup are exposed
  as an MCP server (`mcp_server/server.py`) and consumed by the agent as an
  MCP client — not hardcoded function calls.
- **Confidence-gated answers**: if retrieval finds nothing relevant and no
  tool resolves the question, DocPilot says so explicitly instead of
  guessing.
- **Evals**: `evals/` scores retrieval accuracy, tool-routing accuracy, and
  hallucination rate on a fixed test set — run before shipping any prompt
  change.
- **Observability**: every query and tool call is logged as structured JSON
  (`observability/logger.py`) — latency, tools used, confidence — so
  behavior is debuggable after the fact, not just during a demo.

## Project structure

```
docpilot/
├── api/            FastAPI app (POST /ask)
├── agent/          Core orchestration loop, retriever, MCP client
├── mcp_server/      MCP server exposing GitHub + PyPI tools
├── ingestion/       Scraper, chunker, embedder, pipeline entrypoint
├── vectorstore/      Pinecone + FAISS behind one interface
├── evals/           Test cases + harness
├── observability/    Structured event logging
```

## Running it locally

```bash
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
pip install -r requirements.txt

# 1. Build the knowledge base (defaults to FAISS, no Pinecone account needed)
python -m ingestion.run

# 2. Start the API
uvicorn api.main:app --reload

# 3. Ask it something
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I build a RAG pipeline in LangChain?"}'
```

## Running evals

```bash
python -m evals.run_evals
```

Reports retrieval accuracy, tool-routing accuracy, and hallucination rate
against `evals/test_cases.py`.

## Switching to Pinecone for production

```bash
export VECTOR_STORE_BACKEND=pinecone
export PINECONE_API_KEY=your-key
python -m ingestion.run   # re-index into Pinecone
```

No code changes required — the vector store is selected entirely by
`vectorstore/__init__.py:get_vector_store()`.

## Deployment

```bash
docker build -t docpilot .
docker run -p 8000:8000 --env-file .env docpilot
```

Deploys as-is to Railway, Fly.io, or Render.

## Design decisions worth discussing

- **Pinecone (prod) + FAISS (dev/test) behind one interface** — avoids
  vendor lock-in and lets the test suite run without an external API
  dependency, while still using the industry-standard managed store in
  production.
- **MCP instead of hardcoded function calling** — tools are defined once as
  an MCP server and can be reused by any MCP-compatible client, not just
  this agent.
- **Explicit confidence gating** — a wrong "I don't know" costs a follow-up
  question; a wrong hallucinated answer costs trust. The threshold in
  `agent/retriever.py` is tuned against the eval set, not guessed.
- **No agent framework in the core loop** — the tool-calling loop in
  `agent/core.py` is framework-free so the control flow is fully visible.
  A LangGraph version would be a natural next step to demonstrate framework
  fluency on top of the fundamentals.

## What I'd add next

- Reranking after retrieval (cross-encoder) for higher precision
- Hybrid search (BM25 + vector) for exact-match queries like error codes
- A scheduled re-ingestion job to keep the doc corpus current
- Langfuse/LangSmith integration in place of the JSONL logger for a real
  observability dashboard
