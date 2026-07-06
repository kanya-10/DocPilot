"""
Core agent loop.

Flow per query:
  1. Retrieve relevant doc chunks (RAG)
  2. Give the LLM the retrieved context + MCP tool schemas
  3. LLM decides: answer from context, call a tool (GitHub issues / PyPI
     version), or both
  4. If a tool is called, execute it via MCP and feed the result back
  5. If confidence is low and no tool resolved it, return an explicit
     "I don't know" rather than a hallucinated answer

This loop is intentionally framework-free (no LangGraph) so it's easy to
read end-to-end and easy to explain in an interview. See README for how
this maps onto LangGraph if you want to show that too.
"""

import json
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI, RateLimitError

from agent.retriever import Retriever
from agent.mcp_client import MCPToolClient
from observability.logger import log_event

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are DocPilot, an assistant for developers using the LangChain Python library.

You have two ways to answer:
1. Documentation context provided below, retrieved from LangChain's official docs.
2. Tools: search_github_issues (for bugs/edge cases not covered in docs) and
   get_latest_pypi_version (to check the current released version).

Rules:
- Prefer the documentation context when it answers the question. Cite sources using plain [1], [2] etc. -- do not use any other citation format or special bracket tokens.
- If the question sounds like a bug, error, or "why doesn't this work", search GitHub issues.
- If the question is about versions or "is this still the latest way to do X", check the PyPI version.
- If neither the docs nor the tools give you a confident answer, say so explicitly.
  Do not guess or fabricate an answer.
"""


@dataclass
class AgentResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    confidence: str = "high"  # "high" | "low"


class DocPilotAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.retriever = Retriever()

    def _complete_with_retry(self, messages, tools, max_retries: int = 4):
        """
        Groq's free tier has a low tokens-per-minute limit, which is easy to
        hit during eval runs or bursts of tool-calling. Retry with backoff
        instead of failing the whole request on a transient 429.
        """
        delay = 5.0
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools,
                )
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                log_event("rate_limited", attempt=attempt, sleeping_seconds=delay)
                time.sleep(delay)
                delay *= 2

    async def answer(self, question: str) -> AgentResponse:
        docs = self.retriever.retrieve(question)
        context = self.retriever.format_context(docs)
        sources = [d.metadata.get("source", "unknown") for d in docs]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documentation context:\n{context}\n\nQuestion: {question}"},
        ]

        tools_used: list[str] = []

        async with MCPToolClient() as mcp_tools:
            tool_schemas = await mcp_tools.list_tool_schemas()

            response = self._complete_with_retry(messages, tool_schemas)
            message = response.choices[0].message

            # Tool-calling loop: keep executing tools until the model is
            # ready to produce a final answer.
            while message.tool_calls:
                messages.append(message)
                for call in message.tool_calls:
                    args = json.loads(call.function.arguments)
                    result = await mcp_tools.call_tool(call.function.name, args)
                    tools_used.append(call.function.name)
                    log_event("tool_call", tool=call.function.name, args=args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })
                response = self._complete_with_retry(messages, tool_schemas)
                message = response.choices[0].message

        final_answer = message.content or ""
        confidence = "low" if _looks_uncertain(final_answer) else "high"

        log_event(
            "query_answered",
            question=question,
            tools_used=tools_used,
            num_sources=len(sources),
            confidence=confidence,
        )

        return AgentResponse(
            answer=final_answer,
            sources=sources,
            tools_used=tools_used,
            confidence=confidence,
        )


def _looks_uncertain(answer: str) -> bool:
    markers = [
        "i don't know", "not sure", "couldn't find", "no relevant documentation",
        "don't have information", "doesn't contain", "does not contain",
        "no information", "i'm sorry, but", "unable to find", "cannot find",
    ]
    # LLM output often uses typographic apostrophes (') instead of straight
    # ones ('); normalize so marker matching isn't silently broken by this.
    lowered = answer.lower().replace("\u2019", "'")
    return any(m in lowered for m in markers)
