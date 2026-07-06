"""
Eval test set for DocPilot.

Each case has:
  - question: the input
  - expected_source_contains: a substring expected to appear in at least one
    retrieved source URL (tests retrieval quality)
  - expected_tool: a tool name expected to be called, or None if the
    question should be answerable from docs alone (tests routing quality)
  - answerable: whether a confident answer should be possible at all
    (tests that DocPilot says "I don't know" instead of hallucinating on
    out-of-scope questions)
"""

EVAL_CASES = [
    {
        "question": "What is a retriever in LangChain?",
        "expected_source_contains": "retrievers",
        "expected_tool": None,
        "answerable": True,
    },
    {
        "question": "How do I build a basic RAG pipeline with LangChain?",
        "expected_source_contains": "rag",
        "expected_tool": None,
        "answerable": True,
    },
    {
        "question": "What's the current version of langchain on PyPI?",
        "expected_source_contains": None,
        "expected_tool": "get_latest_pypi_version",
        "answerable": True,
    },
    {
        "question": "I'm getting a streaming callback error with LangChain, has anyone else reported this?",
        "expected_source_contains": None,
        "expected_tool": "search_github_issues",
        "answerable": True,
    },
    {
        "question": "How do prompt templates work in LangChain?",
        "expected_source_contains": "prompt_templates",
        "expected_tool": None,
        "answerable": True,
    },
    {
        "question": "What's the best pizza topping for a Tuesday?",
        "expected_source_contains": None,
        "expected_tool": None,
        "answerable": False,
    },
    {
        "question": "How does LangChain's tool calling concept work?",
        "expected_source_contains": "tool_calling",
        "expected_tool": None,
        "answerable": True,
    },
    {
        "question": "What is the airspeed velocity of an unladen swallow according to LangChain docs?",
        "expected_source_contains": None,
        "expected_tool": None,
        "answerable": False,
    },
]
