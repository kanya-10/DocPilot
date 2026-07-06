"""
Text chunking for ingestion.

We chunk by paragraphs first, then greedily pack paragraphs into chunks up to
a target token budget, with overlap so context isn't lost at chunk boundaries.
This is simpler than a sliding window over raw characters and respects
natural document structure (better retrieval quality than fixed-size splits).
"""

import re
import uuid


def _approx_token_count(text: str) -> int:
    # Rough heuristic: ~4 chars per token for English text. Good enough for
    # chunk sizing without pulling in a tokenizer dependency at ingestion time.
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    source: str,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[dict]:
    """
    Split `text` into overlapping chunks.

    Returns a list of dicts: {"id": str, "text": str, "metadata": {"source": source}}
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[dict] = []
    current: list[str] = []
    current_tokens = 0

    def flush():
        if not current:
            return
        chunk_text_value = "\n\n".join(current)
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": chunk_text_value,
            "metadata": {"source": source},
        })

    for para in paragraphs:
        para_tokens = _approx_token_count(para)

        if current_tokens + para_tokens > target_tokens and current:
            flush()
            # carry the tail of the previous chunk forward for overlap
            overlap_paras: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                t = _approx_token_count(p)
                if overlap_count + t > overlap_tokens:
                    break
                overlap_paras.insert(0, p)
                overlap_count += t
            current = overlap_paras
            current_tokens = overlap_count

        current.append(para)
        current_tokens += para_tokens

    flush()
    return chunks
