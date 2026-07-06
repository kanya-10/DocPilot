"""
Scrapes LangChain's documentation pages into plain text for ingestion.

We keep this deliberately simple and polite:
  - a small, explicit list of doc pages (not a full-site crawler) so the demo
    corpus is reproducible and doesn't hammer their site
  - a delay between requests
  - strips nav/sidebar boilerplate, keeps main content only

For a real production system you'd use their sitemap.xml and a proper crawler
with dedup + change detection; that's out of scope for a portfolio demo but
worth mentioning in an interview as "the next thing I'd build."
"""

import time
import httpx
from bs4 import BeautifulSoup

# A representative slice of LangChain's Python docs. Swap/extend this list
# to widen the corpus.
DOC_URLS = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/rag/",
    "https://python.langchain.com/docs/concepts/agents/",
    "https://python.langchain.com/docs/concepts/tool_calling/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/chat_models/",
    "https://python.langchain.com/docs/concepts/prompt_templates/",
    "https://python.langchain.com/docs/concepts/output_parsers/",
    "https://python.langchain.com/docs/concepts/streaming/",
    "https://python.langchain.com/docs/how_to/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/agents/",
]


def fetch_page_text(url: str, client: httpx.Client) -> str:
    resp = client.get(url, timeout=20, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return ""

    for tag in main.select("nav, script, style, footer, .theme-doc-sidebar-container"):
        tag.decompose()

    text = main.get_text(separator="\n\n")
    lines = [line.strip() for line in text.split("\n\n")]
    return "\n\n".join(line for line in lines if line)


def scrape_all(urls: list[str] = DOC_URLS, delay_seconds: float = 1.0) -> list[dict]:
    """Returns list of {"url": str, "text": str}, skipping pages that fail to fetch."""
    pages = []
    with httpx.Client(headers={"User-Agent": "DocPilot-Ingestion/1.0"}) as client:
        for url in urls:
            try:
                text = fetch_page_text(url, client)
                if text:
                    pages.append({"url": url, "text": text})
            except httpx.HTTPError as e:
                print(f"[scrape] skipping {url}: {e}")
            time.sleep(delay_seconds)
    return pages
