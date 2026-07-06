"""
MCP server exposing two tools to the DocPilot agent:

  1. search_github_issues -- searches LangChain's GitHub repo for issues
     matching a query. Useful when the docs don't cover a bug/edge case but
     someone else has already hit it and reported it.

  2. get_latest_pypi_version -- returns the current published version of
     langchain on PyPI. Useful for catching "the docs describe version X but
     you might be on a different one" mismatches.

Run standalone for local testing:
    python -m mcp_server.server

In production this is spawned/connected to by the agent process as an
MCP client (see agent/mcp_client.py).
"""

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("docpilot-tools")

GITHUB_REPO = "langchain-ai/langchain"
PYPI_PACKAGE = "langchain"


@mcp.tool()
def search_github_issues(query: str, max_results: int = 5) -> list[dict]:
    """
    Search open and closed GitHub issues in the LangChain repo for a query.

    Args:
        query: search terms, e.g. an error message or feature name
        max_results: max number of issues to return (default 5)

    Returns:
        List of {title, url, state, number} for matching issues.
    """
    search_url = "https://api.github.com/search/issues"
    params = {
        "q": f"{query} repo:{GITHUB_REPO}",
        "per_page": max_results,
        "sort": "updated",
        "order": "desc",
    }
    with httpx.Client(headers={"Accept": "application/vnd.github+json"}) as client:
        resp = client.get(search_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "title": item["title"],
            "url": item["html_url"],
            "state": item["state"],
            "number": item["number"],
        }
        for item in data.get("items", [])
    ]


@mcp.tool()
def get_latest_pypi_version(package: str = PYPI_PACKAGE) -> dict:
    """
    Get the current published version of a package on PyPI.

    Args:
        package: package name (defaults to "langchain")

    Returns:
        {"package": str, "version": str, "released": str}
    """
    url = f"https://pypi.org/pypi/{package}/json"
    with httpx.Client() as client:
        resp = client.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

    version = data["info"]["version"]
    release_files = data.get("releases", {}).get(version, [])
    released = release_files[0]["upload_time"] if release_files else "unknown"

    return {"package": package, "version": version, "released": released}


if __name__ == "__main__":
    mcp.run()
