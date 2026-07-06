"""
MCP client wrapper. Connects to mcp_server/server.py over stdio and exposes
a simple call_tool() interface plus a list of tool schemas the LLM can use
for function calling.
"""

import sys
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_server", "server.py")


class MCPToolClient:
    """
    Manages a connection to the DocPilot MCP tool server for the lifetime of
    an agent session. Use as an async context manager:

        async with MCPToolClient() as tools:
            schemas = await tools.list_tool_schemas()
            result = await tools.call_tool("get_latest_pypi_version", {"package": "langchain"})
    """

    def __init__(self):
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPToolClient":
        params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc_info):
        await self._stack.aclose()

    async def list_tool_schemas(self) -> list[dict]:
        """Return tool definitions in OpenAI function-calling format."""
        result = await self.session.list_tools()
        schemas = []
        for tool in result.tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            })
        return schemas

    async def call_tool(self, name: str, arguments: dict):
        result = await self.session.call_tool(name, arguments)
        # MCP returns a list of content blocks; DocPilot's tools return
        # a single text/JSON block we can pass straight back to the LLM.
        return "\n".join(block.text for block in result.content if hasattr(block, "text"))
