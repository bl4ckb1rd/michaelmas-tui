import json
import os
import logging
from typing import Any, Dict, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from pydantic import create_model

logger = logging.getLogger(__name__)


class MCPManager:
    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = config_path
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.tools: List[StructuredTool] = []

    async def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            logger.warning(f"MCP config file not found at {self.config_path}")
            return {}
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}

    async def connect_to_server(self, name: str, config: Dict[str, Any]):
        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env", {})

        # Merge with current env
        full_env = os.environ.copy()
        full_env.update(env)

        logger.info(f"Connecting to MCP server: {name} ({command} {' '.join(args)})")

        server_params = StdioServerParameters(command=command, args=args, env=full_env)

        try:
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions[name] = session
            logger.info(f"Connected to MCP server: {name}")

            # Discover tools
            await self.discover_tools(name, session)

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {name}: {e}")

    async def discover_tools(self, server_name: str, session: ClientSession):
        try:
            result = await session.list_tools()
            for tool in result.tools:
                tool_name = f"{server_name}_{tool.name}"
                logger.info(f"Discovered tool: {tool_name}")

                # Create dynamic Pydantic model for args
                fields = {}
                if tool.inputSchema and "properties" in tool.inputSchema:
                    for prop_name, prop_schema in tool.inputSchema[
                        "properties"
                    ].items():
                        # Simplification: treating everything as Any or str for now to avoid complex type mapping issues
                        # In a real implementation, we'd map JSON schema types to Python types
                        fields[prop_name] = (Any, ...)

                args_schema = create_model(f"{tool_name}Args", **fields)

                async def _tool_func(**kwargs):
                    # Execute tool via session
                    res = await session.call_tool(tool.name, arguments=kwargs)
                    # Format result
                    if res.isError:
                        return f"Error: {res.content}"
                    return "\n".join([c.text for c in res.content if c.type == "text"])

                langchain_tool = StructuredTool.from_function(
                    func=None,
                    coroutine=_tool_func,
                    name=tool_name,
                    description=tool.description or "",
                    args_schema=args_schema,
                )
                self.tools.append(langchain_tool)

        except Exception as e:
            logger.error(f"Failed to list tools for {server_name}: {e}")

    async def initialize(self):
        config = await self.load_config()
        servers = config.get("mcpServers", {})

        for name, server_config in servers.items():
            await self.connect_to_server(name, server_config)

    async def cleanup(self):
        await self.exit_stack.aclose()


# Global instance
mcp_manager = MCPManager()
