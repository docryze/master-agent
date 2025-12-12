import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def init_multi_server_mcp_client(dir: str = "./code_dir"):
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", dir],
            }
        }
    )
    return await client.get_tools()
