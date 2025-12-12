import asyncio
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
"""
协调者
"""


async def main():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/docryze/workspace/docryze/agent/master-agent/code_dir"],
            }
        }
    )
    all_tools = await client.get_tools()

    agent = create_agent(
        model="deepseek-chat",
        system_prompt="你是一个助手",
        middleware=[
            SummarizationMiddleware(
                model="deepseek-chat",
                trigger=("tokens", 4000),
                keep=("messages", 20),
            ),
            TodoListMiddleware()
        ],
        tools=all_tools,
        store=None,
    )

    response = agent.astream(
        {"messages": [
            HumanMessage("直接写一个python web项目")
        ]},
        stream_mode="values"
    )

    async for chunk in response:
        if 'messages' in chunk:
            last_message = chunk['messages'][-1]
            if isinstance(last_message, SystemMessage):
                last_message.pretty_print()
            elif isinstance(last_message, HumanMessage):
                last_message.pretty_print()
            elif isinstance(last_message, AIMessage):
                last_message.pretty_print()
                print(last_message.model_dump())
                print(last_message.content_blocks)
            elif isinstance(last_message, ToolMessage):
                last_message.pretty_print()
            else:
                print(last_message)
        else:
            print(chunk)
        # for message in messages:
        #     if isinstance(message, AIMessageChunk):
        #         print(message.content, end="", flush=True)
        #     else:
        #         print(type(message), message)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
