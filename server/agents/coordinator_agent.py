from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain.agents import create_agent
import asyncio
from server.mcp.mcp import init_multi_server_mcp_client


"""
协调者
"""


async def main():
    all_tools = await init_multi_server_mcp_client()

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
        stream_mode="messages"
    )

    # async for chunk in response:
    # if 'messages' in chunk:
    #     last_message = chunk['messages'][-1]
    #     if isinstance(last_message, SystemMessage):
    #         last_message.pretty_print()
    #     elif isinstance(last_message, HumanMessage):
    #         last_message.pretty_print()
    #     elif isinstance(last_message, AIMessage):
    #         last_message.pretty_print()
    #         # print(last_message.model_dump())
    #         # print(last_message.content_blocks)
    #         print(last_message.content, end="", flush=True)
    #     elif isinstance(last_message, ToolMessage):
    #         last_message.pretty_print()
    #     else:
    #         print(last_message)
    # else:
    #     print(chunk)

    pre_message_type = None
    async for messages in response:
        for message in messages:
            if isinstance(message, BaseMessage):
                if pre_message_type != message.type:
                    pre_message_type = message.type
                    print()
            if isinstance(message, SystemMessage):
                message.pretty_print()
            elif isinstance(message, HumanMessage):
                message.pretty_print()
            elif isinstance(message, AIMessage):
                # message.pretty_print()
                # print(last_message.model_dump())
                # print(last_message.content_blocks)
                # print(message.content, end="", flush=True)
                for content_block in message.content_blocks:
                    content_block_type = content_block.get('type')
                    if content_block_type == 'text':
                        print(message.content, end="", flush=True)
                    elif content_block_type == 'tool_call_chunk':
                        if content_block.get("name"):
                            print()
                            print(content_block.get("name"))
                        if content_block.get("args"):
                            print(content_block.get(
                                "args"), end="", flush=True)
                    else:
                        print(content_block, end="", flush=True)

            # elif isinstance(message, ToolMessage):
            #     message.pretty_print()
            # else:
            #     print(type(message))


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
