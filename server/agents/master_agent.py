from langchain.agents import create_agent
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy


def main():
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
    )

    response = agent.stream(
        {"messages": [
            HumanMessage("直接写一个python web项目")
        ]},
        stream_mode="values"
    )

    for chunk in response:
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
    main()
