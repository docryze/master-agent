from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain.messages import HumanMessage, AIMessageChunk

load_dotenv()


def main():
    agent = create_agent(
        model="deepseek-chat",
        system_prompt="你是一个助手",
    )

    response = agent.stream(
        {"messages": [
            HumanMessage("帮我写一个python web项目")
        ]},
        stream_mode="messages"
    )

    for messages in response:
        for message in messages:
            if isinstance(message, BaseMessageChunk):
                print(message.content, end="", flush=True)


if __name__ == "__main__":

    main()
