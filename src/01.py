from langchain.chat_models import init_chat_model
from langgraph import graph
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()


class ProgramState(BaseModel):
    status: str


model = init_chat_model("deepseek-chat")

response = model.invoke("介绍自己")

print(response)
