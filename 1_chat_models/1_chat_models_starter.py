from langchain_together import ChatTogether
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    together_api_key=api_key
)

result = llm.invoke("Who is the richest man in the world?")
print(result.content)
