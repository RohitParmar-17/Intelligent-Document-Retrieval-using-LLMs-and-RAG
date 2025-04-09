from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_together import ChatTogether
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    together_api_key=api_key
)

chat_history = []

sysmessage = SystemMessage("You are a helpful AI assistant")
chat_history.append(sysmessage)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")