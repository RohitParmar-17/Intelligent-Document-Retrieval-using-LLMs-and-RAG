from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_together import ChatTogether
import os


load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

PROJECT_ID = "langchain-1b2b1"
SESSION_ID = "user_session_new"  
COLLECTION_NAME = "chat_history"

print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    together_api_key=api_key
)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")