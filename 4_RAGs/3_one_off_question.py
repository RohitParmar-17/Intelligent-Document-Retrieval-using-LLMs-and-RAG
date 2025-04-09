import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_together import ChatTogether

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

embeddings = GPT4AllEmbeddings()

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "Where is dracula's castle located?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

model = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    together_api_key=api_key
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n--- Generated Response ---")
print("Content only:")
print(result.content)