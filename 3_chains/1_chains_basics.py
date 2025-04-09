from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_together import ChatTogether
import os

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    together_api_key=api_key
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

print(prompt_template)

chain = prompt_template | llm | StrOutputParser()
result = chain.invoke({"animal": "elephant", "fact_count": 2})
print(result)