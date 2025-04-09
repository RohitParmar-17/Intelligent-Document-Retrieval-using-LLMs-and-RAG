from dotenv import load_dotenv
from langchain_together import ChatTogether
import os
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    together_api_key=api_key
)

template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone":"energetic",
    "company":"samsung",
    "position":"AI Engineer",
    "skill":"AI"
})

# result = llm.invoke(prompt)
# print(result.content)
# print(prompt_template)
# print(prompt)

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "ball", "joke_count": 3})
result = llm.invoke(prompt)
print(result.content)
