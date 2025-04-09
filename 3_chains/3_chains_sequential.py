from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_together import ChatTogether
import os

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    together_api_key=api_key
)


animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})


chain = animal_facts_template | llm | StrOutputParser() | prepare_for_translation | translation_template | llm | StrOutputParser() | count_words

result = chain.invoke({"animal": "cat", "count": 2})

print(result)