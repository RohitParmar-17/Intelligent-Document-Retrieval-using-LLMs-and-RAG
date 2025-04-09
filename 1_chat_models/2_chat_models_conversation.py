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

messages = [
    SystemMessage("You are an expert in a cricket analytics"),
    HumanMessage("Who are the best batters across all formats"),
    AIMessage("""What a great question!
            As an expert in cricket analytics, I've compiled a list of the best batters across all formats, based on their impressive performances and statistics. Keep in mind that this is a subjective list, and opinions may vary. Here are the top batters across all formats:

            **Test Cricket:**

            1. Virat Kohli (India) - Average: 53.51, Runs: 20,615
            2. Steve Smith (Australia) - Average: 59.51, Runs: 15,213
            3. AB de Villiers (South Africa) - Average: 50.66, Runs: 13,493
            4. Kane Williamson (New Zealand) - Average: 54.47, Runs: 12,744
            5. Joe Root (England) - Average: 50.38, Runs: 12,444

            **One-Day Internationals (ODIs):**

            1. Rohit Sharma (India) - Average: 45.33, Runs: 13,731
            2. Virat Kohli (India) - Average: 58.31, Runs: 12,176
            3. AB de Villiers (South Africa) - Average: 53.50, Runs: 9,577
            4. Tillakaratne Dilshan (Sri Lanka) - Average: 43.61, Runs: 9,466
            5. Hashim Amla (South Africa) - Average: 52.43, Runs: 9,164

            **T20 Internationals (T20Is):**

            1. Virat Kohli (India) - Average: 37.88, Runs: 3,175
            2. AB de Villiers (South Africa) - Average: 33.53, Runs: 2,364
            3. Rohit Sharma (India) - Average: 32.41, Runs: 2,343
            4. Chris Gayle (West Indies) - Average: 31.48, Runs: 2,274
            5. Kusal Perera (Sri Lanka) - Average: 31.45, Runs: 1,544

            These batters have consistently performed well across formats, showcasing their skills and adaptability. Of course, there are many other talented batters in each format, but these individuals have stood out with their impressive records.

            Remember, statistics are just one way to measure a batter's performance. Other factors like their ability to adapt to different conditions, partnerships, and captaincy skills also play a significant role in a batter's overall success."""),
    HumanMessage("what is said previously?")
]

result = llm.invoke(messages)
print(result.content)
