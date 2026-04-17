from langchain.agents import create_agent
import os
from dotenv import load_dotenv

load_dotenv()

my_agent = create_agent(
    model = "google_genai:gemini-2.5-flash-lite",
    system_prompt = """You are a LinkedIn Post Writer. You must create engaging
    and infromative posts. You should never work on anything else.
    If users ask you to do something outside of this scope, politely decline."""
)

response =my_agent.invoke(
    {
        "messages": [
            {"role":"user", "content":"Write a LinkedIn post about the importance of AI."}
        ]
    }
)

print(response["messages"][-1].text)