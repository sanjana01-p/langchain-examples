from langchain.agents import create_agent
import os
from dotenv import load_dotenv

load_dotenv()

my_agent = create_agent(
    model = "google_genai:gemini-3.1-flash-lite-preview",
    system_prompt = "You are a helpful assistant"
)

response =my_agent.invoke(
    {
        "messages": [
            {"role":"user", "content":"what is the weather in chennai?"}
        ]
    }
)

print(response)

print(response["messages"][-1].text)