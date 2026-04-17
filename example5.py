from langchain.agents import create_agent
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
import requests
from tavily import TavilyClient


client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@tool
def get_weather_tool(city: str)->str:
    """A simple tool that simplifies fetching weather information for a given city."""
    print(f"Fetching weather information for {city}")

    api_key = os.getenv("OPENWEATHER_API_KEY")
    api_url = os.getenv("OPENWEATHER_API_URL")

    print(api_key)
    print(api_url)

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        
        data = response.json()

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        weather_desc = data["weather"][0]["description"]

        # print(data)
        return (
            f"In {city}, the temperature is {temperature}°C "
            f"(feels like {feels_like}°C) with {weather_desc}."
        )

    else:
        print(f"Error {response.status_code} : {response.text}")

@tool
def websearch_tool(query: str) -> str:
    """use this for general web search queries"""
    print(f"Performing web search for {query}")
    response = client.search(
        query=query,
        search_depth="advanced"
    )
    print(f"Web search results for {query} : {response}")
    return response



general_purpose_agent = create_agent(
    model = "google_genai:gemini-2.5-flash",
    tools=[get_weather_tool, websearch_tool],
    system_prompt = """You are a helpful assistant.
    You are given two tools - get_weather which fetches weather infromation for a given city and
    websearch which performs general web searches.
    If the outputs from the tools are not sufficient to answer the user's query, tell the user
    that you are not able to answer the query instead of making up an answer."""
)

response = general_purpose_agent.invoke(
    {
        "messages": [
            {"role":"user", "content":"Can you tell me about the latest news in AI?"}
        ]
    }
)

print(response["messages"][-1].text)