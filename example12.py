from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import List
import requests
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# Define structured output schema
class NewsArticle(BaseModel):
    """A single news article."""
    title: str = Field(description = "Title of the article")
    source:str = Field(description = "Publisher or source name")
    summary:str = Field(description = "Brief summary of the article")
    url:str = Field(description = "URL of the article, empty string if unavailable")

class AINewsResponse(BaseModel):
    """Structured respose for AI news queries."""
    topic: str = Field(description = "The news topic searched")
    articles: List[NewsArticle] = Field(description = "List of relevant articles found")
    overall_summary: str = Field(description = "Highllevel summary of current AI news")

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
        model = "google_genai:gemini-3.1-flash-lite-preview",
        tools = [get_weather_tool, websearch_tool],
        response_format = AINewsResponse,
        system_prompt = """You are a helpful assistant.
        You are given two tools - get_weather which fetches weather infromation for a given city and
        websearch which performs general web searches.
        If the outputs from the tools are not sufficient to answer the user's query, tell the user
        that you are not able to answer the query instead of making up an answer."""
    )

def main():
    topic = input("\nEnter a topic to search news about: ").strip()

    if not topic:
        print("No topic provided. Exiting.")
        return 
    
    print(f"\n... Searching for news on: '{topic}' ...\n")


    response = general_purpose_agent.invoke(
        {
            "messages": [
                {"role":"user", "content":f"Tell me about the latest news on {topic}"}
            ]
        }
    )

        
    news: AINewsResponse = response["structured_response"]
    print(f"Topic: {news.topic}")
    print(f"\nOverall Summary:\n{news.overall_summary}")
    print(f"\nArticles ({len(news.articles)}):")
    for article in news.articles:
        print(f"\n [{article.source}] {article.title}")
        print(f" {article.summary}")
        if article.url:
            print(f" {article.url}")

if __name__ == "__main__":
    main()