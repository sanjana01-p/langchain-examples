from langchain.agents import create_agent
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
import requests

load_dotenv(override=True)

@tool
def get_weather(city: str)->str:
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


weather_agent = create_agent(
    model = "gpt-4o-mini",
    tools=[get_weather],
    system_prompt = "You are a helpful assistant that provides weather information."
)

response =weather_agent.invoke(
    {
        "messages": [
            {"role":"user", "content":"what is the weather in Chennai today?"}
        ]
    },
    config={
        "tags": ["weather-agent", "example3"],
        "metadata": {
            "user_id": "user_001",
            "feature": "weather_lookup",
            "env": "dev"
        },
        "run_name": "weather_query_run"
    }
)

print(response["messages"][-1].text)