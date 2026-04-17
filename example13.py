 # Agent with Tool call + Guardrails + Edge Case testing
from dotenv import load_dotenv
from langchain_core.tools import tool   
import os
from langchain.agents import create_agent
import requests

# Load the .env file
load_dotenv()

@tool
def calculator(expression: str) -> str:
  """Evaluate a math expression."""
  print(f"Calculating the result for {expression}...")
  try:
    result = eval(expression, {"__builtins__": {}}, {})
    return f"The result of {expression} is {result}."
  except Exception as e:
    return f"Sorry, I couldn't evaluate the expression: {expression}. Error: {str(e)}"

@tool
def get_weather(city: str) -> str:
  """Get the current weather for a real city. Input must be a valid city name."""
  print(f"Fetching weather for {city}...")
  try:
    # Geocoding API to get latitude and longitude of the city
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geocode_response = requests.get(geocode_url)
    geocode_data = geocode_response.json()

    if "results" not in geocode_data or len(geocode_data["results"]) == 0:
      return f"It looks like the city you mentioned is invalid or fictional."

    latitude = geocode_data["results"][0]["latitude"]
    longitude = geocode_data["results"][0]["longitude"]

    # Weather API to get current weather using latitude and longitude
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()

    if "current_weather" not in weather_data:
      return f"Sorry, I couldn't fetch the weather for {city}."

    temperature = weather_data["current_weather"]["temperature"]
    return f"The current temperature in {city} is {temperature}°C."
  except Exception as e:
    return f"Sorry, I couldn't fetch the weather for {city}. Error: {str(e)}"


agent = create_agent(
  model="google_genai:gemini-3.1-flash-lite-preview",     # brain of the agent
  tools=[get_weather, calculator],     # registering the tools
  system_prompt="""You are an assistant with two tools: weather and calculator.
    Weather:
    Provide current weather only for real cities using the weather tool.
    - If the city is valid, call the weather tool.
    - If the city is fictional or invalid, do not call the tool and reply exactly:
    "It looks like the city you mentioned is invalid or fictional."

    Calculator:
    Use the calculator tool for math calculations.

    Constraints:
    - Only use tools when appropriate.
    - Do not fabricate weather data or calculation results..""" ,
)

result = agent.invoke({
  "messages": [
    # {"role": "user", "content": "What is the weather like in Chennai?"}
    # {"role": "user", "content": "What is 5+80?"}
    # {"role": "user", "content": "Get the temperature in London and add 10 to it."}
    # {"role": "user", "content": "Get the temperature in Olympus Mons and add 10 to it."}
    # {"role": "user", "content": "Get the temperature in London and Paris and return the average."}
    # {"role": "user", "content": "What is the temperature difference between Chennai and Tokyo?"}
    # {"role": "user", "content": "Which is warmer today, Berlin or Madrid?"}
    # {"role": "user", "content": "What is the temperature and add 5 to it."} # without city name
    # {"role": "user", "content": "Add 10 to the temperature in London"}
    {"role": "user", "content": """Tell me the temperature in London and add 10 to it, 
    but for Chennai take the temperature and double it, and for New York subtract 5 
     from whatever the temperature currently is."""}
  ]
})

print(result["messages"][-1].content)
