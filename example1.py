from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import os 
os.environ["GEMINI_API_KEY"] = "AIzaSyBuc0AMmj_q2OI4gihiU6zFR1j29oSEG8c"


model = init_chat_model("google_genai:gemini-3.1-flash-lite-preview")


response =model.invoke("Why do parrots talk?")

print(response)