from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os 

from dotenv import load_dotenv
load_dotenv() 

# role based prompt using template variables

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a seasoned sales person, experienced in writing persuasive "
     "product descriptions. You write compelling product descriptions for "
     "e-commerce applications."
    ),
    ("user",
     """Write a product description for the {query} using the following details:
     Product Name
     Category
     Features
     Dimensions
     Price
     Release Date"""
    )
])


llm_api_key = os.getenv("GOOGLE_API_KEY")

if not llm_api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key = llm_api_key,
)
# initiaizes the llm
pipeline = prompt | model

# pass some generic input and let the model figure it out
response = pipeline.invoke({
    "query": """LG 4KTV XY21232AB, Electronics, 120 w sound, super clarity, dual glass, magic remote,
    100x80x20cm, Rs.123242/-, June 2025"""
})

print(response.content)