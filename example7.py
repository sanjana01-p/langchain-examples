from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv() 

os.environ["GEMINI_API_KEY"] = "AIzaSyAD_Kx0qJwlE3asyRjuDjLY_mcnfQ9bqDI"


# Initialize Tavily Search tool
tavily_tool = TavilySearch(
    max_results=10,
    topic="general",
    search_depth = "advanced"
)

writer_agent = create_agent(
    model = "google_genai:gemini-3.1-pro-preview",
    system_prompt = """You are a creative content writer who writes interesting and
      engaging content on a given topic. You have acecss to web search capabilities. Before writing,
      use the search tool to research the topic thoroughly to ensure accuracy and include latest information.""",
      tools=[tavily_tool]
)

print("Writer agent with search tool created successully \n")

editor_agent = create_agent(
    model = "google_genai:gemini-3.1-pro-preview",
    system_prompt = """You are an editor who reviews the content written by content writer and 
    edits the content to make it more impactful and reach the users. 
    Check for grammatical errors and correct it."""
)

print("Editor agent created succesfully \n")

def run_sequential_pipeline(topic: str):

    """
    Sequential multi-agent pipeline.
    1. Writer agent creates content on a given topic.
    2. Editor agent refines that content.
    """
    print(f"Topic: {topic}\n")
        
    writer_result = writer_agent.invoke(
        {
             "messages": [
                HumanMessage(
                    content=f"""Please write a detailed article on the given topic: '{topic}'
                    
                    Instructions:
                    1. First, search for latest information and trends about this topic
                    2. Then write a comprehensive articleincorporating your research
                    3. Include relevant facts, statistics, and current developments.
                    """
                )
            ]
        }
    )

    writer_message = writer_result["messages"]
    written_content = writer_message[-1].content

    print("=== Writer Agent's output delivered to Editor Agent ===")

    print(" === Editor agent refining content === ")


    editor_result = editor_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"""Please refine the following article: '{written_content}'
                    
                    Focus on:
                    - Clarity and Flow
                    - Grammar and style 
                    - Structure and readabilty
                    - Fact consistency
                    """
                )
            ]
        }
    )

    editor_message = editor_result["messages"]
    refined_content = editor_message[-1].content

    print("=== Editor Agent Output is ready and delivered.")

    return {
        "topic": topic,
        "draft": written_content,
        "final": refined_content
    }

if __name__=="__main__":
    topic = "The Future of Artificial Intelligence in Everyday Life in 2026"
    result = run_sequential_pipeline(topic)

    print("\n== Final Output ==")
    print(f"Topic:{result['topic']}\n")

    print("Draft Content:\n")
    print(result['draft'])

    print("\nRefined Content:\n")
    print(result['final'])