from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import  tool
from langchain.messages import  HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def get_current_weather(location: str) -> str:
    """This tool searches web and return the weather of a city

    Args:
        location (str): the name of the city to search weather for

    Returns:
        str: the weather of the city
    """
    # In a real implementation, you would use an API to get the weather data.
    # For this example, we'll just return a dummy weather report.
    return tavily_client.search(f"current weather in {location}").get("summary", "No weather information found.")
agent=create_agent(
    tools=[get_current_weather],
    model=ChatOpenAI(model="gpt-5-mini")
    
)


def main():
    print("Hello from langchain-course!")
    location = input("Enter the city name for weather forecast: ")
    result = agent.invoke(
        {"messages": [HumanMessage(content=f"What is the current weather in {location}?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
