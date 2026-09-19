import os
import certifi


import requests

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


from langchain.agents import (
    create_react_agent,
    AgentExecutor
)

from langchain import hub

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

missing_keys = [
    name
    for name, value in {
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
        "WEATHERSTACK_API_KEY": WEATHERSTACK_API_KEY,
    }.items()
    if not value
]
if missing_keys:
    raise RuntimeError(
        "Missing required environment variables: " + ", ".join(missing_keys)
    )

# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearchResults(max_results=2)

# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# ==========================================
# LLM
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# ==========================================
# PROMPT
# ==========================================

prompt = hub.pull("hwchase17/react")

# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]

# ==========================================
# CREATE AGENT
# ==========================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ==========================================
# RUN
# ==========================================

response = agent_executor.invoke({
    "input": (
        "Find the capital of India "
        "and then find its current weather."
    )
})

print("\n========================")
print("FINAL OUTPUT")
print("========================\n")

print(response["output"])