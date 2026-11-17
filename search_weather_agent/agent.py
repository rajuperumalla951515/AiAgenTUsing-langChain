"""Build the LangChain agent and expose its weather tool."""

from __future__ import annotations

import os

import certifi
import requests

os.environ["SSL_CERT_FILE"] = certifi.where()


def _load_react_prompt():
    """Load the hosted prompt across supported LangChain package layouts."""
    try:
        from langchain import hub
    except ImportError:
        from langchainhub import Client

        return Client().pull("hwchase17/react")
    return hub.pull("hwchase17/react")


def _fetch_weather_data(city: str, weatherstack_api_key: str) -> str:
    url = (
        "https://api.weatherstack.com/current?"
        f"access_key={weatherstack_api_key}&query={city}"
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


def create_weather_tool(weatherstack_api_key: str):
    """Create a weather tool with its API key kept outside the agent input."""
    from langchain.tools import tool

    @tool
    def get_weather_data(city: str) -> str:
        """Fetch current weather information for a city."""
        return _fetch_weather_data(city, weatherstack_api_key)

    return get_weather_data


def build_agent_executor(
    google_api_key: str,
    tavily_api_key: str,
    weatherstack_api_key: str,
    *,
    max_results: int = 1,
) -> AgentExecutor:
    """Create the search-and-weather agent with the supplied API keys."""
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_google_genai import ChatGoogleGenerativeAI

    os.environ["TAVILY_API_KEY"] = tavily_api_key
    search_tool = TavilySearchResults(max_results=max_results)
    weather_tool = create_weather_tool(weatherstack_api_key)
    tools = [search_tool, weather_tool]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=google_api_key,
        max_retries=2,
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=_load_react_prompt(),
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        max_execution_time=45,
    )
