import os

from dotenv import load_dotenv

from search_weather_agent import build_agent_executor

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

agent_executor = build_agent_executor(
    google_api_key=GOOGLE_API_KEY,
    tavily_api_key=TAVILY_API_KEY,
    weatherstack_api_key=WEATHERSTACK_API_KEY,
    max_results=2,
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