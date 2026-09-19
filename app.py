import os
import requests
import streamlit as st
import certifi
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

def get_secret(name: str) -> str | None:
    return os.getenv(name) or st.secrets.get(name)


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
WEATHERSTACK_API_KEY = get_secret("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================


st.set_page_config(
    page_title="Signal // AI Research Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.error("Missing deployment secrets: " + ", ".join(missing_keys))
    st.stop()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,wght@6..144,400..700&display=swap');

    :root {
        --ink: #17212b;
        --muted: #687783;
        --line: #dbe3e8;
        --surface: #f7f9fa;
        --panel: #ffffff;
        --accent: #087f8c;
        --accent-soft: #e3f4f3;
    }

    html, body, [class*="css"], .stApp {
        font-family: "Google Sans", "Google Sans Flex", "Google Sans Text", sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: #f4f7f8;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #eef3f4;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.35rem;
    }

    .desk-mark {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 2.2rem;
    }

    .desk-symbol {
        display: grid;
        width: 2rem;
        height: 2rem;
        place-items: center;
        border-radius: 7px;
        background: var(--ink);
        color: #9fe3df;
        font-size: 1.25rem;
    }

    .desk-name {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .desk-kicker {
        margin-top: 0.15rem;
        color: var(--muted);
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .main-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 2.2rem;
        padding-bottom: 1.25rem;
    }

    .main-head h1 {
        margin: 0;
        color: var(--ink);
        font-size: clamp(1.7rem, 3vw, 2.5rem);
        font-weight: 650;
        letter-spacing: -0.02em;
    }

    .main-head p {
        margin: 0.45rem 0 0;
        color: var(--muted);
        font-size: 0.9rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border: 1px solid #b8dfdc;
        border-radius: 999px;
        background: var(--accent-soft);
        color: #17656b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 0.45rem 0.7rem;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .status-dot {
        width: 0.42rem;
        height: 0.42rem;
        border-radius: 50%;
        background: #16a085;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        margin-bottom: 0.8rem;
        padding: 1rem 1.1rem;
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.6;
    }

    [data-testid="stChatInput"] {
        border-color: #b9cbd0;
    }

    .sidebar-label {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .sidebar-rule {
        border-top: 1px solid var(--line);
        margin: 1.25rem 0;
    }

    .tool-row {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid var(--line);
        color: var(--ink);
        font-size: 0.82rem;
        padding: 0.7rem 0;
    }

    .tool-row span:last-child {
        color: var(--accent);
        font-weight: 700;
    }

    .stButton > button {
        border: 1px solid #b9cbd0;
        border-radius: 6px;
        background: var(--panel);
        color: var(--ink);
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-head">
        <div>
            <h1>Research desk</h1>
            <p>Search, weather intelligence, and grounded answers in one workspace.</p>
        </div>
        <div class="status-pill"><span class="status-dot"></span>Systems ready</div>
    </div>
    """,
    unsafe_allow_html=True,
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
    verbose=True,
    handle_parsing_errors=True
)

# ==========================================
# CHAT INTERFACE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
        <div class="desk-mark">
            <div class="desk-symbol">◈</div>
            <div>
                <div class="desk-name">SIGNAL / AI</div>
                <div class="desk-kicker">Research console</div>
            </div>
        </div>
        <div class="sidebar-label">Session</div>
        <div class="tool-row"><span>Conversation turns</span><span>{}</span></div>
        <div class="sidebar-rule"></div>
        <div class="sidebar-label">Connected tools</div>
        <div class="tool-row"><span>Web search</span><span>ON</span></div>
        <div class="tool-row"><span>Weather data</span><span>ON</span></div>
        """.format(len(st.session_state.get("messages", []))),
        unsafe_allow_html=True,
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input(
    "Ask about the weather, news, or anything you want to research"
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    conversation = "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in st.session_state.messages[:-1]
    )
    agent_input = (
        f"Previous conversation:\n{conversation}\n\n"
        f"Current user question: {user_query}"
        if conversation
        else user_query
    )

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_executor.invoke({"input": agent_input})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                answer = f"Error: {error}"
                st.error(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )