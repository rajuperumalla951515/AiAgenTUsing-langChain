import os
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st
import certifi
from dotenv import load_dotenv
from supabase import Client, create_client

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
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_SECRET_KEY = get_secret("SUPABASE_SECRET_KEY")

RESPONSE_DATASET = Path(__file__).parent / "data" / "agent_responses.csv"


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    except Exception:
        return None


def get_saved_response(query: str) -> str | None:
    query_key = normalize_query(query)
    response_cache = st.session_state.setdefault("response_cache", {})
    if query_key in response_cache:
        return response_cache[query_key]

    supabase = get_supabase_client()
    if supabase:
        try:
            result = (
                supabase.table("agent_responses")
                .select("response")
                .eq("query_key", query_key)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if result.data:
                response = result.data[0]["response"]
                response_cache[query_key] = response
                return response
        except Exception:
            pass

    if not RESPONSE_DATASET.exists():
        return None

    try:
        with RESPONSE_DATASET.open("r", encoding="utf-8", newline="") as dataset_file:
            records = list(csv.DictReader(dataset_file))
        for record in reversed(records):
            if record.get("query_key") == query_key:
                response_cache[query_key] = record["response"]
                return record["response"]
    except (OSError, csv.Error, KeyError):
        return None

    return None


def save_response(query: str, response: str) -> None:
    query_key = normalize_query(query)
    st.session_state.setdefault("response_cache", {})[query_key] = response
    record = {
        "query": query,
        "query_key": query_key,
        "response": response,
        "model": "gemini-3.6-flash",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    supabase = get_supabase_client()
    if supabase:
        try:
            supabase.table("agent_responses").insert(record).execute()
        except Exception:
            pass

    try:
        RESPONSE_DATASET.parent.mkdir(parents=True, exist_ok=True)
        file_exists = RESPONSE_DATASET.exists()
        with RESPONSE_DATASET.open("a", encoding="utf-8", newline="") as dataset_file:
            writer = csv.DictWriter(dataset_file, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except OSError:
        # The response remains available in session memory if the host is read-only.
        pass

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
        --ink: #f5f1fa;
        --muted: #aaa1b5;
        --line: rgba(255, 255, 255, 0.12);
        --surface: #17121e;
        --panel: #261c31;
        --accent: #bf65f6;
        --accent-soft: #38214b;
    }

    html, body, [class*="css"], .stApp {
        font-family: "Google Sans", "Google Sans Flex", "Google Sans Text", sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: #100d15;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #17131d;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.35rem 1rem;
    }

    .desk-mark {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 1.1rem;
    }

    .desk-symbol {
        display: grid;
        width: 2rem;
        height: 2rem;
        place-items: center;
        border-radius: 7px;
        background: #33273f;
        color: #d4a5ff;
        font-size: 1.25rem;
    }

    .desk-name {
        color: #f4eff8;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .desk-kicker {
        margin-top: 0.15rem;
        color: #8f8798;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .main-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px 20px 0 0;
        background: linear-gradient(135deg, #32213e 0%, #1f172b 48%, #17121f 100%);
        margin: 0 auto;
        max-width: 930px;
        padding: 1rem 1.4rem;
    }

    .main-head h1 {
        margin: 0;
        color: #f5f0fa;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0;
    }

    .main-head p {
        display: none;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 999px;
        background: rgba(0, 0, 0, 0.28);
        color: #d9cfe2;
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
        background: #9b4de0;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        background: #251c30;
        margin-bottom: 0.8rem;
        padding: 1rem 1.1rem;
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.6;
    }

    [data-testid="stChatInput"] {
        border-color: rgba(205, 128, 255, 0.55);
        background: #21182a;
    }

    .sidebar-label {
        color: #877d91;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .sidebar-rule {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1.25rem 0;
    }

    .tool-row {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        color: #e6dfea;
        font-size: 0.82rem;
        padding: 0.7rem 0;
    }

    .tool-row span:last-child {
        color: #c277f4;
        font-weight: 700;
    }

    .stButton > button {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 6px;
        background: #2b2136;
        color: #e9e1ee;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #a65bda;
        color: #d39af4;
    }

    .workspace {
        max-width: 930px;
        min-height: 76vh;
        margin: 0 auto;
        padding: 0 3.2rem 2rem;
        background: radial-gradient(circle at 50% 18%, #392047 0%, #21182c 38%, #17121f 78%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 0;
        border-radius: 0 0 20px 20px;
    }

    .welcome {
        padding: 6rem 0 2.4rem;
        text-align: center;
    }

    .welcome-orb {
        width: 4.1rem;
        height: 4.1rem;
        margin: 0 auto 1.4rem;
        border: 1px solid #c77df6;
        border-radius: 50%;
        background: radial-gradient(circle at 32% 25%, #e8bbff, #8334a8 54%, #302367);
        box-shadow: 0 0 26px rgba(183, 91, 241, 0.42);
    }

    .welcome h1 {
        margin: 0;
        color: #eee6f4;
        font-size: clamp(1.5rem, 3vw, 2rem);
        font-weight: 450;
    }

    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.55rem;
        margin-bottom: 1.2rem;
    }

    .quick-action {
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 999px;
        background: rgba(25, 17, 32, 0.62);
        color: #d9cfe1;
        font-size: 0.76rem;
        padding: 0.52rem 0.85rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.7rem;
        margin-top: 1rem;
    }

    .feature-card {
        min-height: 5rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        background: rgba(12, 9, 16, 0.3);
        padding: 0.9rem;
    }

    .feature-card strong {
        display: block;
        color: #eee5f4;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .feature-card span {
        display: block;
        margin-top: 0.4rem;
        color: #978b9e;
        font-size: 0.72rem;
    }

    .app-greeting {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        max-width: 930px;
        margin: 0 auto 1.25rem;
        padding: 0 0.2rem;
    }

    .main-logo {
        display: grid;
        width: 2.6rem;
        height: 2.6rem;
        flex: 0 0 2.6rem;
        place-items: center;
        border: 1px solid rgba(210, 143, 255, 0.6);
        border-radius: 50%;
        background: radial-gradient(circle at 30% 25%, #e8bdff, #8135a9 55%, #30215b);
        box-shadow: 0 0 20px rgba(183, 91, 241, 0.35);
        color: #fff;
        font-size: 1.25rem;
    }

    .greeting-text {
        color: #f3edf7;
        font-size: 1rem;
        font-weight: 500;
    }

    .greeting-subtext {
        margin-top: 0.18rem;
        color: #918699;
        font-size: 0.76rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearchResults(max_results=1)

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


@st.cache_resource(show_spinner=False)
def get_agent_executor():
    """Build the agent once instead of rebuilding it on every chat rerun."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
        max_retries=2,
    )
    prompt = hub.pull("hwchase17/react")
    tools = [search_tool, get_weather_data]
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        max_execution_time=45,
    )


agent_executor = get_agent_executor()

# ==========================================
# CHAT INTERFACE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "greeting" not in st.session_state:
    st.session_state.greeting = random.choice(
        [
            "Hello buddy, how are you?",
            "Hey there, what are we exploring today?",
            "Good to see you. What can I help with?",
            "Welcome back. Ready when you are.",
            "Hi there. What is on your mind?",
        ]
    )

st.markdown(
    f"""
    <div class="app-greeting">
        <div class="main-logo">◈</div>
        <div>
            <div class="greeting-text">{st.session_state.greeting}</div>
            <div class="greeting-subtext">Your AI research assistant is ready.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        <div class="sidebar-label">Workspace</div>
        <div class="tool-row"><span>Active chat</span><span>{}</span></div>
        <div class="tool-row"><span>Chat</span><span>›</span></div>
        <div class="tool-row"><span>Archived</span><span>›</span></div>
        <div class="tool-row"><span>Library</span><span>›</span></div>
        <div class="sidebar-rule"></div>
        <div class="sidebar-label">Projects</div>
        <div class="tool-row"><span>New project</span><span>＋</span></div>
        <div class="tool-row"><span>Weather research</span><span>›</span></div>
        """.format(len(st.session_state.get("messages", []))),
        unsafe_allow_html=True,
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask anything...")

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
            saved_answer = get_saved_response(user_query)
            if saved_answer is not None:
                answer = saved_answer
                st.caption("Saved response")
                st.markdown(answer)
            else:
                with st.spinner("Agent is thinking..."):
                    try:
                        response = agent_executor.invoke({"input": agent_input})
                        answer = response["output"]
                        save_response(user_query, answer)
                        st.markdown(answer)
                    except Exception as error:
                        answer = f"Error: {error}"
                        st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )