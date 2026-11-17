# Search & Weather AI Agent

A LangChain-powered research assistant that combines web search with live
weather data. Ask a question in the Streamlit chat interface and the agent
decides when to search Tavily and when to call Weatherstack, using Gemini to
produce the final answer.

## What this project contains

- **Streamlit chat UI** in [`app.py`](./app.py), with session history and a
  dark research-console layout.
- **Reusable agent package** in
  [`search_weather_agent/`](./search_weather_agent/), which creates the
  LangChain ReAct agent and contains the weather tool.
- **CLI demo** in [`main.py`](./main.py), useful for testing the agent outside
  Streamlit.
- **Response caching** in `app.py`: exact repeated questions are read from
  Supabase when configured, then from the local CSV dataset, and finally from
  the current Streamlit session.
- **Supabase schema** in [`supabase_schema.sql`](./supabase_schema.sql) for
  persistent response storage.
- **Research notebook** in
  [`research/agent_demo.ipynb`](./research/agent_demo.ipynb).

## How it works

1. Load API keys from `.env` locally or Streamlit Secrets in deployment.
2. Build a Gemini ReAct agent with two tools:
   - Tavily Search for current web research.
   - Weatherstack for current city weather.
3. Return the agent's response in the chat interface.
4. Save successful responses for exact-query reuse.

## Project structure

```text
.
├── app.py                         # Streamlit deployment entrypoint
├── main.py                        # Command-line demo entrypoint
├── search_weather_agent/
│   ├── __init__.py                # Public package exports
│   └── agent.py                   # Agent factory and weather tool
├── data/
│   └── README.md                  # Runtime response dataset notes
├── research/
│   └── agent_demo.ipynb           # Experiments and learning notebook
├── supabase_schema.sql             # Persistent cache table
├── requirements.txt
├── runtime.txt
└── .streamlit/config.toml
```

`data/agent_responses.csv` is created at runtime and is intentionally ignored
by Git. Streamlit Cloud storage is temporary; use Supabase for persistence.

## Setup

### Requirements

- Python 3.11
- API keys for Google Gemini, Tavily, and Weatherstack
- Optional Supabase project for persistent response storage

### Install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```dotenv
GOOGLE_API_KEY="your-google-api-key"
TAVILY_API_KEY="your-tavily-api-key"
WEATHERSTACK_API_KEY="your-weatherstack-api-key"
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SECRET_KEY="your-supabase-secret-key"
```

Never commit `.env` or API keys. For Streamlit Cloud, add the same values in
the app's **Secrets** settings and select Python 3.11.

If Supabase is enabled, run [`supabase_schema.sql`](./supabase_schema.sql) once
in the Supabase SQL Editor.

## Run

Start the Streamlit application:

```bash
streamlit run app.py
```

Run the CLI example:

```bash
python main.py
```

Open the notebook with Jupyter:

```bash
jupyter notebook research/agent_demo.ipynb
```

## Notes

- The agent uses the hosted LangChain ReAct prompt from LangChain Hub.
- Network calls require valid API keys and an active internet connection.
- Repeated exact questions can be served without another Gemini or Tavily
  request when a saved response is available.
