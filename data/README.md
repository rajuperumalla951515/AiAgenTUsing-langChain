# Agent response dataset

The app creates `agent_responses.jsonl` here at runtime. Each line stores the
original question, a normalized lookup key, the Gemini response, model name,
and UTC creation time. Exact repeated questions are served from this dataset
and session memory without another Gemini or Tavily request.

Streamlit Cloud storage is temporary and can be cleared when the app restarts.
Use an external database or object store if responses must persist permanently.