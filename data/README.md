# Agent response dataset

The app creates `agent_responses.csv` here at runtime. Each row stores the
original question, a normalized lookup key, the Gemini response, model name,
and UTC creation time. Exact repeated questions are served from this dataset
and session memory without another Gemini or Tavily request.

Streamlit Cloud storage is temporary and can be cleared when the app restarts.
For persistent storage, run `supabase_schema.sql` in Supabase SQL Editor and
configure `SUPABASE_URL` and `SUPABASE_SECRET_KEY`. The app uses Supabase first
and falls back to the local CSV when those variables are unavailable.