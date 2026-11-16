# Wine Study Assistant

A retrieval-augmented chatbot for students preparing wine certifications such as WSET Levels 2 and 3.
It answers only from a curated collection of 189 English Wikipedia articles (CC BY-SA 4.0), cites its
sources and says so when a topic is not covered. Built exclusively with free and open-source tools.

> Independent study aid, not affiliated with or endorsed by WSET or any certification body.

## How it works

1. The question is rewritten with the recent conversation when it is a follow-up ("And in Austria?").
2. The rewritten question is embedded with `BAAI/bge-small-en-v1.5` and searched in a FAISS index of 3,197 chunks.
3. Maximal Marginal Relevance selects five relevant but diverse passages.
4. If no passage reaches the relevance threshold, the assistant declines and suggests related topics.
5. Otherwise an open LLM answers from the passages only, citing each statement.
6. Every turn is logged anonymously in JSON Lines.

## Project structure

```
wine-assistant/
├── app.py                  # Streamlit interface
├── rag.py                  # retrieval, prompting, generation, memory and logging
├── config.yaml             # every tunable parameter (models, top-k, threshold, prompts...)
├── requirements.txt        # dependencies of the deployed app
├── data/
│   ├── chunks.jsonl        # knowledge base chunks with title, section and URL
│   └── wine_kb.faiss       # vector index (row i of chunks.jsonl <-> vector i)
└── .streamlit/secrets.toml # Groq API key, local only (excluded by .gitignore)
```

The knowledge base and the index are built in the development notebook (`Tarea 4.ipynb`).

## Models and licenses

| Role | Model | License |
| --- | --- | --- |
| LLM, deployed | `openai/gpt-oss-20b` through the Groq free tier | Apache 2.0 |
| LLM, local development | `Qwen/Qwen3-4B-Instruct-2507` with Transformers | Apache 2.0 |
| Embeddings | `BAAI/bge-small-en-v1.5` | MIT |

The provider is selected with `llm.provider` in `config.yaml` (`groq` or `local`).

## Run locally

```bash
# from this folder, with a Python 3.12+ virtual environment activated
pip install -r requirements.txt
# put your key in .streamlit/secrets.toml:  GROQ_API_KEY = "..."
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud

**Live app:** https://wine-study-assistant.streamlit.app

The app runs on the free tier of Streamlit Community Cloud and calls the LLM through the Groq free tier,
so the whole deployment costs nothing. To reproduce it:

1. **Get a Groq API key.** Create a free account at [console.groq.com](https://console.groq.com) and
   generate a key under *API Keys*. Groq shows the key only once, so store it safely.
2. **Put the code on GitHub.** Fork this repository, or push this folder to a new public repository.
   The knowledge base files in `data/` are included (about 8 MB in total), so no build step is needed.
   Never commit `.streamlit/secrets.toml`; it is already excluded by `.gitignore`.
3. **Create the app.** Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub
   account and choose *Create app → Deploy a public app from GitHub*. Select the repository, the
   branch `main` and the main file `app.py`.
4. **Configure it before deploying.** Open *Advanced settings*:
   - *Python version*: 3.12 or later (the reference deployment uses 3.14).
   - *Secrets*: add one line with your key, `GROQ_API_KEY = "your-groq-api-key"`.
     The app reads it with `st.secrets`, so the key never appears in the code.
5. **Deploy.** The first start takes a few minutes, because the platform installs the CPU build of
   PyTorch and downloads the embedding model. Later visits start in seconds, unless the app has been
   put to sleep after a period without traffic, in which case the first visitor wakes it up.

To change the key later, open the app menu and go to *Settings → Secrets*; the app restarts with
the new value. Every push to `main` redeploys the app automatically.

**Free-tier limits.** Streamlit Community Cloud gives each app about 1 GB of memory, which is why
the deployed version calls a hosted LLM instead of loading one. The Groq free tier allows 30 requests
per minute and 1,000 per day for `openai/gpt-oss-20b`; when the quota is exhausted, the app shows a
message asking the user to wait a minute.

**Logs.** Each turn is printed as a JSON line, so it appears in the app logs of the Streamlit
dashboard (*Manage app*). The local `logs/interactions.jsonl` file is not persistent on the free
platform, which restarts containers without keeping their files.
