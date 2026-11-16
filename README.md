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

*(Step-by-step instructions are added after the first deployment.)*
