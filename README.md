# FalloutAssist

The active application is the LangChain and LangGraph Jira implementation in `langchain-falloutassist/`. It combines the Jira specialist, Rovo and local retrieval, reranking, ingestion controls, Jira scan/commenting, source quality, feedback, rules, settings, and audit in one dashboard and one server.

## Run the app

From this repository root, run:

```powershell
& .\langchain-falloutassist\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

Open <http://127.0.0.1:8010>. Use the `langchain-falloutassist/README.md` for setup, database, Jira/Rovo, worker, and configuration details.

The former application entry point is retained as `legacy_app.py` for reference; `app.py` now starts the active LangChain application. Do not start a second FalloutAssist server on another port.
