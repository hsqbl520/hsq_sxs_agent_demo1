# Socratic Dialogue MVP

FastAPI backend for a Socratic questioning agent. This project includes:

- structured extraction
- rule-based judging
- finite-state transitions (`S0`-`S5`)
- question planning and light rewrite

## Project layout

```text
socratic-dialogue-mvp/
├── app/               # FastAPI app and web UI
├── tests/             # API and evaluation tests
├── run.sh             # install deps and start server
├── test.sh            # run pytest
└── eval.sh            # run gold evaluation
```

## Environment setup

### Option A: use the existing Windows Conda Python

The current scripts default to:

```text
/mnt/c/Users/10985/miniconda3/python.exe
```

If that path exists on your machine, you can use the project scripts directly.

### Option B: use your own Python 3.11+

If that Conda path does not exist, either:

1. edit `run.sh`, `test.sh`, and `eval.sh` to point to your own Python, or
2. create a virtual environment and run the commands manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

## Required configuration

Create a local `.env` file in this directory. Do not commit it.

Example:

```env
EXTRACTOR_MODE=llm
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your_api_key_here
```

Useful variables:

- `EXTRACTOR_MODE=llm|mock`
- `LLM_API_KEY=...`
- `LLM_BASE_URL=...`
- `LLM_MODEL=...`
- `DATABASE_URL=sqlite:///./socratic.db` for local SQLite
- `REDIS_URL=...` if runtime cache is needed

If you do not want to call a real model during development, use:

```env
EXTRACTOR_MODE=mock
```

## Run locally

From this directory:

```bash
cd /mnt/d/colab/hsq_sxs_agent_demo1/socratic-dialogue-mvp
bash run.sh
```

After startup:

- chat UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Common commands

Start API:

```bash
bash run.sh
```

Run tests:

```bash
bash test.sh
```

Run evaluation:

```bash
bash eval.sh
```

`run.sh` auto-loads `.env` if present.

## Collaboration notes

- Never commit `.env`, local databases, or virtual environments.
- If you change API behavior, update tests under `tests/`.
- If you change onboarding steps, keep this README in sync.
