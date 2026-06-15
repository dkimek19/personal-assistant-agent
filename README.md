# Personal Assistant Agent

A single-user personal assistant agent powered by a local LLM (Ollama). Start a
conversation from the Web UI, Telegram, or Discord and it shares the same
session (memory) across all of them, calling tools for calendar, tasks,
weather, web search, documents, and code execution as needed.

## Key Features

- **Unified multi-interface sessions** — conversations from the Web UI,
  Telegram, and Discord share a single `working_memory`
  (`assistant/session_resolver.py`, `assistant/session_store.py`)
- **Tool calling** — uses Ollama's `/api/chat` tool calling to automatically
  invoke the tools below when needed (`assistant/agent_core.py`,
  `assistant/tools/`)
  - Weather lookup (Open-Meteo)
  - Google Calendar: read/create/update/delete events
  - Google Tasks: read/create/update/complete tasks
  - Web search + summarization (SearXNG)
  - Read and create PDF/DOCX documents
  - Code execution in a Docker sandbox
- **Slash commands**
  - `/note` — save a note to SQLite (`assistant/notes.py`)
  - `/remember` — save to long-term memory (`assistant/long_term_memory.py`)
  - `/compress` — manually compress the conversation context (auto-compression
    also kicks in once the context grows too large) (`assistant/compression.py`)
- **Automation (macOS launchd)** — `assistant/launchd.py`
  - Keeps the main Web UI running (auto-restart on crash)
  - Daily SQLite backup at 3 AM, with backups older than 30 days purged
    automatically
  - Telegram reminder 30 minutes before calendar events, checked every 5
    minutes
  - Hourly disk usage check, with a Telegram warning if usage exceeds 20GB

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package/virtualenv management)
- [Ollama](https://ollama.com/) running locally with the desired model pulled
  (default: `gemma4:12b-mlx`)
- Docker (for the code execution tool)
- A SearXNG instance (for the web search tool)
- Google Cloud OAuth client (for the Calendar/Tasks tools)

## Installation

```bash
uv sync
```

## Environment Variables (`.env`)

Create a `.env` file in the project root and fill in the values below (it is
excluded from commits via `.gitignore`).

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_URL` | Ollama server address | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name to use | `gemma4:12b-mlx` |
| `SEARXNG_URL` | SearXNG instance address | `http://localhost:8888` |
| `TELEGRAM_TOKEN` | Telegram bot token (for the Telegram interface) | - |
| `DISCORD_TOKEN` | Discord bot token (for the Discord interface) | - |
| `ASSISTANT_CREDENTIALS_DIR` | Location of Google OAuth `credentials.json` / token files | `~/assistant/credentials` |

To use Google Calendar/Tasks, place the `credentials.json` you obtained from
the Google Cloud Console under `ASSISTANT_CREDENTIALS_DIR`. On first use, an
OAuth flow runs and a token file is created in the same directory.

## Running

### Web UI

```bash
uv run python -m assistant.main
```

Open `http://127.0.0.1:8000` for the chat UI plus weather/calendar/notes
widgets.

### Telegram bot

```bash
uv run python -m assistant.interfaces.telegram_bot
```

### Discord bot

```bash
uv run python -m assistant.interfaces.discord_bot
```

## Registering Background Automation (macOS)

```bash
uv run python -c "from assistant.launchd import install_all; install_all()"
```

This registers and loads four launchd jobs into `~/Library/LaunchAgents`: the
main agent (`com.personalassistant.agent`), backups
(`com.personalassistant.backup`), calendar alerts
(`com.personalassistant.calendar-alerts`), and disk monitoring
(`com.personalassistant.disk-monitor`).

## Testing

```bash
uv run pytest -q
```

Most tests run without any external services. A few SLA/integration tests that
require a live Ollama/SearXNG connection are automatically skipped if those
services aren't available.

## Project Structure

```
assistant/
  agent_core.py       # shared entry point for all interfaces (handle_user_message)
  session_store.py    # SQLite-backed shared working_memory / long-term memory
  session_resolver.py # maps per-interface users to a common user_id/session_id
  notes.py            # /note
  long_term_memory.py # /remember
  compression.py      # /compress, automatic context compression
  backup.py           # DB backup/cleanup
  calendar_alerts.py  # calendar reminders (Telegram)
  disk_monitor.py     # disk usage warnings (Telegram)
  launchd.py          # macOS launchd plist generation/installation
  interfaces/         # web_ui (FastAPI), telegram_bot, discord_bot
  tools/              # weather, calendar, tasks, searxng, documents, code_execution
tests/                # pytest tests (836 passed, 3 skipped)
```
