"""Main entry point for the personal assistant agent's Web UI service (AC20).

Runs the FastAPI Web UI app (:mod:`assistant.interfaces.web_ui`) via
uvicorn. This is the long-running process that a launchd agent (see
:mod:`assistant.launchd`) supervises and automatically restarts if it exits
or crashes.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import logging

import uvicorn

from assistant.interfaces.web_ui import app

#: Host/port the Web UI service binds to.
HOST: str = "127.0.0.1"
PORT: int = 8000


def main() -> None:
    """Run the Web UI service via uvicorn (blocks until the server stops)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
