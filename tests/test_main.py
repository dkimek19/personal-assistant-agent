"""Tests for the Web UI service entry point (assistant.main, AC20)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import assistant.main as main_module
from assistant.interfaces.web_ui import app


class TestMain:
    def test_runs_uvicorn_with_web_ui_app(self):
        with patch.object(main_module.uvicorn, "run") as mock_run:
            main_module.main()

        mock_run.assert_called_once_with(
            app, host=main_module.HOST, port=main_module.PORT, log_level="info"
        )

    def test_app_is_web_ui_app(self):
        assert main_module.app is app


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
