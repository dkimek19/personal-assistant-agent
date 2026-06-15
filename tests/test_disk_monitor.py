"""Tests for the disk usage warning (assistant.disk_monitor, AC22).

Covers:
- get_directory_size: sums file sizes recursively, handles empty/missing dirs.
- format_disk_warning_message: human-readable warning text.
- DiskWarningState: was_sent / mark_sent / clear round-trip via tmp_path.
- run_disk_usage_check: below threshold (no warning, state cleared), above
  threshold + not yet sent (warning sent + state marked), above threshold +
  already sent (no duplicate), and missing TELEGRAM_CHAT_ID when a warning
  needs to be sent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from assistant.disk_monitor import (
    DiskWarningState,
    format_disk_warning_message,
    get_directory_size,
    run_disk_usage_check,
)


class TestGetDirectorySize:
    def test_sums_file_sizes_recursively(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"x" * 100)
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_bytes(b"y" * 250)

        assert get_directory_size(tmp_path) == 350

    def test_empty_directory_returns_zero(self, tmp_path):
        assert get_directory_size(tmp_path) == 0

    def test_missing_directory_returns_zero(self, tmp_path):
        assert get_directory_size(tmp_path / "does-not-exist") == 0


class TestFormatDiskWarningMessage:
    def test_includes_path_and_sizes_in_gb(self):
        threshold = 20 * 1024 ** 3
        size = 21 * 1024 ** 3

        message = format_disk_warning_message(size, threshold, "/Users/me/assistant")

        assert "/Users/me/assistant" in message
        assert "21.00 GB" in message
        assert "20 GB" in message


class TestDiskWarningState:
    def test_was_sent_false_initially(self, tmp_path):
        state = DiskWarningState(state_path=tmp_path / "flag")

        assert state.was_sent() is False

    def test_mark_sent_then_was_sent_true(self, tmp_path):
        state = DiskWarningState(state_path=tmp_path / "flag")

        state.mark_sent()

        assert state.was_sent() is True

    def test_clear_resets_state(self, tmp_path):
        state = DiskWarningState(state_path=tmp_path / "flag")
        state.mark_sent()

        state.clear()

        assert state.was_sent() is False

    def test_clear_when_not_sent_is_a_noop(self, tmp_path):
        state = DiskWarningState(state_path=tmp_path / "flag")

        state.clear()  # should not raise

        assert state.was_sent() is False


class TestRunDiskUsageCheck:
    async def test_below_threshold_does_not_send(self, tmp_path):
        (tmp_path / "small.txt").write_bytes(b"x" * 100)
        state = DiskWarningState(state_path=tmp_path / "flag")
        send = AsyncMock()

        result = await run_disk_usage_check(
            path=tmp_path,
            threshold_bytes=1000,
            chat_id=12345,
            state=state,
            send=send,
        )

        assert result["exceeded"] is False
        assert result["sent"] is False
        send.assert_not_awaited()
        assert state.was_sent() is False

    async def test_above_threshold_sends_and_marks_state(self, tmp_path):
        (tmp_path / "big.txt").write_bytes(b"x" * 1000)
        state = DiskWarningState(state_path=tmp_path / "flag")
        send = AsyncMock()

        result = await run_disk_usage_check(
            path=tmp_path,
            threshold_bytes=500,
            chat_id=12345,
            state=state,
            token="fake-token",
            send=send,
        )

        assert result["exceeded"] is True
        assert result["sent"] is True
        send.assert_awaited_once()
        args, kwargs = send.call_args
        assert args[0] == 12345
        assert "0.00 GB" in args[1] or "GB" in args[1]
        assert kwargs["token"] == "fake-token"
        assert state.was_sent() is True

    async def test_above_threshold_already_sent_does_not_resend(self, tmp_path):
        (tmp_path / "big.txt").write_bytes(b"x" * 1000)
        state = DiskWarningState(state_path=tmp_path / "flag")
        state.mark_sent()
        send = AsyncMock()

        result = await run_disk_usage_check(
            path=tmp_path,
            threshold_bytes=500,
            chat_id=12345,
            state=state,
            send=send,
        )

        assert result["exceeded"] is True
        assert result["sent"] is False
        send.assert_not_awaited()

    async def test_dropping_below_threshold_clears_state(self, tmp_path):
        (tmp_path / "small.txt").write_bytes(b"x" * 100)
        state = DiskWarningState(state_path=tmp_path / "flag")
        state.mark_sent()
        send = AsyncMock()

        result = await run_disk_usage_check(
            path=tmp_path,
            threshold_bytes=1000,
            chat_id=12345,
            state=state,
            send=send,
        )

        assert result["exceeded"] is False
        assert state.was_sent() is False
        send.assert_not_awaited()

    async def test_missing_chat_id_raises_runtime_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        (tmp_path / "big.txt").write_bytes(b"x" * 1000)
        state = DiskWarningState(state_path=tmp_path / "flag")
        send = AsyncMock()

        with pytest.raises(RuntimeError):
            await run_disk_usage_check(
                path=tmp_path,
                threshold_bytes=500,
                state=state,
                send=send,
            )

        send.assert_not_awaited()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
