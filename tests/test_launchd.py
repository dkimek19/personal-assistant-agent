"""Tests for the launchd auto-restart agent plist (assistant.launchd, AC20).

Covers:
- build_plist: structure/keys for the launchd job description.
- write_plist: round-trips through plistlib.
- generate_default_plist: uses sys.executable + assistant.main, with
  overridable project_dir/log_dir.
- default_plist_path: standard per-user LaunchAgents location.
- install: writes the plist and creates the log directory, with overridable
  paths via tmp_path.
- load_agent / unload_agent: invoke launchctl via subprocess.run (mocked).
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from assistant.launchd import (
    LABEL,
    LABEL_BACKUP,
    LABEL_CALENDAR_ALERTS,
    LABEL_DISK_MONITOR,
    build_periodic_plist,
    build_plist,
    default_plist_path,
    generate_backup_plist,
    generate_calendar_alerts_plist,
    generate_default_plist,
    generate_disk_monitor_plist,
    install,
    install_all,
    load_agent,
    unload_agent,
    write_plist,
)


class TestBuildPlist:
    def test_returns_expected_structure(self):
        plist = build_plist(
            program_arguments=["/usr/bin/python3", "-m", "assistant.main"],
            working_directory="/some/project",
            stdout_path="/some/logs/out.log",
            stderr_path="/some/logs/err.log",
        )

        assert plist == {
            "Label": LABEL,
            "ProgramArguments": ["/usr/bin/python3", "-m", "assistant.main"],
            "WorkingDirectory": "/some/project",
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": "/some/logs/out.log",
            "StandardErrorPath": "/some/logs/err.log",
        }

    def test_accepts_path_arguments_and_stringifies_them(self):
        plist = build_plist(
            program_arguments=[Path("/usr/bin/python3"), "-m", "assistant.main"],
            working_directory=Path("/some/project"),
            stdout_path=Path("/some/logs/out.log"),
            stderr_path=Path("/some/logs/err.log"),
        )

        assert plist["ProgramArguments"] == ["/usr/bin/python3", "-m", "assistant.main"]
        assert plist["WorkingDirectory"] == "/some/project"
        assert plist["StandardOutPath"] == "/some/logs/out.log"
        assert plist["StandardErrorPath"] == "/some/logs/err.log"

    def test_custom_label_run_at_load_and_keep_alive(self):
        plist = build_plist(
            label="com.example.custom",
            program_arguments=["/bin/true"],
            working_directory="/tmp",
            stdout_path="/tmp/out.log",
            stderr_path="/tmp/err.log",
            run_at_load=False,
            keep_alive={"SuccessfulExit": False},
        )

        assert plist["Label"] == "com.example.custom"
        assert plist["RunAtLoad"] is False
        assert plist["KeepAlive"] == {"SuccessfulExit": False}


class TestWritePlist:
    def test_round_trips_via_plistlib(self, tmp_path):
        plist = build_plist(
            program_arguments=["/usr/bin/python3", "-m", "assistant.main"],
            working_directory="/some/project",
            stdout_path="/some/logs/out.log",
            stderr_path="/some/logs/err.log",
        )
        dest = tmp_path / "nested" / "agent.plist"

        result = write_plist(plist, dest)

        assert result == dest
        assert dest.is_file()
        with dest.open("rb") as f:
            loaded = plistlib.load(f)
        assert loaded == plist


class TestGenerateDefaultPlist:
    def test_uses_sys_executable_and_assistant_main(self, tmp_path):
        plist = generate_default_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert plist["ProgramArguments"] == [sys.executable, "-m", "assistant.main"]

    def test_uses_provided_project_dir_and_log_dir(self, tmp_path):
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        plist = generate_default_plist(project_dir=project_dir, log_dir=log_dir)

        assert plist["WorkingDirectory"] == str(project_dir)
        assert plist["StandardOutPath"] == str(log_dir / "assistant.out.log")
        assert plist["StandardErrorPath"] == str(log_dir / "assistant.err.log")

    def test_defaults_run_at_load_and_keep_alive_true(self, tmp_path):
        plist = generate_default_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True

    def test_default_project_dir_is_project_root(self):
        plist = generate_default_plist(log_dir=Path("/tmp/logs"))

        project_root = Path(__file__).resolve().parent.parent
        assert plist["WorkingDirectory"] == str(project_root)

    def test_default_log_dir_is_under_home_assistant_logs(self):
        plist = generate_default_plist(project_dir=Path("/tmp/project"))

        expected_log_dir = Path.home() / "assistant" / "logs"
        assert plist["StandardOutPath"] == str(expected_log_dir / "assistant.out.log")
        assert plist["StandardErrorPath"] == str(expected_log_dir / "assistant.err.log")


class TestDefaultPlistPath:
    def test_returns_standard_launchagents_path(self):
        path = default_plist_path()

        assert path == Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"

    def test_uses_provided_label(self):
        path = default_plist_path(label="com.example.custom")

        assert path == Path.home() / "Library" / "LaunchAgents" / "com.example.custom.plist"


class TestInstall:
    def test_writes_plist_to_dest_path(self, tmp_path):
        dest_path = tmp_path / "LaunchAgents" / f"{LABEL}.plist"
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        result = install(dest_path=dest_path, project_dir=project_dir, log_dir=log_dir)

        assert result == dest_path
        assert dest_path.is_file()

        with dest_path.open("rb") as f:
            loaded = plistlib.load(f)
        assert loaded["Label"] == LABEL
        assert loaded["ProgramArguments"] == [sys.executable, "-m", "assistant.main"]
        assert loaded["WorkingDirectory"] == str(project_dir)
        assert loaded["StandardOutPath"] == str(log_dir / "assistant.out.log")
        assert loaded["StandardErrorPath"] == str(log_dir / "assistant.err.log")
        assert loaded["RunAtLoad"] is True
        assert loaded["KeepAlive"] is True

    def test_creates_log_directory(self, tmp_path):
        dest_path = tmp_path / "LaunchAgents" / f"{LABEL}.plist"
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()

        install(dest_path=dest_path, project_dir=project_dir, log_dir=log_dir)

        assert log_dir.is_dir()

    def test_default_dest_path_is_default_plist_path(self, tmp_path):
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        with patch("assistant.launchd.write_plist") as mock_write:
            mock_write.return_value = default_plist_path()
            install(project_dir=project_dir, log_dir=log_dir)

        args, _ = mock_write.call_args
        assert args[1] == default_plist_path()


class TestBuildPeriodicPlist:
    def test_returns_expected_structure_with_start_interval(self):
        plist = build_periodic_plist(
            label="com.example.periodic",
            program_arguments=["/usr/bin/python3", "-m", "assistant.disk_monitor"],
            working_directory="/some/project",
            stdout_path="/some/logs/out.log",
            stderr_path="/some/logs/err.log",
            start_interval=3600,
        )

        assert plist == {
            "Label": "com.example.periodic",
            "ProgramArguments": ["/usr/bin/python3", "-m", "assistant.disk_monitor"],
            "WorkingDirectory": "/some/project",
            "RunAtLoad": False,
            "StandardOutPath": "/some/logs/out.log",
            "StandardErrorPath": "/some/logs/err.log",
            "StartInterval": 3600,
        }

    def test_returns_expected_structure_with_start_calendar_interval(self):
        plist = build_periodic_plist(
            label="com.example.daily",
            program_arguments=["/usr/bin/python3", "-m", "assistant.backup"],
            working_directory="/some/project",
            stdout_path="/some/logs/out.log",
            stderr_path="/some/logs/err.log",
            start_calendar_interval={"Hour": 3, "Minute": 0},
        )

        assert plist["StartCalendarInterval"] == {"Hour": 3, "Minute": 0}
        assert "StartInterval" not in plist

    def test_run_at_load_can_be_enabled(self):
        plist = build_periodic_plist(
            label="com.example.daily",
            program_arguments=["/bin/true"],
            working_directory="/tmp",
            stdout_path="/tmp/out.log",
            stderr_path="/tmp/err.log",
            start_interval=60,
            run_at_load=True,
        )

        assert plist["RunAtLoad"] is True

    def test_no_keep_alive_key(self):
        plist = build_periodic_plist(
            label="com.example.daily",
            program_arguments=["/bin/true"],
            working_directory="/tmp",
            stdout_path="/tmp/out.log",
            stderr_path="/tmp/err.log",
            start_interval=60,
        )

        assert "KeepAlive" not in plist

    def test_raises_if_neither_interval_provided(self):
        with pytest.raises(ValueError):
            build_periodic_plist(
                label="com.example.daily",
                program_arguments=["/bin/true"],
                working_directory="/tmp",
                stdout_path="/tmp/out.log",
                stderr_path="/tmp/err.log",
            )

    def test_raises_if_both_intervals_provided(self):
        with pytest.raises(ValueError):
            build_periodic_plist(
                label="com.example.daily",
                program_arguments=["/bin/true"],
                working_directory="/tmp",
                stdout_path="/tmp/out.log",
                stderr_path="/tmp/err.log",
                start_interval=60,
                start_calendar_interval={"Hour": 3},
            )


class TestGenerateBackupPlist:
    def test_uses_sys_executable_and_assistant_backup(self, tmp_path):
        plist = generate_backup_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert plist["Label"] == LABEL_BACKUP
        assert plist["ProgramArguments"] == [sys.executable, "-m", "assistant.backup"]

    def test_uses_provided_project_dir_and_log_dir(self, tmp_path):
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        plist = generate_backup_plist(project_dir=project_dir, log_dir=log_dir)

        assert plist["WorkingDirectory"] == str(project_dir)
        assert plist["StandardOutPath"] == str(log_dir / "backup.out.log")
        assert plist["StandardErrorPath"] == str(log_dir / "backup.err.log")

    def test_runs_daily_via_start_calendar_interval(self, tmp_path):
        plist = generate_backup_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert "StartCalendarInterval" in plist
        assert "Hour" in plist["StartCalendarInterval"]
        assert "StartInterval" not in plist


class TestGenerateCalendarAlertsPlist:
    def test_uses_sys_executable_and_calendar_alerts_module(self, tmp_path):
        plist = generate_calendar_alerts_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert plist["Label"] == LABEL_CALENDAR_ALERTS
        assert plist["ProgramArguments"] == [sys.executable, "-m", "assistant.calendar_alerts"]

    def test_uses_provided_project_dir_and_log_dir(self, tmp_path):
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        plist = generate_calendar_alerts_plist(project_dir=project_dir, log_dir=log_dir)

        assert plist["WorkingDirectory"] == str(project_dir)
        assert plist["StandardOutPath"] == str(log_dir / "calendar_alerts.out.log")
        assert plist["StandardErrorPath"] == str(log_dir / "calendar_alerts.err.log")

    def test_runs_periodically_via_start_interval(self, tmp_path):
        plist = generate_calendar_alerts_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert isinstance(plist["StartInterval"], int)
        assert plist["StartInterval"] > 0
        assert "StartCalendarInterval" not in plist


class TestGenerateDiskMonitorPlist:
    def test_uses_sys_executable_and_disk_monitor_module(self, tmp_path):
        plist = generate_disk_monitor_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert plist["Label"] == LABEL_DISK_MONITOR
        assert plist["ProgramArguments"] == [sys.executable, "-m", "assistant.disk_monitor"]

    def test_uses_provided_project_dir_and_log_dir(self, tmp_path):
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        plist = generate_disk_monitor_plist(project_dir=project_dir, log_dir=log_dir)

        assert plist["WorkingDirectory"] == str(project_dir)
        assert plist["StandardOutPath"] == str(log_dir / "disk_monitor.out.log")
        assert plist["StandardErrorPath"] == str(log_dir / "disk_monitor.err.log")

    def test_runs_periodically_via_start_interval(self, tmp_path):
        plist = generate_disk_monitor_plist(project_dir=tmp_path / "project", log_dir=tmp_path / "logs")

        assert isinstance(plist["StartInterval"], int)
        assert plist["StartInterval"] > 0
        assert "StartCalendarInterval" not in plist


class TestInstallAll:
    def test_writes_all_four_plists_to_dest_dir(self, tmp_path):
        dest_dir = tmp_path / "LaunchAgents"
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        with patch("assistant.launchd.load_agent") as mock_load:
            paths = install_all(dest_dir=dest_dir, project_dir=project_dir, log_dir=log_dir)

        assert set(paths) == {LABEL, LABEL_BACKUP, LABEL_CALENDAR_ALERTS, LABEL_DISK_MONITOR}
        for label, path in paths.items():
            assert path == dest_dir / f"{label}.plist"
            assert path.is_file()
            with path.open("rb") as f:
                loaded = plistlib.load(f)
            assert loaded["Label"] == label

        assert mock_load.call_count == 4

    def test_creates_log_directory(self, tmp_path):
        dest_dir = tmp_path / "LaunchAgents"
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()

        with patch("assistant.launchd.load_agent"):
            install_all(dest_dir=dest_dir, project_dir=project_dir, log_dir=log_dir)

        assert log_dir.is_dir()

    def test_load_false_does_not_call_load_agent(self, tmp_path):
        dest_dir = tmp_path / "LaunchAgents"
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        with patch("assistant.launchd.load_agent") as mock_load:
            install_all(dest_dir=dest_dir, project_dir=project_dir, log_dir=log_dir, load=False)

        mock_load.assert_not_called()

    def test_default_dest_dir_is_launchagents_dir(self, tmp_path):
        project_dir = tmp_path / "project"
        log_dir = tmp_path / "logs"

        with patch("assistant.launchd.load_agent"), patch("assistant.launchd.write_plist") as mock_write:
            mock_write.side_effect = lambda plist, dest: Path(dest)
            install_all(project_dir=project_dir, log_dir=log_dir)

        expected_dir = default_plist_path().parent
        for call in mock_write.call_args_list:
            _, dest = call.args
            assert Path(dest).parent == expected_dir


class TestLoadUnloadAgent:
    def test_load_agent_calls_launchctl_load(self, tmp_path):
        plist_path = tmp_path / f"{LABEL}.plist"

        with patch("assistant.launchd.subprocess.run") as mock_run:
            load_agent(plist_path)

        mock_run.assert_called_once_with(["launchctl", "load", "-w", str(plist_path)], check=True)

    def test_unload_agent_calls_launchctl_unload(self, tmp_path):
        plist_path = tmp_path / f"{LABEL}.plist"

        with patch("assistant.launchd.subprocess.run") as mock_run:
            unload_agent(plist_path)

        mock_run.assert_called_once_with(["launchctl", "unload", "-w", str(plist_path)], check=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
