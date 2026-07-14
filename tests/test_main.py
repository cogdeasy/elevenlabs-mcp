"""Tests for the elevenlabs_mcp.__main__ CLI entry point."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from elevenlabs_mcp.__main__ import (
    generate_config,
    get_claude_config_path,
    get_python_path,
)


class TestGetClaudeConfigPath:
    def test_windows_path(self, monkeypatch, temp_dir):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: temp_dir))
        expected = temp_dir / "AppData" / "Roaming" / "Claude"
        expected.mkdir(parents=True)
        assert get_claude_config_path() == expected

    def test_darwin_path(self, monkeypatch, temp_dir):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: temp_dir))
        expected = temp_dir / "Library" / "Application Support" / "Claude"
        expected.mkdir(parents=True)
        assert get_claude_config_path() == expected

    def test_linux_path_uses_xdg_config_home(self, monkeypatch, temp_dir):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(temp_dir))
        expected = temp_dir / "Claude"
        expected.mkdir()
        assert get_claude_config_path() == expected

    def test_unknown_platform_returns_none(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "sunos5")
        assert get_claude_config_path() is None

    def test_missing_directory_returns_none(self, monkeypatch, temp_dir):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(temp_dir / "nonexistent"))
        assert get_claude_config_path() is None


class TestGetPythonPath:
    def test_returns_current_interpreter(self):
        assert get_python_path() == sys.executable


class TestGenerateConfig:
    def test_explicit_api_key(self):
        config = generate_config("explicit-key")
        server = config["mcpServers"]["ElevenLabs"]
        assert server["env"]["ELEVENLABS_API_KEY"] == "explicit-key"
        assert server["command"] == sys.executable
        assert server["args"][0].endswith("server.py")

    def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
        config = generate_config()
        assert (
            config["mcpServers"]["ElevenLabs"]["env"]["ELEVENLABS_API_KEY"] == "env-key"
        )

    def test_missing_api_key_exits(self, monkeypatch, capsys):
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            generate_config()
        assert exc_info.value.code == 1
        assert "API key is required" in capsys.readouterr().out


class TestCliInvocation:
    """E2E: run `python -m elevenlabs_mcp` as a subprocess."""

    def test_print_flag_outputs_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "elevenlabs_mcp", "--api-key", "cli-key", "--print"],
            capture_output=True,
            text=True,
            check=True,
        )
        config = json.loads(result.stdout)
        assert (
            config["mcpServers"]["ElevenLabs"]["env"]["ELEVENLABS_API_KEY"] == "cli-key"
        )

    def test_writes_config_file_to_custom_path(self, temp_dir):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "elevenlabs_mcp",
                "--api-key",
                "cli-key",
                "--config-path",
                str(temp_dir),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        config_file = temp_dir / "claude_desktop_config.json"
        config = json.loads(config_file.read_text())
        assert (
            config["mcpServers"]["ElevenLabs"]["env"]["ELEVENLABS_API_KEY"] == "cli-key"
        )
