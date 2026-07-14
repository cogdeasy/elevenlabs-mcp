"""Unit tests for the elevenlabs://{filename} resource handler in
elevenlabs_mcp.server (get_elevenlabs_resource).
"""

import base64
from unittest.mock import patch

import pytest

import elevenlabs_mcp.server as server
from elevenlabs_mcp.utils import ElevenLabsMcpError


@pytest.fixture
def base_dir(temp_dir):
    with patch.object(server, "base_path", str(temp_dir)):
        yield temp_dir


class TestGetElevenlabsResource:
    def test_text_file_returns_text_resource(self, base_dir):
        (base_dir / "transcript.txt").write_text("hello transcript")
        resource = server.get_elevenlabs_resource("transcript.txt")
        assert str(resource.uri) == "elevenlabs://transcript.txt"
        assert resource.mimeType == "text/plain"
        assert resource.text == "hello transcript"

    def test_binary_file_returns_base64_data(self, base_dir):
        audio_bytes = b"\x00\x01binary-audio"
        (base_dir / "speech.mp3").write_bytes(audio_bytes)
        resource = server.get_elevenlabs_resource("speech.mp3")
        assert str(resource.uri) == "elevenlabs://speech.mp3"
        assert resource.mimeType == "audio/mpeg"
        assert resource.data == base64.b64encode(audio_bytes).decode("utf-8")

    def test_missing_file_raises_file_not_found(self, base_dir):
        with pytest.raises(FileNotFoundError, match="Resource file not found"):
            server.get_elevenlabs_resource("missing.txt")

    def test_path_traversal_rejected(self, base_dir):
        with pytest.raises(ElevenLabsMcpError, match="outside of allowed directory"):
            server.get_elevenlabs_resource("../outside.txt")

    def test_undecodable_text_file_raises(self, base_dir):
        (base_dir / "broken.txt").write_bytes(b"\xff\xfe\xfa invalid utf8 \xff")
        with pytest.raises(ElevenLabsMcpError, match="Failed to decode"):
            server.get_elevenlabs_resource("broken.txt")

    def test_unreadable_file_raises_file_not_found(self, base_dir):
        (base_dir / "locked.txt").write_text("secret")
        with patch("builtins.open", side_effect=IOError("permission denied")):
            with pytest.raises(FileNotFoundError, match="Failed to read resource"):
                server.get_elevenlabs_resource("locked.txt")

    def test_unknown_extension_defaults_to_octet_stream(self, base_dir):
        (base_dir / "data.xyz").write_bytes(b"blob")
        resource = server.get_elevenlabs_resource("data.xyz")
        assert resource.mimeType == "application/octet-stream"
        assert resource.data == base64.b64encode(b"blob").decode("utf-8")


class TestIsBrokenPipeError:
    def test_broken_pipe(self):
        assert server._is_broken_pipe_error(BrokenPipeError())

    def test_other_exception(self):
        assert not server._is_broken_pipe_error(ValueError("nope"))

    def test_group_of_broken_pipes(self):
        group = BaseExceptionGroup("g", [BrokenPipeError(), BrokenPipeError()])
        assert server._is_broken_pipe_error(group)

    def test_mixed_group(self):
        group = BaseExceptionGroup("g", [BrokenPipeError(), ValueError("nope")])
        assert not server._is_broken_pipe_error(group)
