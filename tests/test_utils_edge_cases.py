"""Edge-path tests for elevenlabs_mcp.utils not covered by test_utils.py."""

import os
from pathlib import Path

import pytest
from mcp.types import BlobResourceContents, EmbeddedResource, TextContent

from elevenlabs_mcp.utils import (
    ElevenLabsMcpError,
    create_resource_response,
    generate_resource_uri,
    get_mime_type,
    get_output_mode_description,
    handle_input_file,
    handle_large_text,
    handle_multiple_files_output_mode,
    handle_output_mode,
    make_output_path,
    parse_conversation_transcript,
    try_find_similar_files,
)


class TestMakeOutputPath:
    def test_unwriteable_directory_raises(self, temp_dir):
        target = temp_dir / "readonly"
        target.mkdir()
        target.chmod(0o500)
        try:
            if os.access(target / "sub", os.W_OK):
                pytest.skip("running as privileged user; cannot make unwriteable dir")
            with pytest.raises(ElevenLabsMcpError, match="not writeable"):
                make_output_path(str(target / "sub"))
        finally:
            target.chmod(0o700)


class TestHandleInputFile:
    def test_relative_path_without_base_path_raises(self, monkeypatch):
        monkeypatch.delenv("ELEVENLABS_MCP_BASE_PATH", raising=False)
        with pytest.raises(ElevenLabsMcpError, match="absolute path"):
            handle_input_file("relative/audio.mp3")

    def test_missing_file_suggests_similar_files(self, temp_dir):
        (temp_dir / "recording.mp3").touch()
        with pytest.raises(ElevenLabsMcpError, match="Did you mean"):
            handle_input_file(str(temp_dir / "recordings.mp3"))

    def test_missing_file_without_similar_files(self, temp_dir):
        with pytest.raises(ElevenLabsMcpError, match="does not exist"):
            handle_input_file(str(temp_dir / "nothing_alike.mp3"))

    def test_directory_is_not_a_file(self, temp_dir):
        directory = temp_dir / "audio.mp3"
        directory.mkdir()
        with pytest.raises(ElevenLabsMcpError, match="is not a file"):
            handle_input_file(str(directory))

    def test_non_audio_file_rejected(self, temp_dir):
        text_file = temp_dir / "notes.txt"
        text_file.touch()
        with pytest.raises(ElevenLabsMcpError, match="not an audio or video file"):
            handle_input_file(str(text_file))

    def test_non_audio_file_allowed_when_check_disabled(self, temp_dir):
        text_file = temp_dir / "notes.txt"
        text_file.touch()
        assert handle_input_file(str(text_file), audio_content_check=False) == text_file


class TestTryFindSimilarFiles:
    def test_returns_empty_when_no_similar_files(self, temp_dir):
        (temp_dir / "unrelated.bin").touch()
        assert try_find_similar_files("zzzzzz.mp3", temp_dir) == []

    def test_filters_non_audio_matches(self, temp_dir):
        (temp_dir / "speech.mp3").touch()
        (temp_dir / "speech.doc").touch()
        result = try_find_similar_files("speech1.mp3", temp_dir)
        assert result == [temp_dir / "speech.mp3"]


class TestHandleLargeText:
    def test_short_text_returned_verbatim(self):
        assert handle_large_text("short", max_length=100) == "short"

    def test_long_text_saved_to_temp_file(self):
        text = "x" * 50
        result = handle_large_text(text, max_length=10, content_type="transcript")
        assert "Transcript saved to temporary file:" in result
        temp_path = result.split(": ")[1].split("\n")[0]
        assert Path(temp_path).read_text(encoding="utf-8") == text
        os.unlink(temp_path)


class TestParseConversationTranscript:
    def test_entries_with_timestamps(self):
        class Entry:
            role = "agent"
            message = "Hello"
            timestamp = "00:01"

        transcript, is_temp = parse_conversation_transcript([Entry()])
        assert transcript == "[00:01] agent: Hello"
        assert is_temp is False

    def test_long_transcript_saved_to_temp_file(self):
        class Entry:
            role = "user"
            message = "y" * 100
            timestamp = None

        transcript, is_temp = parse_conversation_transcript([Entry()], max_length=50)
        assert is_temp is True
        assert "Transcript saved to temporary file:" in transcript
        temp_path = transcript.split(": ")[1].split("\n")[0]
        assert "user: " in Path(temp_path).read_text(encoding="utf-8")
        os.unlink(temp_path)


class TestGetMimeType:
    @pytest.mark.parametrize(
        "extension,expected",
        [
            ("mp3", "audio/mpeg"),
            (".mp3", "audio/mpeg"),
            ("WAV", "audio/wav"),
            ("txt", "text/plain"),
            ("json", "application/json"),
            ("mp4", "video/mp4"),
            ("unknown", "application/octet-stream"),
        ],
    )
    def test_mime_types(self, extension, expected):
        assert get_mime_type(extension) == expected


class TestGenerateResourceUri:
    def test_uri_format(self):
        assert generate_resource_uri("file.mp3") == "elevenlabs://file.mp3"


class TestCreateResourceResponse:
    def test_text_file_returns_text_resource(self):
        resource = create_resource_response(b"hello", "notes.txt", "txt")
        assert isinstance(resource, EmbeddedResource)
        assert resource.resource.text == "hello"
        assert str(resource.resource.uri) == "elevenlabs://notes.txt"

    def test_directory_embedded_in_uri(self, temp_dir):
        resource = create_resource_response(b"data", "a.mp3", "mp3", directory=temp_dir)
        assert str(resource.resource.uri).endswith("a.mp3")
        assert (temp_dir / "a.mp3").as_posix() in str(resource.resource.uri)

    def test_undecodable_text_falls_back_to_blob(self):
        resource = create_resource_response(b"\xff\xfe\xfd", "notes.txt", "txt")
        assert isinstance(resource.resource, BlobResourceContents)


class TestHandleOutputMode:
    def test_resources_mode_returns_embedded_resource(self, temp_dir):
        result = handle_output_mode(b"audio", temp_dir, "out.mp3", "resources")
        assert isinstance(result, EmbeddedResource)
        assert not (temp_dir / "out.mp3").exists()

    def test_both_mode_saves_and_returns_resource(self, temp_dir):
        result = handle_output_mode(b"audio", temp_dir, "out.mp3", "both")
        assert isinstance(result, EmbeddedResource)
        assert (temp_dir / "out.mp3").read_bytes() == b"audio"

    def test_invalid_mode_raises(self, temp_dir):
        with pytest.raises(ValueError, match="Invalid output mode"):
            handle_output_mode(b"audio", temp_dir, "out.mp3", "bogus")


class TestHandleMultipleFilesOutputMode:
    def test_resources_mode_returns_embedded_resources(self):
        resource = create_resource_response(b"a", "a.mp3", "mp3")
        result = handle_multiple_files_output_mode([resource], "resources")
        assert result == [resource]

    def test_resources_mode_without_resources(self):
        text = TextContent(type="text", text="Success. File saved as: /tmp/a.mp3.")
        result = handle_multiple_files_output_mode([text], "both")
        assert isinstance(result, TextContent)
        assert result.text == "No files generated"

    def test_files_mode_with_additional_info(self):
        text = TextContent(type="text", text="Success. File saved as: /tmp/a.mp3. Done")
        result = handle_multiple_files_output_mode([text], "files", "IDs: v1")
        assert isinstance(result, TextContent)
        assert "IDs: v1" in result.text

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid output mode"):
            handle_multiple_files_output_mode([], "bogus")


class TestGetOutputModeDescription:
    @pytest.mark.parametrize(
        "mode,fragment",
        [
            ("files", "Saves output file"),
            ("resources", "base64-encoded MCP resource"),
            ("both", "AND returns"),
            ("bogus", "depends on ELEVENLABS_MCP_OUTPUT_MODE"),
        ],
    )
    def test_descriptions(self, mode, fragment):
        assert fragment in get_output_mode_description(mode)
