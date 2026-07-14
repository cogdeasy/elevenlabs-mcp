"""Unit tests for elevenlabs_mcp.tools.stt.format_diarized_transcript and
the diarization path of speech_to_text.

The ElevenLabs client is mocked; no real API key or network access is needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.types import TextContent

import elevenlabs_mcp.server as server
from elevenlabs_mcp.tools.stt import format_diarized_transcript
from elevenlabs_mcp.utils import ElevenLabsMcpError


@pytest.fixture
def mock_client():
    with patch.object(server, "client") as client:
        yield client


def make_word(speaker_id, text, type=None):
    if type is None:
        return SimpleNamespace(speaker_id=speaker_id, text=text)
    return SimpleNamespace(speaker_id=speaker_id, text=text, type=type)


class TestFormatDiarizedTranscript:
    def test_single_speaker_attribute_words(self):
        transcription = SimpleNamespace(
            text="hello world",
            words=[make_word("speaker_1", "hello"), make_word("speaker_1", "world")],
        )
        assert format_diarized_transcript(transcription) == "SPEAKER 1: hello world"

    def test_multiple_speakers_grouped(self):
        transcription = SimpleNamespace(
            text="",
            words=[
                make_word("speaker_1", "hi"),
                make_word("speaker_1", "there"),
                make_word("speaker_2", "hello"),
                make_word("speaker_1", "bye"),
            ],
        )
        assert format_diarized_transcript(transcription) == (
            "SPEAKER 1: hi there\n\nSPEAKER 2: hello\n\nSPEAKER 1: bye"
        )

    def test_dict_words(self):
        class Transcription:
            def __init__(self):
                self.text = "fallback"
                self.words = [
                    {"speaker_id": "speaker_1", "text": "one"},
                    {"speaker_id": "speaker_2", "text": "two"},
                ]

        assert format_diarized_transcript(Transcription()) == (
            "SPEAKER 1: one\n\nSPEAKER 2: two"
        )

    def test_words_found_via_dict_scan(self):
        class Transcription:
            pass

        transcription = Transcription()
        transcription.text = "fallback"
        transcription.segments = [
            {"speaker_id": "speaker_1", "text": "scanned"},
        ]
        # delete the ``words`` attribute path: object has no ``words``
        assert format_diarized_transcript(transcription) == "SPEAKER 1: scanned"

    def test_no_words_falls_back_to_text(self):
        transcription = SimpleNamespace(text="plain text", words=[])
        assert format_diarized_transcript(transcription) == "plain text"

    def test_no_words_attribute_falls_back_to_text(self):
        class Transcription:
            pass

        transcription = Transcription()
        transcription.text = "no words here"
        assert format_diarized_transcript(transcription) == "no words here"

    def test_spacing_words_skipped(self):
        transcription = SimpleNamespace(
            text="",
            words=[
                make_word("speaker_1", "hello"),
                make_word("speaker_1", " ", type="spacing"),
                make_word("speaker_1", "world", type="word"),
            ],
        )
        assert format_diarized_transcript(transcription) == "SPEAKER 1: hello world"

    def test_spacing_dict_words_skipped(self):
        transcription = SimpleNamespace(
            text="",
            words=[
                {"speaker_id": "speaker_1", "text": "hello"},
                {"speaker_id": "speaker_1", "text": " ", "type": "spacing"},
                {"speaker_id": "speaker_1", "text": "world", "type": "word"},
            ],
        )
        assert format_diarized_transcript(transcription) == "SPEAKER 1: hello world"

    def test_words_missing_speaker_or_text_skipped(self):
        transcription = SimpleNamespace(
            text="",
            words=[
                make_word(None, "orphan"),
                make_word("speaker_1", None),
                {"other": "irrelevant"},
                make_word("speaker_1", "kept"),
            ],
        )
        assert format_diarized_transcript(transcription) == "SPEAKER 1: kept"

    def test_exception_falls_back_to_text(self):
        class ExplodingWords:
            def __iter__(self):
                raise RuntimeError("boom")

            def __len__(self):
                return 1

        transcription = SimpleNamespace(text="safe text", words=ExplodingWords())
        assert format_diarized_transcript(transcription) == "safe text"


class TestSpeechToTextDiarization:
    def test_diarize_returns_formatted_transcript_directly(
        self, mock_client, sample_audio_file
    ):
        mock_client.speech_to_text.convert.return_value = SimpleNamespace(
            text="hello world",
            words=[
                make_word("speaker_1", "hello"),
                make_word("speaker_2", "world"),
            ],
        )
        result = server.speech_to_text(
            input_file_path=str(sample_audio_file),
            diarize=True,
            save_transcript_to_file=False,
            return_transcript_to_client_directly=True,
        )
        assert isinstance(result, TextContent)
        assert result.text == "SPEAKER 1: hello\n\nSPEAKER 2: world"
        _, kwargs = mock_client.speech_to_text.convert.call_args
        assert kwargs["diarize"] is True

    def test_no_diarize_returns_plain_text(self, mock_client, sample_audio_file):
        mock_client.speech_to_text.convert.return_value = SimpleNamespace(
            text="plain transcript"
        )
        result = server.speech_to_text(
            input_file_path=str(sample_audio_file),
            diarize=False,
            save_transcript_to_file=False,
            return_transcript_to_client_directly=True,
        )
        assert result.text == "plain transcript"

    def test_diarize_saved_to_file(self, mock_client, sample_audio_file, temp_dir):
        mock_client.speech_to_text.convert.return_value = SimpleNamespace(
            text="hello",
            words=[make_word("speaker_1", "hello")],
        )
        result = server.speech_to_text(
            input_file_path=str(sample_audio_file),
            diarize=True,
            save_transcript_to_file=True,
            output_directory=str(temp_dir),
        )
        assert isinstance(result, TextContent)
        saved = list(temp_dir.glob("stt_*.txt"))
        assert len(saved) == 1
        assert saved[0].read_text() == "SPEAKER 1: hello"

    def test_no_output_requested_raises(self, mock_client, sample_audio_file):
        with pytest.raises(ElevenLabsMcpError):
            server.speech_to_text(
                input_file_path=str(sample_audio_file),
                save_transcript_to_file=False,
                return_transcript_to_client_directly=False,
            )
