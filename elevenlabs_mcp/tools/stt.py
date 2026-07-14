"""Speech-to-text (transcription) tools."""

from typing import Union

from mcp.types import EmbeddedResource, TextContent, ToolAnnotations

from elevenlabs_mcp import server
from elevenlabs_mcp.utils import (
    get_output_mode_description,
    handle_input_file,
    handle_output_mode,
    make_error,
    make_output_file,
    make_output_path,
)


def format_diarized_transcript(transcription) -> str:
    """Format transcript with speaker labels from diarized response."""
    try:
        # Try to access words array - the exact attribute might vary
        words = None
        if hasattr(transcription, "words"):
            words = transcription.words
        elif hasattr(transcription, "__dict__"):
            # Try to find words in the response dict
            for key, value in transcription.__dict__.items():
                if key == "words" or (
                    isinstance(value, list)
                    and len(value) > 0
                    and (
                        hasattr(value[0], "speaker_id")
                        if hasattr(value[0], "__dict__")
                        else (
                            "speaker_id" in value[0]
                            if isinstance(value[0], dict)
                            else False
                        )
                    )
                ):
                    words = value
                    break

        if not words:
            return transcription.text

        formatted_lines = []
        current_speaker = None
        current_text = []

        for word in words:
            # Get speaker_id - might be an attribute or dict key
            word_speaker = None
            if hasattr(word, "speaker_id"):
                word_speaker = word.speaker_id
            elif isinstance(word, dict) and "speaker_id" in word:
                word_speaker = word["speaker_id"]

            # Get text - might be an attribute or dict key
            word_text = None
            if hasattr(word, "text"):
                word_text = word.text
            elif isinstance(word, dict) and "text" in word:
                word_text = word["text"]

            if not word_speaker or not word_text:
                continue

            # Skip spacing/punctuation types if they exist
            if hasattr(word, "type") and word.type == "spacing":
                continue
            elif isinstance(word, dict) and word.get("type") == "spacing":
                continue

            if current_speaker != word_speaker:
                # Save previous speaker's text
                if current_speaker and current_text:
                    speaker_label = current_speaker.upper().replace("_", " ")
                    formatted_lines.append(f"{speaker_label}: {' '.join(current_text)}")

                # Start new speaker
                current_speaker = word_speaker
                current_text = [word_text.strip()]
            else:
                current_text.append(word_text.strip())

        # Add final speaker's text
        if current_speaker and current_text:
            speaker_label = current_speaker.upper().replace("_", " ")
            formatted_lines.append(f"{speaker_label}: {' '.join(current_text)}")

        return "\n\n".join(formatted_lines)

    except Exception:
        # Fallback to regular text if something goes wrong
        return transcription.text


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=f"""Transcribe speech from an audio file. When save_transcript_to_file=True: {get_output_mode_description(server.output_mode)}. When return_transcript_to_client_directly=True, always returns text directly regardless of output mode.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        file_path: Path to the audio file to transcribe
        language_code: ISO 639-3 language code for transcription. If not provided, the language will be detected automatically.
        diarize: Whether to diarize the audio file. If True, which speaker is currently speaking will be annotated in the transcription.
        save_transcript_to_file: Whether to save the transcript to a file.
        return_transcript_to_client_directly: Whether to return the transcript to the client directly.
        output_directory: Directory where files should be saved (only used when saving files).
            Defaults to $HOME/Desktop if not provided.

    Returns:
        TextContent containing the transcription or MCP resource with transcript data.
    """,
)
def speech_to_text(
    input_file_path: str,
    language_code: str | None = None,
    diarize: bool = False,
    save_transcript_to_file: bool = True,
    return_transcript_to_client_directly: bool = False,
    output_directory: str | None = None,
) -> Union[TextContent, EmbeddedResource]:
    if not save_transcript_to_file and not return_transcript_to_client_directly:
        make_error("Must save transcript to file or return it to the client directly.")
    file_path = handle_input_file(input_file_path)
    if save_transcript_to_file:
        output_path = make_output_path(output_directory, server.base_path)
        output_file_name = make_output_file("stt", file_path.name, "txt")
    with file_path.open("rb") as f:
        audio_bytes = f.read()

    if language_code == "" or language_code is None:
        language_code = None

    transcription = server.client.speech_to_text.convert(
        model_id="scribe_v2",
        file=audio_bytes,
        language_code=language_code,
        enable_logging=True,
        diarize=diarize,
        tag_audio_events=True,
    )

    # Format transcript with speaker identification if diarization was enabled
    if diarize:
        formatted_transcript = format_diarized_transcript(transcription)
    else:
        formatted_transcript = transcription.text

    if return_transcript_to_client_directly:
        return TextContent(type="text", text=formatted_transcript)

    if save_transcript_to_file:
        transcript_bytes = formatted_transcript.encode("utf-8")

        # Handle different output modes
        success_message = f"Transcription saved to {file_path}"
        return handle_output_mode(
            transcript_bytes,
            output_path,
            output_file_name,
            server.output_mode,
            success_message,
        )

    # This should not be reached due to validation at the start of the function
    return TextContent(type="text", text="No output mode specified")
