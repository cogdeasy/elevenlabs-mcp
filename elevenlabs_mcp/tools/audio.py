"""Sound effects, audio isolation, and audio playback tools."""

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


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=f"""Convert text description of a sound effect to sound effect with a given duration. {get_output_mode_description(server.output_mode)}.
    
    Duration must be between 0.5 and 5 seconds.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        text: Text description of the sound effect
        duration_seconds: Duration of the sound effect in seconds
        output_directory: Directory where files should be saved (only used when saving files).
            Defaults to $HOME/Desktop if not provided.
        loop: Whether to loop the sound effect. Defaults to False.
        output_format (str, optional): Output format of the generated audio. Formatted as codec_sample_rate_bitrate. So an mp3 with 22.05kHz sample rate at 32kbs is represented as mp3_22050_32. MP3 with 192kbps bitrate requires you to be subscribed to Creator tier or above. PCM with 44.1kHz sample rate requires you to be subscribed to Pro tier or above. Note that the μ-law format (sometimes written mu-law, often approximated as u-law) is commonly used for Twilio audio inputs.
            Defaults to "mp3_44100_128". Must be one of:
            mp3_22050_32
            mp3_44100_32
            mp3_44100_64
            mp3_44100_96
            mp3_44100_128
            mp3_44100_192
            pcm_8000
            pcm_16000
            pcm_22050
            pcm_24000
            pcm_44100
            ulaw_8000
            alaw_8000
            opus_48000_32
            opus_48000_64
            opus_48000_96
            opus_48000_128
            opus_48000_192
    """,
)
def text_to_sound_effects(
    text: str,
    duration_seconds: float = 2.0,
    output_directory: str | None = None,
    output_format: str = "mp3_44100_128",
    loop: bool = False,
) -> Union[TextContent, EmbeddedResource]:
    if duration_seconds < 0.5 or duration_seconds > 5:
        make_error("Duration must be between 0.5 and 5 seconds")
    output_path = make_output_path(output_directory, server.base_path)
    output_file_name = make_output_file("sfx", text, "mp3")

    audio_data = server.client.text_to_sound_effects.convert(
        text=text,
        output_format=output_format,
        duration_seconds=duration_seconds,
        loop=loop,
    )
    audio_bytes = b"".join(audio_data)

    # Handle different output modes
    return handle_output_mode(
        audio_bytes, output_path, output_file_name, server.output_mode
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=f"""Isolate audio from a file. {get_output_mode_description(server.output_mode)}.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """,
)
def isolate_audio(
    input_file_path: str, output_directory: str | None = None
) -> Union[TextContent, EmbeddedResource]:
    file_path = handle_input_file(input_file_path)
    output_path = make_output_path(output_directory, server.base_path)
    output_file_name = make_output_file("iso", file_path.name, "mp3")
    with file_path.open("rb") as f:
        audio_bytes = f.read()
    audio_data = server.client.audio_isolation.convert(
        audio=audio_bytes,
    )
    audio_bytes = b"".join(audio_data)

    # Handle different output modes
    return handle_output_mode(
        audio_bytes, output_path, output_file_name, server.output_mode
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True),
    description="Play an audio file. Supports WAV and MP3 formats.",
)
def play_audio(input_file_path: str) -> TextContent:
    file_path = handle_input_file(input_file_path)
    with open(file_path, "rb") as f:
        server.play(f.read(), use_ffmpeg=False)
    return TextContent(type="text", text=f"Successfully played audio file: {file_path}")
