"""Text-to-speech and speech-to-speech (voice changer) tools."""

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
    description=f"""Convert text to speech with a given voice. {get_output_mode_description(server.output_mode)}.
    
    Only one of voice_id or voice_name can be provided. If none are provided, the default voice will be used.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

     Args:
        text (str): The text to convert to speech.
        voice_name (str, optional): The name of the voice to use.
        model_id (str, optional): The model ID to use for speech synthesis. Options include:
            - eleven_v3: High quality model with most expressive voices
            - eleven_multilingual_v2: High quality multilingual model (29 languages)
            - eleven_flash_v2_5: Fastest model with ultra-low latency (32 languages)
            - eleven_turbo_v2_5: Balanced quality and speed (32 languages)
            - eleven_flash_v2: Fast English-only model
            - eleven_turbo_v2: Balanced English-only model
            Defaults to eleven_multilingual_v2 or environment variable ELEVENLABS_MODEL_ID.
        stability (float, optional): Stability of the generated audio. Determines how stable the voice is and the randomness between each generation. Lower values introduce broader emotional range for the voice. Higher values can result in a monotonous voice with limited emotion. Range is 0 to 1.
        similarity_boost (float, optional): Similarity boost of the generated audio. Determines how closely the AI should adhere to the original voice when attempting to replicate it. Range is 0 to 1.
        style (float, optional): Style of the generated audio. Determines the style exaggeration of the voice. This setting attempts to amplify the style of the original speaker. It does consume additional computational resources and might increase latency if set to anything other than 0. Range is 0 to 1.
        use_speaker_boost (bool, optional): Use speaker boost of the generated audio. This setting boosts the similarity to the original speaker. Using this setting requires a slightly higher computational load, which in turn increases latency.
        speed (float, optional): Speed of the generated audio. Controls the speed of the generated speech. Values range from 0.7 to 1.2, with 1.0 being the default speed. Lower values create slower, more deliberate speech while higher values produce faster-paced speech. Extreme values can impact the quality of the generated speech. Range is 0.7 to 1.2.
        output_directory (str, optional): Directory where files should be saved (only used when saving files).
            Defaults to $HOME/Desktop if not provided.
        language: ISO 639-1 language code for the voice.
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

    Returns:
        Text content with file path or MCP resource with audio data, depending on output mode.
    """,
)
def text_to_speech(
    text: str,
    voice_name: str | None = None,
    output_directory: str | None = None,
    voice_id: str | None = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0,
    use_speaker_boost: bool = True,
    speed: float = 1.0,
    language: str = "en",
    output_format: str = "mp3_44100_128",
    model_id: str | None = None,
) -> Union[TextContent, EmbeddedResource]:
    if text == "":
        make_error("Text is required.")

    if voice_id is not None and voice_name is not None:
        make_error("voice_id and voice_name cannot both be provided.")

    voice = None
    if voice_id is not None:
        voice = server.client.voices.get(voice_id=voice_id)
    elif voice_name is not None:
        voices = server.client.voices.search(search=voice_name)
        if len(voices.voices) == 0:
            make_error("No voices found with that name.")
        voice = next((v for v in voices.voices if v.name == voice_name), None)
        if voice is None:
            make_error(f"Voice with name: {voice_name} does not exist.")

    voice_id = voice.voice_id if voice else server.DEFAULT_VOICE_ID

    output_path = make_output_path(output_directory, server.base_path)
    output_file_name = make_output_file("tts", text, "mp3")

    if model_id is None:
        model_id = (
            "eleven_flash_v2_5"
            if language in ["hu", "no", "vi"]
            else "eleven_multilingual_v2"
        )

    audio_data = server.client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        voice_settings={
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
            "speed": speed,
        },
    )
    audio_bytes = b"".join(audio_data)

    # Handle different output modes
    success_message = f"Success. File saved as: {{file_path}}. Voice used: {voice.name if voice else server.DEFAULT_VOICE_ID}"
    return handle_output_mode(
        audio_bytes, output_path, output_file_name, server.output_mode, success_message
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=f"""Transform audio from one voice to another using provided audio files. {get_output_mode_description(server.output_mode)}.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """,
)
def speech_to_speech(
    input_file_path: str,
    voice_name: str = "Adam",
    output_directory: str | None = None,
) -> Union[TextContent, EmbeddedResource]:
    voices = server.client.voices.search(search=voice_name)

    if len(voices.voices) == 0:
        make_error("No voice found with that name.")

    voice = next((v for v in voices.voices if v.name == voice_name), None)

    if voice is None:
        make_error(f"Voice with name: {voice_name} does not exist.")

    assert voice is not None  # Type assertion for type checker
    file_path = handle_input_file(input_file_path)
    output_path = make_output_path(output_directory, server.base_path)
    output_file_name = make_output_file("sts", file_path.name, "mp3")

    with file_path.open("rb") as f:
        audio_bytes = f.read()

    audio_data = server.client.speech_to_speech.convert(
        model_id="eleven_multilingual_sts_v2",
        voice_id=voice.voice_id,
        audio=audio_bytes,
    )

    audio_bytes = b"".join(audio_data)

    # Handle different output modes
    return handle_output_mode(
        audio_bytes, output_path, output_file_name, server.output_mode
    )
