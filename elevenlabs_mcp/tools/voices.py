"""Voice library, voice design, voice cloning, and model listing tools."""

import base64
from typing import IO, Literal

from mcp.types import EmbeddedResource, TextContent, ToolAnnotations

from elevenlabs_mcp import server
from elevenlabs_mcp.model import McpLanguage, McpModel, McpVoice
from elevenlabs_mcp.utils import (
    get_output_mode_description,
    handle_input_file,
    handle_multiple_files_output_mode,
    handle_output_mode,
    make_error,
    make_output_file,
    make_output_path,
)


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="""
    Search for existing voices, a voice that has already been added to the user's ElevenLabs voice library.
    Searches in name, description, labels and category.

    Args:
        search: Search term to filter voices by. Searches in name, description, labels and category.
        sort: Which field to sort by. `created_at_unix` might not be available for older voices.
        sort_direction: Sort order, either ascending or descending.

    Returns:
        List of voices that match the search criteria.
    """,
)
def search_voices(
    search: str | None = None,
    sort: Literal["created_at_unix", "name"] = "name",
    sort_direction: Literal["asc", "desc"] = "desc",
) -> list[McpVoice]:
    response = server.client.voices.search(
        search=search, sort=sort, sort_direction=sort_direction
    )
    return [
        McpVoice(id=voice.voice_id, name=voice.name, category=voice.category)
        for voice in response.voices
    ]


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="List all available models",
)
def list_models() -> list[McpModel]:
    response = server.client.models.list()
    return [
        McpModel(
            id=model.model_id,
            name=model.name,
            languages=[
                McpLanguage(language_id=lang.language_id, name=lang.name)
                for lang in model.languages
            ],
        )
        for model in response
    ]


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="Get details of a specific voice",
)
def get_voice(voice_id: str) -> McpVoice:
    """Get details of a specific voice."""
    response = server.client.voices.get(voice_id=voice_id)
    return McpVoice(
        id=response.voice_id,
        name=response.name,
        category=response.category,
        fine_tuning_status=response.fine_tuning.state,
    )


@server.mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, openWorldHint=True),
    description="""Create an instant voice clone of a voice using provided audio files.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """,
)
def voice_clone(
    name: str, files: list[str], description: str | None = None
) -> TextContent:
    input_paths = [handle_input_file(file) for file in files]
    file_handles: list[IO[bytes]] = []
    try:
        for path in input_paths:
            file_handles.append(path.open("rb"))
        voice = server.client.voices.ivc.create(
            name=name, description=description, files=file_handles
        )
    finally:
        for f in file_handles:
            f.close()

    return TextContent(
        type="text",
        text=f"""Voice cloned successfully: Name: {voice.name}
        ID: {voice.voice_id}
        Category: {voice.category}
        Description: {voice.description or "N/A"}""",
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=f"""Create voice previews from a text prompt. Creates three previews with slight variations. {get_output_mode_description(server.output_mode)}.

    If no text is provided, the tool will auto-generate text.

    Voice preview files are saved as: voice_design_(generated_voice_id)_(timestamp).mp3

    Example file name: voice_design_Ya2J5uIa5Pq14DNPsbC1_20250403_164949.mp3

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """,
)
def text_to_voice(
    voice_description: str,
    text: str | None = None,
    output_directory: str | None = None,
) -> list[EmbeddedResource] | TextContent:
    if voice_description == "":
        make_error("Voice description is required.")

    previews = server.client.text_to_voice.create_previews(
        voice_description=voice_description,
        text=text,
        auto_generate_text=True if text is None else False,
    )

    output_path = make_output_path(output_directory, server.base_path)

    generated_voice_ids = []
    results = []

    for preview in previews.previews:
        output_file_name = make_output_file(
            "voice_design", preview.generated_voice_id, "mp3", full_id=True
        )
        generated_voice_ids.append(preview.generated_voice_id)
        audio_bytes = base64.b64decode(preview.audio_base_64)

        # Handle different output modes
        result = handle_output_mode(
            audio_bytes, output_path, output_file_name, server.output_mode
        )
        results.append(result)

    # Use centralized multiple files output handling
    additional_info = f"Generated voice IDs are: {', '.join(generated_voice_ids)}"
    return handle_multiple_files_output_mode(
        results, server.output_mode, additional_info
    )


@server.mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, openWorldHint=True),
    description="""Add a generated voice to the voice library. Uses the voice ID from the `text_to_voice` tool.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """,
)
def create_voice_from_preview(
    generated_voice_id: str,
    voice_name: str,
    voice_description: str,
) -> TextContent:
    voice = server.client.text_to_voice.create(
        voice_name=voice_name,
        voice_description=voice_description,
        generated_voice_id=generated_voice_id,
    )

    return TextContent(
        type="text",
        text=f"Success. Voice created: {voice.name} with ID:{voice.voice_id}",
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="""Search for a voice across the entire ElevenLabs voice library.

    Args:
        page: Page number to return (0-indexed)
        page_size: Number of voices to return per page (1-100)
        search: Search term to filter voices by

    Returns:
        TextContent containing information about the shared voices
    """,
)
def search_voice_library(
    page: int = 0,
    page_size: int = 10,
    search: str | None = None,
) -> TextContent:
    response = server.client.voices.get_shared(
        page=page,
        page_size=page_size,
        search=search,
    )

    if not response.voices:
        return TextContent(
            type="text", text="No shared voices found with the specified criteria."
        )

    voice_list = []
    for voice in response.voices:
        language_info = "N/A"
        if hasattr(voice, "verified_languages") and voice.verified_languages:
            languages = []
            for lang in voice.verified_languages:
                accent_info = (
                    f" ({lang.accent})"
                    if hasattr(lang, "accent") and lang.accent
                    else ""
                )
                languages.append(f"{lang.language}{accent_info}")
            language_info = ", ".join(languages)

        details = [
            f"Name: {voice.name}",
            f"ID: {voice.voice_id}",
            f"Category: {getattr(voice, 'category', 'N/A')}",
        ]
        # TODO: Make cleaner
        if hasattr(voice, "gender") and voice.gender:
            details.append(f"Gender: {voice.gender}")
        if hasattr(voice, "age") and voice.age:
            details.append(f"Age: {voice.age}")
        if hasattr(voice, "accent") and voice.accent:
            details.append(f"Accent: {voice.accent}")
        if hasattr(voice, "description") and voice.description:
            details.append(f"Description: {voice.description}")
        if hasattr(voice, "use_case") and voice.use_case:
            details.append(f"Use Case: {voice.use_case}")

        details.append(f"Languages: {language_info}")

        if hasattr(voice, "preview_url") and voice.preview_url:
            details.append(f"Preview URL: {voice.preview_url}")

        voice_info = "\n".join(details)
        voice_list.append(voice_info)

    formatted_info = "\n\n".join(voice_list)
    return TextContent(type="text", text=f"Shared Voices:\n\n{formatted_info}")
