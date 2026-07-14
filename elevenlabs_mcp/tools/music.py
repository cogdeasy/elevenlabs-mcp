"""Music generation tools: composition, video-to-music, and inpainting."""

from typing import IO, Literal, Union, cast

from mcp.types import EmbeddedResource, TextContent, ToolAnnotations

from elevenlabs.types import CompositionPlan, MusicPrompt

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
    description=f"""Convert a prompt to music and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop. {get_output_mode_description(server.output_mode)}.

    Two models are supported:
    - music_v2 (default): latest model. Composition plans use a `chunks` array where each chunk is either a `GenerationChunk` (text, duration_ms, positive_styles, negative_styles, context_adherence, optional conditioning_ref + condition_strength) or an `AudioRefChunk` ({{song_id, range: {{start_ms, end_ms}}}}) for inpainting. Inpainting also requires the source song to have been stored — call this tool with store_for_inpainting=True or use upload_music_for_inpainting first to get a song_id.
    - music_v1: legacy model. Composition plans use positive_global_styles, negative_global_styles, sections.

    Args:
        prompt: Prompt to convert to music. Must provide either prompt or composition_plan.
        output_directory: Directory to save the output audio file
        composition_plan: Composition plan dict. Shape depends on model_id (see above). Must provide either prompt or composition_plan.
        music_length_ms: Length of the generated music in milliseconds (3000-600000). Cannot be used if composition_plan is provided.
        model_id: Which music model to use. One of "music_v1" or "music_v2". Defaults to "music_v2".
        force_instrumental: If True, the model will avoid generating lyrics/vocals.
        store_for_inpainting: If True, the generated song is stored server-side and the returned song_id can be used in later inpainting calls (as an AudioRefChunk.song_id, or conditioning_ref).
        seed: Optional integer seed for reproducible generation (music_v2 only).

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.""",
)
def compose_music(
    prompt: str | None = None,
    output_directory: str | None = None,
    composition_plan: dict | None = None,
    music_length_ms: int | None = None,
    model_id: Literal["music_v1", "music_v2"] = "music_v2",
    force_instrumental: bool = False,
    store_for_inpainting: bool = False,
    seed: int | None = None,
) -> Union[TextContent, EmbeddedResource]:
    if prompt is None and composition_plan is None:
        make_error(
            f"Either prompt or composition_plan must be provided. Prompt: {prompt}"
        )

    if prompt is not None and composition_plan is not None:
        make_error("Only one of prompt or composition_plan must be provided")

    if music_length_ms is not None and composition_plan is not None:
        make_error("music_length_ms cannot be used if composition_plan is provided")

    output_path = make_output_path(output_directory, server.base_path)
    output_file_name = make_output_file("music", "", "mp3")

    plan: CompositionPlan | MusicPrompt | None = None
    if composition_plan is not None:
        if "chunks" in composition_plan:
            plan = CompositionPlan.model_validate(composition_plan)
        else:
            plan = MusicPrompt.model_validate(composition_plan)

    song_id: str | None = None
    if store_for_inpainting:
        detailed = server.client.music.compose_detailed(
            prompt=prompt,
            music_length_ms=music_length_ms,
            composition_plan=plan,
            model_id=model_id,  # type: ignore[arg-type]
            force_instrumental=force_instrumental,
            store_for_inpainting=True,
        )
        audio_bytes = detailed.audio
        song_id = detailed.song_id
    else:
        audio_data = server.client.music.compose(
            prompt=prompt,
            music_length_ms=music_length_ms,
            composition_plan=plan,
            model_id=model_id,
            force_instrumental=force_instrumental,
            seed=seed,
        )
        audio_bytes = b"".join(audio_data)

    success_message = f"Success. File saved as: {{file_path}}. Model: {model_id}." + (
        f" song_id (for inpainting): {song_id}" if song_id else ""
    )
    return handle_output_mode(
        audio_bytes, output_path, output_file_name, server.output_mode, success_message
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="""Create a composition plan for music generation. Usage of this endpoint does not cost any credits but is subject to rate limiting depending on your tier. Composition plans can be used when generating music with the compose_music tool.

    The returned plan shape depends on model_id:
    - music_v2 (default): `{"chunks": [GenerationChunk | AudioRefChunk, ...]}`. Each GenerationChunk has `text`, `duration_ms`, `positive_styles`, `negative_styles`, `context_adherence` and optional `conditioning_ref` + `condition_strength`. AudioRefChunks reference a stored song via `song_id` and `range: {start_ms, end_ms}` for inpainting.
    - music_v1: `{"positive_global_styles": [...], "negative_global_styles": [...], "sections": [...]}`.

    Args:
        prompt: Prompt to create a composition plan for
        music_length_ms: The length of the composition plan to generate in milliseconds. Must be between 10000ms and 300000ms. Optional - if not provided, the model will choose a length based on the prompt.
        source_composition_plan: An optional composition plan dict to use as a source for the new composition plan. Should match the shape of the model_id you request.
        model_id: Which music model to plan for. One of "music_v1" or "music_v2". Defaults to "music_v2".
    """,
)
def create_composition_plan(
    prompt: str,
    music_length_ms: int | None = None,
    source_composition_plan: dict | None = None,
    model_id: Literal["music_v1", "music_v2"] = "music_v2",
) -> dict:
    source: CompositionPlan | MusicPrompt | None = None
    if source_composition_plan is not None:
        if "chunks" in source_composition_plan:
            source = CompositionPlan.model_validate(source_composition_plan)
        else:
            source = MusicPrompt.model_validate(source_composition_plan)

    composition_plan = server.client.music.composition_plan.create(
        prompt=prompt,
        music_length_ms=music_length_ms,
        source_composition_plan=source,
        model_id=model_id,
    )

    return composition_plan.model_dump(exclude_none=True)


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description=f"""Generate background music for one or more video files. {get_output_mode_description(server.output_mode)}.

    The videos are concatenated server-side in the order provided; the generated score targets the combined duration. Constraints: 1-10 videos per call, combined size <= 200 MB, combined duration <= 600 seconds.

    Args:
        input_file_paths: Paths to the video files. Order is preserved.
        description: Optional natural-language direction for the music (e.g. "Build suspense, then resolve with a warm cinematic finish.").
        tags: Optional list of up to 10 short style cues (e.g. ["cinematic", "suspenseful", "uplifting"]).
        model_id: Which music model to use. One of "music_v1" or "music_v2". Defaults to "music_v2".
        output_directory: Directory to save the generated audio file. Defaults to $HOME/Desktop.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """,
)
def video_to_music(
    input_file_paths: list[str],
    description: str | None = None,
    tags: list[str] | None = None,
    model_id: Literal["music_v1", "music_v2"] = "music_v2",
    output_directory: str | None = None,
) -> Union[TextContent, EmbeddedResource]:
    if not 1 <= len(input_file_paths) <= 10:
        make_error("Provide between 1 and 10 video files.")

    video_paths = [
        handle_input_file(p, audio_content_check=False) for p in input_file_paths
    ]

    output_path = make_output_path(output_directory, server.base_path)
    output_file_name = make_output_file("v2m", video_paths[0].name, "mp3")

    file_handles: list[IO[bytes]] = []
    try:
        for p in video_paths:
            file_handles.append(p.open("rb"))
        audio_data = server.client.music.video_to_music(
            videos=cast(list, file_handles),
            description=description,
            tags=tags,
            model_id=model_id,
        )
        audio_bytes = b"".join(audio_data)
    finally:
        for f in file_handles:
            f.close()

    success_message = f"Success. File saved as: {{file_path}}. Model: {model_id}."
    return handle_output_mode(
        audio_bytes, output_path, output_file_name, server.output_mode, success_message
    )


@server.mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, openWorldHint=True),
    description="""Upload an existing audio file to ElevenLabs so it can be referenced in music_v2 inpainting workflows. Returns a song_id you can plug into a composition plan's AudioRefChunks (or a generation chunk's conditioning_ref) to edit or extend the track via the compose_music tool.

    Optionally extracts a composition plan from the uploaded audio so you have a starting point to mutate.

    Note: this endpoint is gated to enterprise customers with inpainting access.

    Args:
        input_file_path: Path to a local audio file to upload.
        extract_composition_plan: Which model to extract a composition plan for ("music_v1" or "music_v2"). Pass None to skip extraction. Defaults to "music_v2".

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """,
)
def upload_music_for_inpainting(
    input_file_path: str,
    extract_composition_plan: Literal["music_v1", "music_v2"] | None = "music_v2",
) -> TextContent:
    file_path = handle_input_file(input_file_path)
    with file_path.open("rb") as f:
        result = server.client.music.upload(
            file=f,
            extract_composition_plan=extract_composition_plan,
        )

    plan_section = ""
    if result.composition_plan is not None:
        plan_json = result.composition_plan.model_dump_json(indent=2, exclude_none=True)
        plan_section = (
            f"\n\nExtracted composition plan ({extract_composition_plan}):\n{plan_json}"
        )

    return TextContent(
        type="text",
        text=f"song_id: {result.song_id}{plan_section}",
    )
