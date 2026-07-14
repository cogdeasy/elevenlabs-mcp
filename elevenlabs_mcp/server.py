"""
ElevenLabs MCP Server

⚠️ IMPORTANT: This server provides access to ElevenLabs API endpoints which may incur costs.
Each tool that makes an API call is marked with a cost warning. Please follow these guidelines:

1. Only use tools when explicitly requested by the user
2. For tools that generate audio, consider the length of the text as it affects costs
3. Some operations like voice cloning or text-to-voice may have higher costs

Tools without cost warnings in their description are free to use as they only read existing data.

This module owns shared server state (the ElevenLabs client, the FastMCP
instance, and configuration) and re-exports every tool handler. The handlers
themselves live in domain modules under ``elevenlabs_mcp.tools`` and read
state dynamically via ``server.client`` etc., so patching attributes on this
module (as the test suite does) affects all tools. Values baked into tool
signatures or descriptions at import time (e.g. the ``create_agent``
``voice_id`` default) are the exception, matching pre-refactor behavior.
"""

import base64
import os
import sys

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import Resource

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play  # noqa: F401  (re-exported; used by tools.audio)

from elevenlabs_mcp import __version__
from elevenlabs_mcp.utils import (
    get_mime_type,
    make_error,
    make_output_path,
    parse_location,
    resolve_resource_path,
)

load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
base_path = os.getenv("ELEVENLABS_MCP_BASE_PATH")
output_mode = os.getenv("ELEVENLABS_MCP_OUTPUT_MODE", "files").strip().lower()
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "cgSgspJ2msm6clMCkdW9")

if output_mode not in {"files", "resources", "both"}:
    raise ValueError(
        "ELEVENLABS_MCP_OUTPUT_MODE must be one of: 'files', 'resources', 'both'"
    )
if not api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable is required")

origin = parse_location(os.getenv("ELEVENLABS_API_RESIDENCY"))

# Add custom client to ElevenLabs to set User-Agent header
custom_client = httpx.Client(
    headers={
        "User-Agent": f"ElevenLabs-MCP/{__version__}",
    },
)

client = ElevenLabs(api_key=api_key, httpx_client=custom_client, base_url=origin)
mcp = FastMCP("ElevenLabs")


@mcp.resource("elevenlabs://{filename}")
def get_elevenlabs_resource(filename: str) -> Resource:
    """
    Resource handler for ElevenLabs generated files.
    """
    base_dir = make_output_path(None, base_path)
    file_path = resolve_resource_path(filename, base_dir)

    if not file_path.exists():
        raise FileNotFoundError(f"Resource file not found: {filename}")

    # Read the file and determine MIME type
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
    except IOError as e:
        raise FileNotFoundError(f"Failed to read resource file {filename}: {e}")

    file_extension = file_path.suffix.lstrip(".")
    mime_type = get_mime_type(file_extension)

    # For text files, return text content
    if mime_type.startswith("text/"):
        try:
            text_content = file_data.decode("utf-8")
            return Resource(
                uri=f"elevenlabs://{filename}",
                name=filename,
                mimeType=mime_type,
                text=text_content,
            )
        except UnicodeDecodeError:
            make_error(
                f"Failed to decode text resource {filename} as UTF-8; MIME type {mime_type} may be incorrect or file is corrupt"
            )

    # For binary files, return base64 encoded data
    base64_data = base64.b64encode(file_data).decode("utf-8")
    return Resource(
        uri=f"elevenlabs://{filename}",
        name=filename,
        mimeType=mime_type,
        data=base64_data,
    )


# Tool modules read shared state (client, mcp, config) from this module, so
# they must be imported after it is defined. Importing them registers every
# tool on `mcp`; the handlers are re-exported here for backwards compatibility.
from elevenlabs_mcp.tools.account import check_subscription  # noqa: E402, F401
from elevenlabs_mcp.tools.audio import (  # noqa: E402, F401
    isolate_audio,
    play_audio,
    text_to_sound_effects,
)
from elevenlabs_mcp.tools.convai import (  # noqa: E402, F401
    add_knowledge_base_to_agent,
    create_agent,
    get_agent,
    get_conversation,
    list_agents,
    list_conversations,
    simulate_conversation,
)
from elevenlabs_mcp.tools.music import (  # noqa: E402, F401
    compose_music,
    create_composition_plan,
    upload_music_for_inpainting,
    video_to_music,
)
from elevenlabs_mcp.tools.stt import (  # noqa: E402, F401
    format_diarized_transcript,
    speech_to_text,
)
from elevenlabs_mcp.tools.telephony import (  # noqa: E402, F401
    list_phone_numbers,
    make_outbound_call,
)
from elevenlabs_mcp.tools.tts import speech_to_speech, text_to_speech  # noqa: E402, F401
from elevenlabs_mcp.tools.voices import (  # noqa: E402, F401
    create_voice_from_preview,
    get_voice,
    list_models,
    search_voice_library,
    search_voices,
    text_to_voice,
    voice_clone,
)


def _is_broken_pipe_error(exc: BaseException) -> bool:
    """Check if an exception is a BrokenPipeError or contains only BrokenPipeErrors."""
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_broken_pipe_error(e) for e in exc.exceptions)
    return False


def main():
    """Run the MCP server"""
    # Log to stderr: stdout carries the JSON-RPC stdio stream and must stay clean.
    print("Starting MCP server", file=sys.stderr)
    try:
        mcp.run()
    except (BrokenPipeError, KeyboardInterrupt):
        pass  # Ignore broken pipe and keyboard interrupt errors
    except (Exception, BaseExceptionGroup) as err:
        if not _is_broken_pipe_error(err):
            raise
    finally:
        try:
            sys.stdout.close()
            sys.stderr.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
