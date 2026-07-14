"""Integration / e2e tests: spin up the MCP server over stdio and exercise
tool listing plus representative tool calls end-to-end.

The server subprocess runs with the ElevenLabs client replaced by a mock, so
no real API key or network access is needed.
"""

import sys
import textwrap

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Bootstrap script for the server subprocess: import the server module with a
# placeholder API key, swap the ElevenLabs client for a MagicMock with canned
# responses, then serve over stdio.
SERVER_BOOTSTRAP = textwrap.dedent(
    """
    import os
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    os.environ["ELEVENLABS_API_KEY"] = "test-api-key"

    import elevenlabs_mcp.server as server

    mock_client = MagicMock()
    mock_client.voices.search.return_value = SimpleNamespace(
        voices=[
            SimpleNamespace(voice_id="voice1", name="Adam", category="premade")
        ]
    )
    mock_client.conversational_ai.agents.list.return_value = SimpleNamespace(
        agents=[SimpleNamespace(name="Agent One", agent_id="agent1")]
    )
    mock_client.voices.ivc.create.return_value = SimpleNamespace(
        voice_id="cloned1",
        name="MyClone",
        category="cloned",
        description=None,
    )
    mock_client.text_to_speech.convert.return_value = [b"tts-audio-bytes"]
    mock_client.text_to_sound_effects.convert.return_value = [b"sfx-audio-bytes"]
    mock_client.user.subscription.get.return_value.model_dump_json.return_value = (
        '{"tier": "free"}'
    )
    mock_client.conversational_ai.phone_numbers.list.return_value = []
    mock_client.music.composition_plan.create.return_value.model_dump.return_value = {
        "chunks": []
    }
    server.client = mock_client
    server.play = MagicMock()

    server.main()
    """
)

EXPECTED_TOOLS = {
    "text_to_speech",
    "speech_to_text",
    "text_to_sound_effects",
    "search_voices",
    "list_models",
    "get_voice",
    "voice_clone",
    "isolate_audio",
    "check_subscription",
    "create_agent",
    "add_knowledge_base_to_agent",
    "list_agents",
    "get_agent",
    "get_conversation",
    "simulate_conversation",
    "list_conversations",
    "speech_to_speech",
    "text_to_voice",
    "create_voice_from_preview",
    "make_outbound_call",
    "search_voice_library",
    "list_phone_numbers",
    "play_audio",
    "compose_music",
    "create_composition_plan",
    "video_to_music",
    "upload_music_for_inpainting",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def server_params():
    return StdioServerParameters(command=sys.executable, args=["-c", SERVER_BOOTSTRAP])


@pytest.mark.anyio
async def test_stdio_list_tools(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tool_names = {tool.name for tool in result.tools}
            assert tool_names == EXPECTED_TOOLS


@pytest.mark.anyio
async def test_stdio_call_search_voices(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_voices", {"search": "Adam"})
            assert not result.isError
            text = result.content[0].text
            assert "Adam" in text
            assert "voice1" in text


@pytest.mark.anyio
async def test_stdio_call_list_agents(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_agents", {})
            assert not result.isError
            assert "Agent One" in result.content[0].text


@pytest.mark.anyio
async def test_stdio_call_voice_clone(server_params, sample_audio_file):
    sample_audio_file.write_bytes(b"fake-mp3-data")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "voice_clone",
                {"name": "MyClone", "files": [str(sample_audio_file)]},
            )
            assert not result.isError
            assert "Voice cloned successfully" in result.content[0].text


@pytest.mark.anyio
async def test_stdio_call_text_to_speech(server_params, temp_dir):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "text_to_speech",
                {"text": "Hello world", "output_directory": str(temp_dir)},
            )
            assert not result.isError
            assert "Success. File saved as:" in result.content[0].text


@pytest.mark.anyio
async def test_stdio_call_text_to_sound_effects(server_params, temp_dir):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "text_to_sound_effects",
                {"text": "thunder", "output_directory": str(temp_dir)},
            )
            assert not result.isError
            assert "Success. File saved as:" in result.content[0].text


@pytest.mark.anyio
async def test_stdio_call_check_subscription(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("check_subscription", {})
            assert not result.isError
            assert '"tier": "free"' in result.content[0].text


@pytest.mark.anyio
async def test_stdio_call_list_phone_numbers(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_phone_numbers", {})
            assert not result.isError
            assert "No phone numbers found" in result.content[0].text


@pytest.mark.anyio
async def test_stdio_call_create_composition_plan(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "create_composition_plan", {"prompt": "calm piano"}
            )
            assert not result.isError
            assert "chunks" in result.content[0].text


@pytest.mark.anyio
async def test_stdio_call_tool_error_is_reported(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("text_to_speech", {"text": ""})
            assert result.isError
