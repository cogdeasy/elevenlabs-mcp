"""Integration tests: read elevenlabs:// resources from the MCP server over
stdio.

The server subprocess runs with a placeholder API key and a temporary base
path, so no real API key or network access is needed.
"""

import sys
import textwrap

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Bootstrap script template for the server subprocess: point the output base
# path at a test-controlled temp dir, import the server module with a
# placeholder API key, then serve over stdio.
SERVER_BOOTSTRAP_TEMPLATE = textwrap.dedent(
    """
    import os

    os.environ["ELEVENLABS_API_KEY"] = "test-api-key"
    os.environ["ELEVENLABS_MCP_BASE_PATH"] = {base_path!r}

    import elevenlabs_mcp.server as server

    server.main()
    """
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def server_params(temp_dir):
    bootstrap = SERVER_BOOTSTRAP_TEMPLATE.format(base_path=str(temp_dir))
    return StdioServerParameters(command=sys.executable, args=["-c", bootstrap])


@pytest.mark.anyio
async def test_stdio_list_resource_templates(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_resource_templates()
            uri_templates = {t.uriTemplate for t in result.resourceTemplates}
            assert "elevenlabs://{filename}" in uri_templates


@pytest.mark.anyio
async def test_stdio_read_text_resource(server_params, temp_dir):
    (temp_dir / "transcript.txt").write_text("stdio transcript")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource("elevenlabs://transcript.txt")
            contents = result.contents[0]
            assert contents.mimeType == "text/plain"
            assert "stdio transcript" in contents.text


@pytest.mark.anyio
async def test_stdio_read_missing_resource_errors(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            with pytest.raises(Exception):
                await session.read_resource("elevenlabs://missing.txt")
