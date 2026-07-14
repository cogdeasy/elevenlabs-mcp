"""Account tools: subscription status."""

from mcp.types import TextContent, ToolAnnotations

from elevenlabs_mcp import server


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="Check the current subscription status. Could be used to measure the usage of the API.",
)
def check_subscription() -> TextContent:
    subscription = server.client.user.subscription.get()
    return TextContent(type="text", text=f"{subscription.model_dump_json(indent=2)}")
