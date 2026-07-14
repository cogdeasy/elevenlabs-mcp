"""Telephony tools: outbound calls and phone number management."""

from mcp.types import TextContent, ToolAnnotations

from elevenlabs_mcp import server
from elevenlabs_mcp.utils import make_error


def _get_phone_number_by_id(phone_number_id: str):
    """Helper function to get phone number details by ID."""
    phone_numbers = server.client.conversational_ai.phone_numbers.list()
    for phone in phone_numbers:
        if phone.phone_number_id == phone_number_id:
            return phone
    make_error(f"Phone number with ID {phone_number_id} not found.")


@server.mcp.tool(
    annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True),
    description="""Make an outbound call using an ElevenLabs agent. Automatically detects provider type (Twilio or SIP trunk) and uses the appropriate API.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: The ID of the agent that will handle the call
        agent_phone_number_id: The ID of the phone number to use for the call
        to_number: The phone number to call (E.164 format: +1xxxxxxxxxx)

    Returns:
        TextContent containing information about the call
    """,
)
def make_outbound_call(
    agent_id: str,
    agent_phone_number_id: str,
    to_number: str,
) -> TextContent:
    # Get phone number details to determine provider type
    phone_number = _get_phone_number_by_id(agent_phone_number_id)

    if phone_number.provider.lower() == "twilio":
        response = server.client.conversational_ai.twilio.outbound_call(
            agent_id=agent_id,
            agent_phone_number_id=agent_phone_number_id,
            to_number=to_number,
        )
        provider_info = "Twilio"
    elif phone_number.provider.lower() == "sip_trunk":
        response = server.client.conversational_ai.sip_trunk.outbound_call(
            agent_id=agent_id,
            agent_phone_number_id=agent_phone_number_id,
            to_number=to_number,
        )
        provider_info = "SIP trunk"
    else:
        make_error(f"Unsupported provider type: {phone_number.provider}")

    return TextContent(
        type="text", text=f"Outbound call initiated via {provider_info}: {response}."
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="List all phone numbers associated with the ElevenLabs account",
)
def list_phone_numbers() -> TextContent:
    """List all phone numbers associated with the ElevenLabs account.

    Returns:
        TextContent containing formatted information about the phone numbers
    """
    response = server.client.conversational_ai.phone_numbers.list()

    if not response:
        return TextContent(type="text", text="No phone numbers found.")

    phone_info = []
    for phone in response:
        assigned_agent = "None"
        if phone.assigned_agent:
            assigned_agent = f"{phone.assigned_agent.agent_name} (ID: {phone.assigned_agent.agent_id})"

        phone_info.append(
            f"Phone Number: {phone.phone_number}\n"
            f"ID: {phone.phone_number_id}\n"
            f"Provider: {phone.provider}\n"
            f"Label: {phone.label}\n"
            f"Assigned Agent: {assigned_agent}"
        )

    formatted_info = "\n\n".join(phone_info)
    return TextContent(type="text", text=f"Phone Numbers:\n\n{formatted_info}")
