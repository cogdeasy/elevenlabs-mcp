"""Conversational AI tools: agents, knowledge bases, and conversations."""

from datetime import datetime
from io import BytesIO

from mcp.types import TextContent, ToolAnnotations

from elevenlabs import PromptEvaluationCriteria
from elevenlabs.types.knowledge_base_locator import KnowledgeBaseLocator

from elevenlabs_mcp import server
from elevenlabs_mcp.convai import create_conversation_config, create_platform_settings
from elevenlabs_mcp.utils import (
    handle_input_file,
    handle_large_text,
    make_error,
    parse_conversation_transcript,
)


@server.mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, openWorldHint=True),
    description="""Create a conversational AI agent with custom configuration.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        name: Name of the agent
        first_message: First message the agent will say i.e. "Hi, how can I help you today?"
        system_prompt: System prompt for the agent
        voice_id: ID of the voice to use for the agent
        language: ISO 639-1 language code for the agent
        llm: LLM to use for the agent
        temperature: Temperature for the agent. The lower the temperature, the more deterministic the agent's responses will be. Range is 0 to 1.
        max_tokens: Maximum number of tokens to generate.
        asr_quality: Quality of the ASR. `high` or `low`.
        model_id: ID of the ElevenLabs model to use for the agent.
        optimize_streaming_latency: Optimize streaming latency. Range is 0 to 4.
        stability: Stability for the agent. Range is 0 to 1.
        similarity_boost: Similarity boost for the agent. Range is 0 to 1.
        turn_timeout: Timeout for the agent to respond in seconds. Defaults to 7 seconds.
        max_duration_seconds: Maximum duration of a conversation in seconds. Defaults to 600 seconds (10 minutes).
        record_voice: Whether to record the agent's voice.
        retention_days: Number of days to retain the agent's data.
    """,
)
def create_agent(
    name: str,
    first_message: str,
    system_prompt: str,
    voice_id: str | None = server.DEFAULT_VOICE_ID,
    language: str = "en",
    llm: str = "gemini-2.0-flash-001",
    temperature: float = 0.5,
    max_tokens: int | None = None,
    asr_quality: str = "high",
    model_id: str = "eleven_turbo_v2",
    optimize_streaming_latency: int = 3,
    stability: float = 0.5,
    similarity_boost: float = 0.8,
    turn_timeout: int = 7,
    max_duration_seconds: int = 300,
    record_voice: bool = True,
    retention_days: int = 730,
) -> TextContent:
    conversation_config = create_conversation_config(
        language=language,
        system_prompt=system_prompt,
        llm=llm,
        first_message=first_message,
        temperature=temperature,
        max_tokens=max_tokens,
        asr_quality=asr_quality,
        voice_id=voice_id,
        model_id=model_id,
        optimize_streaming_latency=optimize_streaming_latency,
        stability=stability,
        similarity_boost=similarity_boost,
        turn_timeout=turn_timeout,
        max_duration_seconds=max_duration_seconds,
    )

    platform_settings = create_platform_settings(
        record_voice=record_voice,
        retention_days=retention_days,
    )

    response = server.client.conversational_ai.agents.create(
        name=name,
        conversation_config=conversation_config,
        platform_settings=platform_settings,
    )

    return TextContent(
        type="text",
        text=f"""Agent created successfully: Name: {name}, Agent ID: {response.agent_id}, System Prompt: {system_prompt}, Voice ID: {voice_id or "Default"}, Language: {language}, LLM: {llm}, You can use this agent ID for future interactions with the agent.""",
    )


@server.mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, openWorldHint=True),
    description="""Add a knowledge base to ElevenLabs workspace. Allowed types are epub, pdf, docx, txt, html.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: ID of the agent to add the knowledge base to.
        knowledge_base_name: Name of the knowledge base.
        url: URL of the knowledge base.
        input_file_path: Path to the file to add to the knowledge base.
        text: Text to add to the knowledge base.
    """,
)
def add_knowledge_base_to_agent(
    agent_id: str,
    knowledge_base_name: str,
    url: str | None = None,
    input_file_path: str | None = None,
    text: str | None = None,
) -> TextContent:
    provided_params = [
        param for param in [url, input_file_path, text] if param is not None
    ]
    if len(provided_params) == 0:
        make_error("Must provide either a URL, a file, or text")
    if len(provided_params) > 1:
        make_error("Must provide exactly one of: URL, file, or text")

    is_file_based = url is None

    if url is not None:
        response = (
            server.client.conversational_ai.knowledge_base.documents.create_from_url(
                name=knowledge_base_name,
                url=url,
            )
        )
    else:
        if text is not None:
            text_bytes = text.encode("utf-8")
            text_io = BytesIO(text_bytes)
            text_io.name = "text.txt"
            text_io.content_type = "text/plain"
            file = text_io
        elif input_file_path is not None:
            path = handle_input_file(
                file_path=input_file_path, audio_content_check=False
            )
            file = open(path, "rb")

        try:
            response = server.client.conversational_ai.knowledge_base.documents.create_from_file(
                name=knowledge_base_name,
                file=file,
            )
        finally:
            file.close()

    agent = server.client.conversational_ai.agents.get(agent_id=agent_id)

    agent_config = agent.conversation_config.agent
    knowledge_base_list = (
        agent_config.get("prompt", {}).get("knowledge_base", []) if agent_config else []
    )
    knowledge_base_list.append(
        KnowledgeBaseLocator(
            type="file" if is_file_based else "url",
            name=knowledge_base_name,
            id=response.id,
        )
    )

    if agent_config and "prompt" not in agent_config:
        agent_config["prompt"] = {}
    if agent_config:
        agent_config["prompt"]["knowledge_base"] = knowledge_base_list

    server.client.conversational_ai.agents.update(
        agent_id=agent_id, conversation_config=agent.conversation_config
    )
    return TextContent(
        type="text",
        text=f"""Knowledge base created with ID: {response.id} and added to agent {agent_id} successfully.""",
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="List all available conversational AI agents",
)
def list_agents() -> TextContent:
    """List all available conversational AI agents.

    Returns:
        TextContent with a formatted list of available agents
    """
    response = server.client.conversational_ai.agents.list()

    if not response.agents:
        return TextContent(type="text", text="No agents found.")

    agent_list = ",".join(
        f"{agent.name} (ID: {agent.agent_id})" for agent in response.agents
    )

    return TextContent(type="text", text=f"Available agents: {agent_list}")


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="Get details about a specific conversational AI agent",
)
def get_agent(agent_id: str) -> TextContent:
    """Get details about a specific conversational AI agent.

    Args:
        agent_id: The ID of the agent to retrieve

    Returns:
        TextContent with detailed information about the agent
    """
    response = server.client.conversational_ai.agents.get(agent_id=agent_id)

    voice_info = "None"
    if response.conversation_config.tts:
        voice_info = f"Voice ID: {response.conversation_config.tts.voice_id}"

    return TextContent(
        type="text",
        text=f"Agent Details: Name: {response.name}, Agent ID: {response.agent_id}, Voice Configuration: {voice_info}, Created At: {datetime.fromtimestamp(response.metadata.created_at_unix_secs).strftime('%Y-%m-%d %H:%M:%S')}",
    )


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="""Gets conversation with transcript. Returns: conversation details and full transcript. Use when: analyzing completed agent conversations.

    Args:
        conversation_id: The unique identifier of the conversation to retrieve, you can get the ids from the list_conversations tool.
    """,
)
def get_conversation(
    conversation_id: str,
) -> TextContent:
    """Get conversation details with transcript"""
    try:
        response = server.client.conversational_ai.conversations.get(conversation_id)

        # Parse transcript using utility function
        transcript, _ = parse_conversation_transcript(response.transcript)

        response_text = f"""Conversation Details:
ID: {response.conversation_id}
Status: {response.status}
Agent ID: {response.agent_id}
Message Count: {len(response.transcript)}

Transcript:
{transcript}"""

        if response.metadata:
            metadata = response.metadata
            duration = getattr(
                metadata,
                "call_duration_secs",
                getattr(metadata, "duration_seconds", "N/A"),
            )
            started_at = getattr(
                metadata, "start_time_unix_secs", getattr(metadata, "started_at", "N/A")
            )
            response_text += (
                f"\n\nMetadata:\nDuration: {duration} seconds\nStarted: {started_at}"
            )

        if response.analysis:
            analysis_summary = getattr(
                response.analysis, "summary", "Analysis available but no summary"
            )
            response_text += f"\n\nAnalysis:\n{analysis_summary}"

        return TextContent(type="text", text=response_text)

    except Exception as e:
        make_error(f"Failed to fetch conversation: {str(e)}")
        # satisfies type checker
        return TextContent(type="text", text="")


@server.mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, openWorldHint=True),
    description="""Simulate a text conversation between a conversational AI agent and a
    simulated user. Runs the full conversation and returns the transcript plus analysis.

    Use this to test agent behaviour, evaluate prompts, and catch failure modes without
    a live call. The simulated user follows the persona you describe.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs.
    Only use when explicitly requested by the user.

    Args:
        agent_id: ID of the agent to test. Use list_agents to find IDs.
        simulated_user_prompt: Instructions for how the simulated user should behave.
            Example: "You are a frustrated customer who cannot find the cancel button."
        first_message: Optional opening message to kick off the conversation.
        extra_evaluation_criteria: Optional list of dicts, each with:
            - id (str): unique key e.g. "issue_resolved"
            - name (str): human label e.g. "Issue Resolved"
            - conversation_goal_prompt (str): the assertion to check
              e.g. "The agent fully resolved the user's issue."
            - use_knowledge_base (bool, optional): whether the evaluator should
              reference the agent's knowledge base when judging. Defaults to False.
        max_turns: Maximum conversation turns. Defaults to 10.
    """,
)
def simulate_conversation(
    agent_id: str,
    simulated_user_prompt: str,
    first_message: str | None = None,
    extra_evaluation_criteria: list[dict] | None = None,
    max_turns: int = 10,
) -> TextContent:
    # Validate criteria fields early
    criteria_objects = []
    if extra_evaluation_criteria:
        for c in extra_evaluation_criteria:
            missing = [
                k for k in ("id", "name", "conversation_goal_prompt") if k not in c
            ]
            if missing:
                make_error(
                    f"Evaluation criterion missing fields {missing}. "
                    "Each criterion needs 'id', 'name', and 'conversation_goal_prompt'."
                )
                return TextContent(type="text", text="")
            criteria_objects.append(
                PromptEvaluationCriteria(
                    id=c["id"],
                    name=c["name"],
                    conversation_goal_prompt=c["conversation_goal_prompt"],
                    use_knowledge_base=c.get("use_knowledge_base", False),
                )
            )

    try:
        response = server.client.conversational_ai.agents.simulate_conversation(
            agent_id=agent_id,
            simulation_specification={
                "simulated_user_config": {
                    "prompt": simulated_user_prompt,
                    **({"first_message": first_message} if first_message else {}),
                }
            },
            extra_evaluation_criteria=criteria_objects or None,
            new_turns_limit=max_turns,
        )
    except Exception as e:
        make_error(f"Failed to simulate conversation: {str(e)}")
        return TextContent(type="text", text="")

    lines = ["## Simulated Conversation\n"]

    # Transcript
    history = getattr(response, "simulated_conversation", []) or []
    for turn in history:
        role = getattr(turn, "role", "unknown").capitalize()
        message = getattr(turn, "message", None)
        if message:
            lines.append(f"**{role}:** {message}")
        for tc in getattr(turn, "tool_calls", []) or []:
            lines.append(f"  _(tool: {getattr(tc, 'tool_name', '?')})_")

    # Analysis
    analysis = getattr(response, "analysis", None)
    if analysis:
        lines.append("\n## Analysis\n")
        summary = getattr(analysis, "transcript_summary", None)
        if summary:
            lines.append(f"**Summary:** {summary}\n")
        call_successful = getattr(analysis, "call_successful", None)
        if call_successful:
            lines.append(f"**Call result:** {call_successful}\n")
        eval_results = getattr(analysis, "evaluation_criteria_results", {}) or {}
        if eval_results:
            lines.append("**Criteria results:**")
            for key, result in eval_results.items():
                r = getattr(result, "result", "unknown")
                rationale = getattr(result, "rationale", "")
                lines.append(f"  **{key}**: {r} — {rationale}")

    return TextContent(type="text", text="\n".join(lines))


@server.mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    description="""Lists agent conversations. Returns: conversation list with metadata. Use when: asked about conversation history.

    Args:
        agent_id (str, optional): Filter conversations by specific agent ID
        cursor (str, optional): Pagination cursor for retrieving next page of results
        call_start_before_unix (int, optional): Filter conversations that started before this Unix timestamp
        call_start_after_unix (int, optional): Filter conversations that started after this Unix timestamp
        page_size (int, optional): Number of conversations to return per page (1-100, defaults to 30)
        max_length (int, optional): Maximum character length of the response text (defaults to 10000)
    """,
)
def list_conversations(
    agent_id: str | None = None,
    cursor: str | None = None,
    call_start_before_unix: int | None = None,
    call_start_after_unix: int | None = None,
    page_size: int = 30,
    max_length: int = 10000,
) -> TextContent:
    """List conversations with filtering options."""
    page_size = min(page_size, 100)

    try:
        response = server.client.conversational_ai.conversations.list(
            cursor=cursor,
            agent_id=agent_id,
            call_start_before_unix=call_start_before_unix,
            call_start_after_unix=call_start_after_unix,
            page_size=page_size,
        )

        if not response.conversations:
            return TextContent(type="text", text="No conversations found.")

        conv_list = []
        for conv in response.conversations:
            start_time = datetime.fromtimestamp(conv.start_time_unix_secs).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            conv_info = f"""Conversation ID: {conv.conversation_id}
Status: {conv.status}
Agent: {conv.agent_name or 'N/A'} (ID: {conv.agent_id})
Started: {start_time}
Duration: {conv.call_duration_secs} seconds
Messages: {conv.message_count}
Call Successful: {conv.call_successful}"""

            conv_list.append(conv_info)

        formatted_list = "\n\n".join(conv_list)

        pagination_info = f"Showing {len(response.conversations)} conversations"
        if response.has_more:
            pagination_info += f" (more available, next cursor: {response.next_cursor})"

        full_text = f"{pagination_info}\n\n{formatted_list}"

        # Use utility to handle large text content
        result_text = handle_large_text(full_text, max_length, "conversation list")

        # If content was saved to file, prepend pagination info
        if result_text != full_text:
            result_text = f"{pagination_info}\n\n{result_text}"

        return TextContent(type="text", text=result_text)

    except Exception as e:
        make_error(f"Failed to list conversations: {str(e)}")
        # This line is unreachable but satisfies type checker
        return TextContent(type="text", text="")
