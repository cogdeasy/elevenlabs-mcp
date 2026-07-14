"""Unit tests for all MCP tool handlers in elevenlabs_mcp.server.

The ElevenLabs client is mocked; no real API key or network access is needed.
"""

import base64
from io import IOBase
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import TextContent

import elevenlabs_mcp.server as server
from elevenlabs_mcp.utils import ElevenLabsMcpError


@pytest.fixture
def mock_client():
    with patch.object(server, "client") as client:
        yield client


def make_voice(voice_id="voice1", name="Adam", category="premade"):
    return SimpleNamespace(
        voice_id=voice_id,
        name=name,
        category=category,
        description="A voice",
        fine_tuning=SimpleNamespace(state={"eleven_multilingual_v2": "fine_tuned"}),
    )


class TestTextToSpeech:
    def test_default_voice(self, mock_client, temp_dir):
        mock_client.text_to_speech.convert.return_value = iter([b"audio-bytes"])
        result = server.text_to_speech(text="Hello", output_directory=str(temp_dir))
        assert isinstance(result, TextContent)
        assert "Success" in result.text
        mock_client.text_to_speech.convert.assert_called_once()
        _, kwargs = mock_client.text_to_speech.convert.call_args
        assert kwargs["voice_id"] == server.DEFAULT_VOICE_ID

    def test_voice_by_name(self, mock_client, temp_dir):
        voice = make_voice()
        mock_client.voices.search.return_value = SimpleNamespace(voices=[voice])
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])
        result = server.text_to_speech(
            text="Hi", voice_name="Adam", output_directory=str(temp_dir)
        )
        assert "Adam" in result.text

    def test_empty_text_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.text_to_speech(text="")

    def test_both_voice_id_and_name_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.text_to_speech(text="Hi", voice_id="v1", voice_name="Adam")

    def test_unknown_voice_name_raises(self, mock_client):
        mock_client.voices.search.return_value = SimpleNamespace(voices=[])
        with pytest.raises(ElevenLabsMcpError):
            server.text_to_speech(text="Hi", voice_name="Nonexistent")


class TestSpeechToText:
    def test_returns_transcript_directly(self, mock_client, sample_audio_file):
        mock_client.speech_to_text.convert.return_value = SimpleNamespace(
            text="hello world"
        )
        result = server.speech_to_text(
            input_file_path=str(sample_audio_file),
            save_transcript_to_file=False,
            return_transcript_to_client_directly=True,
        )
        assert result.text == "hello world"

    def test_saves_transcript_to_file(self, mock_client, sample_audio_file, temp_dir):
        mock_client.speech_to_text.convert.return_value = SimpleNamespace(text="hi")
        result = server.speech_to_text(
            input_file_path=str(sample_audio_file),
            output_directory=str(temp_dir),
        )
        assert isinstance(result, TextContent)
        saved = list(temp_dir.glob("stt_*.txt"))
        assert len(saved) == 1
        assert saved[0].read_text() == "hi"

    def test_no_output_mode_raises(self, mock_client, sample_audio_file):
        with pytest.raises(ElevenLabsMcpError):
            server.speech_to_text(
                input_file_path=str(sample_audio_file),
                save_transcript_to_file=False,
                return_transcript_to_client_directly=False,
            )


class TestTextToSoundEffects:
    def test_success(self, mock_client, temp_dir):
        mock_client.text_to_sound_effects.convert.return_value = iter([b"sfx"])
        result = server.text_to_sound_effects(
            text="explosion", output_directory=str(temp_dir)
        )
        assert isinstance(result, TextContent)

    @pytest.mark.parametrize("duration", [0.4, 5.1])
    def test_invalid_duration_raises(self, mock_client, duration):
        with pytest.raises(ElevenLabsMcpError):
            server.text_to_sound_effects(text="boom", duration_seconds=duration)


class TestSearchVoices:
    def test_returns_voices(self, mock_client):
        mock_client.voices.search.return_value = SimpleNamespace(
            voices=[make_voice(), make_voice(voice_id="voice2", name="Eve")]
        )
        result = server.search_voices(search="a")
        assert [v.name for v in result] == ["Adam", "Eve"]


class TestListModels:
    def test_returns_models(self, mock_client):
        lang = SimpleNamespace(language_id="en", name="English")
        model = SimpleNamespace(model_id="m1", name="Model 1", languages=[lang])
        mock_client.models.list.return_value = [model]
        result = server.list_models()
        assert result[0].id == "m1"
        assert result[0].languages[0].language_id == "en"


class TestGetVoice:
    def test_returns_voice(self, mock_client):
        mock_client.voices.get.return_value = make_voice()
        result = server.get_voice("voice1")
        assert result.id == "voice1"
        assert result.fine_tuning_status == {"eleven_multilingual_v2": "fine_tuned"}


class TestVoiceClone:
    def test_uploads_file_contents_not_path_strings(
        self, mock_client, sample_audio_file
    ):
        """Regression test for upstream issue #62: voice_clone must upload the
        file contents (binary handles), not path strings, otherwise the API
        treats the path text as audio data and reports corrupted files."""
        sample_audio_file.write_bytes(b"fake-mp3-data")
        mock_client.voices.ivc.create.return_value = make_voice(name="Clone")

        result = server.voice_clone(name="Clone", files=[str(sample_audio_file)])

        assert "Voice cloned successfully" in result.text
        _, kwargs = mock_client.voices.ivc.create.call_args
        files_arg = kwargs["files"]
        assert len(files_arg) == 1
        for f in files_arg:
            assert not isinstance(
                f, str
            ), "voice_clone must not pass path strings to the SDK"
            assert isinstance(f, IOBase)
        # Handles must be closed after the call
        assert all(f.closed for f in files_arg)

    def test_missing_file_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.voice_clone(name="Clone", files=["/nonexistent/file.mp3"])


class TestIsolateAudio:
    def test_success(self, mock_client, sample_audio_file, temp_dir):
        mock_client.audio_isolation.convert.return_value = iter([b"clean"])
        result = server.isolate_audio(
            input_file_path=str(sample_audio_file), output_directory=str(temp_dir)
        )
        assert isinstance(result, TextContent)


class TestCheckSubscription:
    def test_success(self, mock_client):
        sub = MagicMock()
        sub.model_dump_json.return_value = '{"tier": "free"}'
        mock_client.user.subscription.get.return_value = sub
        result = server.check_subscription()
        assert "free" in result.text


class TestCreateAgent:
    def test_success(self, mock_client):
        mock_client.conversational_ai.agents.create.return_value = SimpleNamespace(
            agent_id="agent1"
        )
        result = server.create_agent(
            name="Test", first_message="Hi", system_prompt="Be helpful"
        )
        assert "agent1" in result.text


class TestAddKnowledgeBaseToAgent:
    def _mock_agent(self, mock_client):
        agent = SimpleNamespace(
            conversation_config=SimpleNamespace(agent={"prompt": {}})
        )
        mock_client.conversational_ai.agents.get.return_value = agent

    def test_from_url(self, mock_client):
        self._mock_agent(mock_client)
        docs = mock_client.conversational_ai.knowledge_base.documents
        docs.create_from_url.return_value = SimpleNamespace(id="kb1")
        result = server.add_knowledge_base_to_agent(
            agent_id="agent1", knowledge_base_name="KB", url="https://example.com"
        )
        assert "kb1" in result.text

    def test_from_text(self, mock_client):
        self._mock_agent(mock_client)
        docs = mock_client.conversational_ai.knowledge_base.documents
        docs.create_from_file.return_value = SimpleNamespace(id="kb2")
        result = server.add_knowledge_base_to_agent(
            agent_id="agent1", knowledge_base_name="KB", text="Some text"
        )
        assert "kb2" in result.text

    def test_no_source_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.add_knowledge_base_to_agent(
                agent_id="agent1", knowledge_base_name="KB"
            )

    def test_multiple_sources_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.add_knowledge_base_to_agent(
                agent_id="agent1",
                knowledge_base_name="KB",
                url="https://example.com",
                text="Some text",
            )


class TestListAgents:
    def test_with_agents(self, mock_client):
        mock_client.conversational_ai.agents.list.return_value = SimpleNamespace(
            agents=[SimpleNamespace(name="A1", agent_id="agent1")]
        )
        result = server.list_agents()
        assert "A1" in result.text

    def test_empty(self, mock_client):
        mock_client.conversational_ai.agents.list.return_value = SimpleNamespace(
            agents=[]
        )
        assert "No agents found" in server.list_agents().text


class TestGetAgent:
    def test_success(self, mock_client):
        mock_client.conversational_ai.agents.get.return_value = SimpleNamespace(
            name="A1",
            agent_id="agent1",
            conversation_config=SimpleNamespace(tts=SimpleNamespace(voice_id="voice1")),
            metadata=SimpleNamespace(created_at_unix_secs=1700000000),
        )
        result = server.get_agent("agent1")
        assert "agent1" in result.text
        assert "voice1" in result.text


class TestGetConversation:
    def test_success(self, mock_client):
        mock_client.conversational_ai.conversations.get.return_value = SimpleNamespace(
            conversation_id="conv1",
            status="done",
            agent_id="agent1",
            transcript=[SimpleNamespace(role="user", message="hello")],
            metadata=None,
            analysis=None,
        )
        result = server.get_conversation("conv1")
        assert "conv1" in result.text
        assert "user: hello" in result.text

    def test_api_error_raises(self, mock_client):
        mock_client.conversational_ai.conversations.get.side_effect = RuntimeError(
            "boom"
        )
        with pytest.raises(ElevenLabsMcpError):
            server.get_conversation("conv1")


class TestSimulateConversation:
    def test_success(self, mock_client):
        mock_client.conversational_ai.agents.simulate_conversation.return_value = (
            SimpleNamespace(
                simulated_conversation=[
                    SimpleNamespace(role="user", message="hi", tool_calls=[])
                ],
                analysis=None,
            )
        )
        result = server.simulate_conversation(
            agent_id="agent1", simulated_user_prompt="Be a customer"
        )
        assert "Simulated Conversation" in result.text

    def test_invalid_criteria_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.simulate_conversation(
                agent_id="agent1",
                simulated_user_prompt="x",
                extra_evaluation_criteria=[{"id": "only-id"}],
            )


class TestListConversations:
    def test_success(self, mock_client):
        conv = SimpleNamespace(
            conversation_id="conv1",
            status="done",
            agent_name="A1",
            agent_id="agent1",
            start_time_unix_secs=1700000000,
            call_duration_secs=60,
            message_count=4,
            call_successful="success",
        )
        mock_client.conversational_ai.conversations.list.return_value = SimpleNamespace(
            conversations=[conv], has_more=False, next_cursor=None
        )
        result = server.list_conversations()
        assert "conv1" in result.text

    def test_empty(self, mock_client):
        mock_client.conversational_ai.conversations.list.return_value = SimpleNamespace(
            conversations=[], has_more=False, next_cursor=None
        )
        assert "No conversations found" in server.list_conversations().text


class TestSpeechToSpeech:
    def test_success(self, mock_client, sample_audio_file, temp_dir):
        mock_client.voices.search.return_value = SimpleNamespace(voices=[make_voice()])
        mock_client.speech_to_speech.convert.return_value = iter([b"converted"])
        result = server.speech_to_speech(
            input_file_path=str(sample_audio_file),
            voice_name="Adam",
            output_directory=str(temp_dir),
        )
        assert isinstance(result, TextContent)

    def test_unknown_voice_raises(self, mock_client, sample_audio_file):
        mock_client.voices.search.return_value = SimpleNamespace(voices=[])
        with pytest.raises(ElevenLabsMcpError):
            server.speech_to_speech(
                input_file_path=str(sample_audio_file), voice_name="Nobody"
            )


class TestTextToVoice:
    def test_success(self, mock_client, temp_dir):
        preview = SimpleNamespace(
            generated_voice_id="gen1",
            audio_base_64=base64.b64encode(b"audio").decode(),
        )
        mock_client.text_to_voice.create_previews.return_value = SimpleNamespace(
            previews=[preview]
        )
        result = server.text_to_voice(
            voice_description="A deep voice", output_directory=str(temp_dir)
        )
        assert isinstance(result, TextContent)
        assert "gen1" in result.text

    def test_empty_description_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.text_to_voice(voice_description="")


class TestCreateVoiceFromPreview:
    def test_success(self, mock_client):
        mock_client.text_to_voice.create.return_value = make_voice(name="NewVoice")
        result = server.create_voice_from_preview(
            generated_voice_id="gen1",
            voice_name="NewVoice",
            voice_description="desc",
        )
        assert "NewVoice" in result.text


class TestMakeOutboundCall:
    def _phone(self, provider):
        return SimpleNamespace(phone_number_id="phone1", provider=provider)

    def test_twilio(self, mock_client):
        mock_client.conversational_ai.phone_numbers.list.return_value = [
            self._phone("twilio")
        ]
        mock_client.conversational_ai.twilio.outbound_call.return_value = "ok"
        result = server.make_outbound_call(
            agent_id="agent1", agent_phone_number_id="phone1", to_number="+15550000000"
        )
        assert "Twilio" in result.text

    def test_sip_trunk(self, mock_client):
        mock_client.conversational_ai.phone_numbers.list.return_value = [
            self._phone("sip_trunk")
        ]
        mock_client.conversational_ai.sip_trunk.outbound_call.return_value = "ok"
        result = server.make_outbound_call(
            agent_id="agent1", agent_phone_number_id="phone1", to_number="+15550000000"
        )
        assert "SIP trunk" in result.text

    def test_unknown_phone_raises(self, mock_client):
        mock_client.conversational_ai.phone_numbers.list.return_value = []
        with pytest.raises(ElevenLabsMcpError):
            server.make_outbound_call(
                agent_id="agent1",
                agent_phone_number_id="missing",
                to_number="+15550000000",
            )


class TestSearchVoiceLibrary:
    def test_success(self, mock_client):
        voice = SimpleNamespace(
            name="Shared",
            voice_id="shared1",
            category="shared",
            gender=None,
            age=None,
            accent=None,
            description=None,
            use_case=None,
            verified_languages=[],
            preview_url=None,
        )
        mock_client.voices.get_shared.return_value = SimpleNamespace(voices=[voice])
        result = server.search_voice_library(search="Shared")
        assert "shared1" in result.text

    def test_empty(self, mock_client):
        mock_client.voices.get_shared.return_value = SimpleNamespace(voices=[])
        assert "No shared voices" in server.search_voice_library().text


class TestListPhoneNumbers:
    def test_success(self, mock_client):
        phone = SimpleNamespace(
            phone_number="+15550000000",
            phone_number_id="phone1",
            provider="twilio",
            label="Main",
            assigned_agent=None,
        )
        mock_client.conversational_ai.phone_numbers.list.return_value = [phone]
        result = server.list_phone_numbers()
        assert "phone1" in result.text

    def test_empty(self, mock_client):
        mock_client.conversational_ai.phone_numbers.list.return_value = []
        assert "No phone numbers found" in server.list_phone_numbers().text


class TestPlayAudio:
    def test_success(self, mock_client, sample_audio_file):
        sample_audio_file.write_bytes(b"audio")
        with patch.object(server, "play") as mock_play:
            result = server.play_audio(str(sample_audio_file))
        mock_play.assert_called_once()
        assert "Successfully played" in result.text


class TestComposeMusic:
    def test_prompt(self, mock_client, temp_dir):
        mock_client.music.compose.return_value = iter([b"music"])
        result = server.compose_music(
            prompt="happy tune", output_directory=str(temp_dir)
        )
        assert isinstance(result, TextContent)

    def test_store_for_inpainting(self, mock_client, temp_dir):
        mock_client.music.compose_detailed.return_value = SimpleNamespace(
            audio=b"music", song_id="song1"
        )
        result = server.compose_music(
            prompt="tune", output_directory=str(temp_dir), store_for_inpainting=True
        )
        assert "song1" in result.text

    def test_no_prompt_or_plan_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.compose_music()

    def test_prompt_and_plan_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.compose_music(prompt="x", composition_plan={"chunks": []})


class TestCreateCompositionPlan:
    def test_success(self, mock_client):
        plan = MagicMock()
        plan.model_dump.return_value = {"chunks": []}
        mock_client.music.composition_plan.create.return_value = plan
        result = server.create_composition_plan(prompt="calm piano")
        assert result == {"chunks": []}


class TestVideoToMusic:
    def test_success(self, mock_client, sample_video_file, temp_dir):
        sample_video_file.write_bytes(b"video")
        mock_client.music.video_to_music.return_value = iter([b"score"])
        result = server.video_to_music(
            input_file_paths=[str(sample_video_file)],
            output_directory=str(temp_dir),
        )
        assert isinstance(result, TextContent)

    def test_no_files_raises(self, mock_client):
        with pytest.raises(ElevenLabsMcpError):
            server.video_to_music(input_file_paths=[])


class TestUploadMusicForInpainting:
    def test_success(self, mock_client, sample_audio_file):
        sample_audio_file.write_bytes(b"audio")
        mock_client.music.upload.return_value = SimpleNamespace(
            song_id="song1", composition_plan=None
        )
        result = server.upload_music_for_inpainting(str(sample_audio_file))
        assert "song1" in result.text
