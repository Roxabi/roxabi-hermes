"""Text finals are spoken in a joined Discord VC through the delivery adapter.

Regression: the upstream rename of ``_adapter_for_source`` left one overlay call
in ``_send_voice_reply``. The lookup raised, the handler swallowed it, and a
text reply while the bot was in a voice channel never reached TTS.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_discord_text_final_is_spoken_in_joined_vc(tmp_path, monkeypatch):
    """A text final from any Discord chat is played in the joined VC, not sent as a bubble."""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    spoken = []

    def _fake_tts(*, text, output_path, **_ignored):
        spoken.append(text)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(b"\x00" * 32)
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _fake_tts)

    adapter = MagicMock()
    adapter.connected_voice_guild_id.return_value = 111
    adapter.is_in_voice_channel.side_effect = lambda guild_id: guild_id == 111
    adapter.play_in_voice_channel = AsyncMock(return_value=True)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    event = MessageEvent(
        text="salut",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="home", user_id="u1"),
        message_id="m1",
    )

    await runner._send_voice_reply(event, "Je suis la.")

    adapter.play_in_voice_channel.assert_awaited_once()
    assert adapter.play_in_voice_channel.await_args.args[0] == 111
    adapter.send_voice.assert_not_called()
    assert spoken == ["Je suis la."]
