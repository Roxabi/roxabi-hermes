"""Autonomous voice reuse: a final already spoken replays its audio instead of re-synthesizing."""

import asyncio
import json
import os

import pytest

from gateway import run_voice, voice_reuse
from gateway.config import Platform

FINAL = "Ton attention est ailleurs — Metalyde, Ether."


class _VoiceAdapter:
    platform = Platform.DISCORD

    def __init__(self):
        self.played = []

    def connected_voice_guild_id(self):
        return 42

    async def play_in_voice_channel(self, guild_id, path):
        # Read at play time: a replay must hand over a file that still exists.
        with open(path, "rb") as fh:
            self.played.append((path, fh.read()))
        return True


@pytest.fixture
def tts_calls(monkeypatch):
    calls = []

    def fake_tts(text, output_path=None, **_):
        calls.append(text)
        with open(output_path, "wb") as fh:
            fh.write(f"audio-{len(calls)}".encode())
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)
    monkeypatch.setattr(voice_reuse, "_voice_config",
                        lambda: ({"provider": "amandine"}, {"reuse_hours": 24}))
    return calls


def _speak(adapter, text=FINAL):
    return asyncio.run(run_voice.play_autonomous_voice(adapter, text))


def test_the_same_final_is_synthesized_once_then_replayed(tts_calls):
    adapter = _VoiceAdapter()

    assert _speak(adapter) is True
    assert _speak(adapter) is True

    assert tts_calls == [FINAL]
    (first_path, first_audio), (second_path, second_audio) = adapter.played
    assert second_path == first_path
    assert second_audio == first_audio == b"audio-1"
    assert os.path.isfile(first_path)


def test_another_final_or_other_tts_settings_synthesize_again(tts_calls, monkeypatch):
    adapter = _VoiceAdapter()
    _speak(adapter)

    _speak(adapter, "Pense à boire un verre d'eau.")
    monkeypatch.setattr(voice_reuse, "_voice_config",
                        lambda: ({"provider": "edge"}, {"reuse_hours": 24}))
    _speak(adapter)

    assert tts_calls == [FINAL, "Pense à boire un verre d'eau.", FINAL]


def test_with_reuse_off_every_final_is_synthesized_and_its_temp_audio_deleted(tts_calls, monkeypatch):
    monkeypatch.setattr(voice_reuse, "_voice_config",
                        lambda: ({"provider": "amandine"}, {"reuse_hours": 0}))
    adapter = _VoiceAdapter()

    _speak(adapter)
    _speak(adapter)

    assert len(tts_calls) == 2
    assert not any(os.path.exists(path) for path, _ in adapter.played)


def test_stale_audio_is_not_replayed(tts_calls):
    adapter = _VoiceAdapter()
    _speak(adapter)
    audio = adapter.played[0][0]
    manifest = os.path.join(os.path.dirname(audio), os.path.basename(audio).split(".")[0] + ".json")
    with open(manifest, encoding="utf-8") as fh:
        entry = json.load(fh)
    entry["written_at"] -= 25 * 3600
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(entry, fh)

    _speak(adapter)

    assert len(tts_calls) == 2


def test_a_failed_keep_still_speaks_the_fresh_audio(tts_calls, monkeypatch):
    def broken_move(*_):
        raise OSError("disk full")

    monkeypatch.setattr(voice_reuse.shutil, "move", broken_move)
    adapter = _VoiceAdapter()

    assert _speak(adapter) is True

    assert adapter.played[0][1] == b"audio-1"
    assert not os.path.exists(adapter.played[0][0])
