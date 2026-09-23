"""Cron response reuse: a script's ``reuseKey`` replays the final already written for it."""

import json
from unittest.mock import MagicMock, patch

import pytest

import cron.scheduler as scheduler
from cron import response_reuse
from cron import scheduler_script as sched_script


@pytest.fixture(autouse=True)
def _stub_runtime_provider():
    fake_runtime = {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "source": "stub",
        "requested_provider": None,
    }
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_runtime):
        yield


def _job(prompt="Say it in one sentence."):
    return {
        "id": "abc123def456",
        "name": "heartbeat",
        "prompt": prompt,
        "schedule": "*/5 * * * *",
        "script": "tick.sh",
    }


def _gate(reuse_key, minutes):
    """A tick whose facts move every time while its reuse key may stay put."""
    payload = {"heartbeat_candidate": {"inputs": [{"facts": {"minutes": minutes}}]}}
    if reuse_key is not None:
        payload["reuseKey"] = reuse_key
    return json.dumps(payload)


def _run(job, script_output, final="Ton attention est ailleurs — Metalyde, Ether.", **kwargs):
    agent = MagicMock()
    agent.run_conversation = MagicMock(return_value={"final_response": final, "messages": []})
    with patch.object(sched_script, "_run_job_script", return_value=(True, script_output)), \
            patch("run_agent.AIAgent", return_value=agent) as agent_cls:
        result = scheduler.run_job(job, **kwargs)
    return result, agent_cls.call_count


def test_same_reuse_key_replays_the_final_without_waking_the_agent():
    job = _job()
    (ok1, _, first, _), woke1 = _run(job, _gate("sha256:drift", 5.9))
    (ok2, doc2, second, err2), woke2 = _run(job, _gate("sha256:drift", 10.7), final="never asked")

    assert (ok1, woke1) == (True, 1)
    assert (ok2, err2, woke2) == (True, None, 0)
    assert second == first
    assert "reused final from" in doc2
    # The replay is a normal run document: context_from still finds its answer.
    assert doc2.rpartition("## Response")[2].strip() == first


def test_a_new_key_or_an_edited_prompt_wakes_the_agent_again():
    job = _job()
    _run(job, _gate("sha256:drift", 5.9))

    _, woke_new_key = _run(job, _gate("sha256:comms", 5.9))
    _, woke_new_prompt = _run(_job(prompt="Say it warmly."), _gate("sha256:drift", 5.9))

    assert woke_new_key == 1
    assert woke_new_prompt == 1


def test_a_script_without_a_key_always_wakes_the_agent():
    job = _job()
    _run(job, _gate(None, 5.9))

    _, woke = _run(job, _gate(None, 5.9))

    assert woke == 1


def test_a_silent_final_is_not_kept():
    job = _job()
    _run(job, _gate("sha256:drift", 5.9), final="[SILENT]")

    _, woke = _run(job, _gate("sha256:drift", 5.9))

    assert woke == 1


def test_a_manual_run_with_extra_context_neither_replays_nor_keeps():
    job = _job()
    _run(job, _gate("sha256:drift", 5.9))

    (_, _, manual, _), woke_manual = _run(
        job, _gate("sha256:drift", 5.9), final="Manual answer.", extra_prompt="Focus on Ether.")
    (_, _, after, _), woke_after = _run(job, _gate("sha256:drift", 5.9))

    assert (woke_manual, manual) == (1, "Manual answer.")
    assert woke_after == 0
    assert after == "Ton attention est ailleurs — Metalyde, Ether."


def test_the_reuse_window_bounds_replay(monkeypatch):
    job = _job()
    response_reuse.remember(job, "sha256:drift", "Même message.", now=1_000_000.0)

    assert response_reuse.recall(job, "sha256:drift", now=1_000_000.0 + 3600).response == "Même message."
    assert response_reuse.recall(job, "sha256:drift", now=1_000_000.0 + 25 * 3600) is None

    monkeypatch.setattr(response_reuse, "reuse_hours", lambda: 0)
    assert response_reuse.recall(job, "sha256:drift", now=1_000_000.0 + 60) is None


def test_a_final_carrying_media_is_not_kept(tmp_path):
    job = _job()
    response_reuse.remember(job, "sha256:clip", f"Voilà.\nMEDIA:{tmp_path / 'clip.ogg'}")

    assert response_reuse.recall(job, "sha256:clip") is None


@pytest.mark.parametrize("gate_line, expected", [
    ('{"heartbeat_candidate": {}, "reuseKey": "sha256:abc"}', "sha256:abc"),
    ('{"reuseKey": 42}', None),
    ('{"reuseKey": "   "}', None),
    ('{"reuseKey": "' + "x" * 257 + '"}', None),
    ('{"reuseKey": "sha256:abc"}\ntrailing prose', None),
])
def test_reuse_key_is_read_from_the_gate_line_only(gate_line, expected):
    assert scheduler._parse_reuse_key(gate_line) == expected
