"""Oral rewrite truncation policy for Discord VC playback.

``max_tokens`` on the ``discord_vc_oral`` auxiliary call is a runaway/cost guard, NOT a length
policy: the shaping lives in the system prompt ("au plus 5 phrases courtes"). These tests pin
what happens when the cap is what stopped the model — the spoken script must never end
mid-word, while a complete rewrite must come back untouched.
"""

from types import SimpleNamespace

LONG_SOURCE = "## Recap\n" + ("- item path /tmp/example/foo\n" * 40)


def _stub_llm(monkeypatch, content: str, finish_reason: str) -> None:
    def fake_llm(**kwargs):
        assert kwargs["task"] == "discord_vc_oral"
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason)])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_llm)


def test_capped_rewrite_drops_the_dangling_tail(monkeypatch):
    """A cap-truncated rewrite is spoken up to its last finished sentence, never mid-word."""
    from gateway import run_voice

    _stub_llm(
        monkeypatch,
        "Deux paquets critiques sur roxabi-production. Sept advisories sur fast-uri. "
        "Le reste des dépend",
        "length")
    spoken = run_voice.oralize_for_discord_vc(LONG_SOURCE)
    assert spoken.endswith("fast-uri.")
    assert "dépend" not in spoken


def test_complete_rewrite_is_returned_untrimmed(monkeypatch):
    """finish_reason=stop is verbatim: cap handling must not become a global truncation policy."""
    from gateway import run_voice

    full = "Deux paquets critiques. Sept advisories. Rien d'autre à signaler."
    _stub_llm(monkeypatch, full, "stop")
    assert run_voice.oralize_for_discord_vc(LONG_SOURCE) == full


def test_capped_rewrite_without_a_finished_sentence_keeps_the_text(monkeypatch):
    """Floor: with no finished sentence, keep the clipped text — a three-word stub read aloud
    is worse than one clipped sentence."""
    from gateway import run_voice

    capped = "Le digest des alertes de sécurité sur les dépôts Roxabi indique que plusieurs paquets"
    _stub_llm(monkeypatch, capped, "length")
    assert run_voice.oralize_for_discord_vc(LONG_SOURCE) == capped


def test_missing_finish_reason_is_not_treated_as_capped(monkeypatch):
    """Providers that omit finish_reason must keep the old behaviour (no silent trimming)."""
    from gateway import run_voice

    text = "Cron ok. Deux tickets ouverts. Et une remarque finale sans ponctuation terminale"
    def fake_llm(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_llm)
    assert run_voice.oralize_for_discord_vc(LONG_SOURCE) == text


def test_a_short_plain_ack_skips_the_rewrite(monkeypatch):
    """A short plain acknowledgement is already speech: no auxiliary call, no padding."""
    from gateway import run_voice

    calls = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        raise AssertionError("the rewrite must not run for a short plain ack")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_llm)
    assert run_voice.oralize_for_discord_vc("C'est fait.") == "C'est fait."
    assert calls == []


def test_a_short_text_with_a_path_is_still_rewritten(monkeypatch):
    """The bypass is about shape, not length: a path reads badly aloud however short it is."""
    from gateway import run_voice

    _stub_llm(monkeypatch, "Le fichier de config est à jour.", "stop")
    spoken = run_voice.oralize_for_discord_vc("Done: /etc/hermes/config.yaml updated.")
    assert spoken == "Le fichier de config est à jour."


def test_the_bypass_can_be_disabled_by_config(monkeypatch):
    """skip_under_chars: 0 restores "every spoken final goes through the rewrite"."""
    from gateway import run_voice

    monkeypatch.setattr(run_voice, "_oral_settings", lambda: {**run_voice._ORAL_DEFAULTS,
                                                              "skip_under_chars": 0})
    _stub_llm(monkeypatch, "C'est fait.", "stop")
    assert run_voice.oralize_for_discord_vc("ok") == "C'est fait."


def test_disabling_the_rewrite_falls_back_to_stripped_prose(monkeypatch):
    """enabled: false must not call the model at all."""
    from gateway import run_voice

    monkeypatch.setattr(run_voice, "_oral_settings", lambda: {**run_voice._ORAL_DEFAULTS,
                                                              "enabled": False})

    def fake_llm(**kwargs):
        raise AssertionError("disabled rewrite must not call the model")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_llm)
    spoken = run_voice.oralize_for_discord_vc(LONG_SOURCE)
    assert spoken
    assert len(spoken) <= 500
