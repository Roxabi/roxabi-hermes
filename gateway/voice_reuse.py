"""Autonomous voice reuse — replay audio already synthesized for the same spoken final.

``play_autonomous_voice`` turns a cron final into speech: an oral rewrite (an auxiliary model call
on anything but a short plain ack) then TTS (a provider call, often a paid quota). Both depend only
on the final text and the voice/tts settings, so a final that repeats — a replayed cron ``reuseKey``
final, or an agent that wrote the same line again — replays the audio kept for that text instead of
paying for both again. The key covers the text, the platform (container format) and the whole
``tts`` and ``voice.oral_rewrite`` sections: a settings change never replays audio made under the
old ones. Entries live ``voice.reuse_hours`` (``0`` disables) under the profile's audio cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Keep in sync with hermes_cli.config_defaults DEFAULT_CONFIG["voice"]["reuse_hours"].
DEFAULT_REUSE_HOURS = 24.0
_MAX_ENTRIES = 64


def _voice_config() -> tuple[dict, dict]:
    """``(tts section, voice section)`` of the live config; empty on any failure."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        return {}, {}
    tts, voice = cfg.get("tts"), cfg.get("voice")
    return (tts if isinstance(tts, dict) else {}), (voice if isinstance(voice, dict) else {})


def reuse_hours() -> float:
    try:
        return float(_voice_config()[1].get("reuse_hours", DEFAULT_REUSE_HOURS))
    except (TypeError, ValueError):
        return DEFAULT_REUSE_HOURS


def _cache_dir() -> Path:
    # Resolved per call: one gateway serves several profiles, each with its own audio cache.
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("cache/audio", "audio_cache") / "voice_reuse"


def key_for(text: str, platform) -> str:
    tts, voice = _voice_config()
    platform_name = str(getattr(platform, "value", platform) or "")
    material = [text, platform_name, tts, voice.get("oral_rewrite")]
    payload = json.dumps(material, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("written_at"), (int, float)):
        return None
    files = data.get("files")
    # Bare names only: the manifest must never point outside the cache directory.
    if not isinstance(files, list) or not files or not all(
            isinstance(name, str) and name and Path(name).name == name for name in files):
        return None
    return data


def recall(key: str, *, now: Optional[float] = None) -> Optional[list[str]]:
    """Cached audio files for ``key`` in play order, or None when absent, stale or incomplete."""
    window_s = reuse_hours() * 3600
    if window_s <= 0:
        return None
    directory = _cache_dir()
    manifest = _read_manifest(directory / f"{key}.json")
    current = time.time() if now is None else now
    if manifest is None or not 0 <= current - manifest["written_at"] <= window_s:
        return None
    paths = [directory / name for name in manifest["files"]]
    if not all(path.is_file() for path in paths):
        return None
    return [str(path) for path in paths]


def keep(key: str, paths: list[str], *, now: Optional[float] = None) -> list[str]:
    """Move freshly synthesized ``paths`` into the cache under ``key``; returns where they now are.

    The manifest is written last, so a concurrent ``recall`` sees a complete entry or none. On any
    failure the files stay (or are left) where they were, and the caller plays and deletes them as
    before: keeping is an optimisation, never a reason to go silent."""
    window_s = reuse_hours() * 3600
    if window_s <= 0 or not paths:
        return list(paths)
    current = time.time() if now is None else now
    directory = _cache_dir()
    moved: list[str] = []
    try:
        from utils import atomic_json_write

        directory.mkdir(parents=True, exist_ok=True)
        names = []
        for index, source in enumerate(paths):
            name = f"{key}.{index}{Path(source).suffix}"
            # shutil.move, not os.replace: TTS writes under the system temp dir, often another fs.
            shutil.move(source, directory / name)
            moved.append(str(directory / name))
            names.append(name)
        atomic_json_write(directory / f"{key}.json", {"files": names, "written_at": current})
    except Exception as exc:
        logger.warning("Voice reuse: could not keep synthesized audio: %s", exc)
        return moved + list(paths[len(moved):])
    _prune(directory, now=current, window_s=window_s)
    return moved


def _prune(directory: Path, *, now: float, window_s: float) -> None:
    """Drop stale entries, then the oldest beyond ``_MAX_ENTRIES``, then orphaned audio."""
    try:
        manifests = []
        for path in directory.glob("*.json"):
            manifest = _read_manifest(path)
            if manifest is None or not 0 <= now - manifest["written_at"] <= window_s:
                _drop(directory, path, manifest)
            else:
                manifests.append((manifest["written_at"], path, manifest))
        manifests.sort(key=lambda item: item[0], reverse=True)
        for _, path, manifest in manifests[_MAX_ENTRIES:]:
            _drop(directory, path, manifest)
        # Audio left without a manifest (a crash between move and manifest) ages out on its own.
        for path in directory.iterdir():
            if path.suffix != ".json" and path.is_file() and now - path.stat().st_mtime > window_s:
                path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Voice reuse: prune skipped: %s", exc)


def _drop(directory: Path, manifest_path: Path, manifest: Optional[dict]) -> None:
    for name in (manifest or {}).get("files", []):
        (directory / name).unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
