"""Response reuse — replay a cron final instead of re-running the agent on input it already answered.

A pre-run script that knows two ticks would say the same thing prints ``reuseKey`` on its gate line
(the JSON object that may also carry ``wakeAgent``). The key is the script's claim; Hermes only scopes
it to the job's own prompt, skills and model pin, keeps the last deliverable final per key in the
job's output directory, and replays it for ``cron.response_reuse_hours`` instead of running the
agent. Delivery (text and autonomous voice) is unchanged: a replayed final is delivered like a fresh
one. Nothing is kept for a silent, failed, or media-carrying final, nor when the window is ``0``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Keep in sync with hermes_cli.config_defaults DEFAULT_CONFIG["cron"]["response_reuse_hours"].
DEFAULT_RESPONSE_REUSE_HOURS = 24.0
_STORE_FILENAME = "response_reuse.json"
_MAX_ENTRIES = 32


@dataclass(frozen=True)
class Recalled:
    """A final already written for this key, and when (epoch seconds)."""

    response: str
    written_at: float


def reuse_hours() -> float:
    """``cron.response_reuse_hours``: how long a written final may be replayed. ``0`` disables."""
    from cron.jobs import _cron_config_number

    return _cron_config_number("response_reuse_hours", DEFAULT_RESPONSE_REUSE_HOURS, float)


def _store_path(job_id: str) -> Path:
    from cron.jobs import _job_output_dir

    return _job_output_dir(job_id) / _STORE_FILENAME


def _scoped_key(job: dict, reuse_key: str) -> str:
    """The script's key within what else decides the answer: an edited prompt, skill list or
    model pin must never replay a final written under the old one."""
    from cron.scheduler_prompt import _job_skill_names

    material = [
        str(job.get("prompt") or ""),
        _job_skill_names(job),
        str(job.get("model") or ""),
        str(job.get("provider") or ""),
        reuse_key,
    ]
    return hashlib.sha256(json.dumps(material, ensure_ascii=False).encode("utf-8")).hexdigest()


def _fresh_entries(path: Path, *, now: float, window_s: float) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Response reuse: unreadable store %s (%s); starting empty", path, exc)
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {
        key: entry
        for key, entry in entries.items()
        if isinstance(entry, dict)
        and isinstance(entry.get("response"), str)
        and entry["response"].strip()
        and isinstance(entry.get("written_at"), (int, float))
        # A clock stepped backwards must not make an entry immortal.
        and 0 <= now - entry["written_at"] <= window_s
    }


def recall(job: dict, reuse_key: str, *, now: Optional[float] = None) -> Optional[Recalled]:
    """The final written for ``reuse_key`` inside the reuse window, else None."""
    window_s = reuse_hours() * 3600
    if window_s <= 0:
        return None
    current = time.time() if now is None else now
    entry = _fresh_entries(_store_path(job["id"]), now=current, window_s=window_s).get(
        _scoped_key(job, reuse_key))
    if entry is None:
        return None
    return Recalled(response=entry["response"], written_at=float(entry["written_at"]))


def remember(job: dict, reuse_key: str, response: str, *, now: Optional[float] = None) -> None:
    """Keep a deliverable final for ``reuse_key``. Best-effort: a failed write only costs a reuse."""
    window_s = reuse_hours() * 3600
    if window_s <= 0 or not response.strip():
        return
    from gateway.platforms.base import BasePlatformAdapter

    # Attachments are run-scoped files; a replay would point at paths that no longer exist.
    if BasePlatformAdapter.extract_media(response)[0]:
        return
    current = time.time() if now is None else now
    try:
        from cron.jobs import _ensure_cron_dir
        from utils import atomic_json_write

        path = _store_path(job["id"])
        entries = _fresh_entries(path, now=current, window_s=window_s)
        entries[_scoped_key(job, reuse_key)] = {"response": response, "written_at": current}
        newest = sorted(entries.items(), key=lambda item: item[1]["written_at"], reverse=True)
        _ensure_cron_dir(path.parent)
        atomic_json_write(path, {"version": 1, "entries": dict(newest[:_MAX_ENTRIES])})
    except Exception as exc:
        logger.warning("Job '%s': could not keep the final for reuse: %s", job.get("id"), exc)
