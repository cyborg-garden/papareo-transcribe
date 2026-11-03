"""Getting a transcript to the person who asked, in the chat they asked in.

Local transcripts are ordinary files: written to a private temp dir, sent
with the chat's own platform adapter, deleted. Papa Reo transcripts never
take this path; they go through the ovenmitt bridge's send tool. The chat
always comes from the session the gateway bound for this turn, never from
the model.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEND_TIMEOUT_S = 180.0
# Gateway cache names: voice notes are audio_<hex>.<ext> (no user filename);
# documents keep theirs after a doc_<hex>_ prefix.
_GENERATED = re.compile(r"^(audio|voice|video|img)_[0-9a-f]{8,}$")
_DOC_PREFIX = re.compile(r"^doc_[0-9a-f]{8,}_")


class NoChat(Exception):
    """This tool call isn't bound to a chat with a live adapter."""


def now() -> datetime:
    try:
        import hermes_time  # type: ignore[import-not-found]

        return hermes_time.now()
    except Exception:  # noqa: BLE001 - outside the harness
        return datetime.now().astimezone()


def source_stem(path: str) -> str | None:
    """The user's own name for the recording, if it has one."""
    stem = Path(path).stem
    stem = _DOC_PREFIX.sub("", stem)
    if not stem or _GENERATED.match(stem):
        return None
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return stem[:40] or None


def filename_for(path: str, when: datetime) -> str:
    """transcript-<yyyymmdd-hhmm>[-<source stem>].txt; never an engine name."""
    name = f"transcript-{when:%Y%m%d-%H%M}"
    stem = source_stem(path)
    return f"{name}-{stem}.txt" if stem else f"{name}.txt"


def session_platform() -> str | None:
    try:
        from gateway.session_context import get_session_env  # type: ignore[import-not-found]
    except ImportError:
        return None
    return get_session_env("HERMES_SESSION_PLATFORM", "") or None


def _binding() -> tuple[Any, str, Any, Any]:
    """This turn's adapter, chat, thread and gateway loop."""
    from gateway.config import Platform  # type: ignore[import-not-found]
    from gateway.run import _gateway_runner_ref  # type: ignore[import-not-found]
    from gateway.session_context import get_session_env  # type: ignore[import-not-found]

    platform_name = get_session_env("HERMES_SESSION_PLATFORM", "")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    if not platform_name or not chat_id:
        raise NoChat("this turn is not bound to a chat")
    runner = _gateway_runner_ref() if _gateway_runner_ref else None
    if runner is None:
        raise NoChat("no running gateway")
    platform = Platform(platform_name)
    profile = get_session_env("HERMES_SESSION_PROFILE", "") or None
    lookup = getattr(runner, "_authorization_adapter", None)
    adapter = lookup(platform, profile) if lookup else runner.adapters.get(platform)
    if adapter is None:
        raise NoChat(f"no live {platform_name} adapter")
    thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "")
    metadata = {"thread_id": thread_id} if thread_id else None
    return adapter, chat_id, metadata, getattr(runner, "_gateway_loop", None)


def _sender() -> tuple[Any, Any]:
    adapter, chat_id, metadata, loop = _binding()

    async def send(file_path: str, caption: str | None) -> Any:
        return await adapter.send_document(chat_id, file_path, caption=caption, metadata=metadata)

    return send, loop


def _run(coro: Any, loop: Any) -> Any:
    if loop is not None and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=SEND_TIMEOUT_S)
    return asyncio.run(coro)


def send_local(filename: str, text: str, caption: str | None) -> None:
    """Send an unprotected transcript to this chat as a .txt attachment."""
    send, loop = _sender()
    _send_file(send, loop, filename, text, caption)


def bind() -> tuple[str, str, Any, Any]:
    """This chat, held for a background job that outlives the turn: its platform, its
    id, a way to attach a file, and a way to say something."""
    adapter, chat_id, metadata, loop = _binding()
    platform = session_platform() or ""

    async def send(file_path: str, caption: str | None) -> Any:
        return await adapter.send_document(chat_id, file_path, caption=caption, metadata=metadata)

    def attach(filename: str, text: str, caption: str | None) -> None:
        _send_file(send, loop, filename, text, caption)

    def say(message: str) -> None:
        try:
            _run(adapter.send(chat_id, message, metadata=metadata), loop)
        except Exception as e:  # noqa: BLE001 - nothing more can be done, and the job must still end
            logger.warning("papareo_transcribe: couldn't send a job message: %s", type(e).__name__)

    return platform, chat_id, attach, say


def _send_file(send: Any, loop: Any, filename: str, text: str, caption: str | None) -> None:
    tmpdir = tempfile.mkdtemp(prefix="transcript-")  # mode 0700
    try:
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        result = _run(send(path, caption), loop)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not getattr(result, "success", False):
        raise RuntimeError(str(getattr(result, "error", None) or "the platform did not accept the file"))
