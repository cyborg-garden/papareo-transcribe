"""transcribe_recording: local pass, detector, escalation and delivery.

The rule for escalation: check there is a safe way out before spending
money on Papa Reo. The safe way out is the vault plus a send destination
the operator has declared for this chat's platform. If either is missing,
nothing is spent and the user gets the local transcript with a note.

What the agent gets back: for a local transcript, the text (it is
unprotected, so the agent may discuss it); for a Papa Reo transcript, only
a handle. Papa Reo text never appears in a return value or a log line.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .detector import Decision
from .local import LocalResult
from .papareo import NotConfigured, PapareoClient, PapareoError, with_footer

logger = logging.getLogger(__name__)

# Swarm Map's Transcription card sets this vault up; transcripts go nowhere else.
VAULT = "papareo"
# The operator's export folder for this vault (an `fs` destination), if any.
FOLDER = "folder"
ENGINES = ("auto", "papareo", "local")
SUPPORTED = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".opus",
    ".aac", ".flac", ".amr", ".3gp", ".caf", ".mov",
}
MAX_BYTES = 100 * 1024 * 1024


@dataclass
class Deps:
    local: Callable[[str], LocalResult]
    detect: Callable[[str, str], Decision]
    papareo: PapareoClient
    vault: Callable[[], Any]  # -> ovenmitt.VaultClient
    send_local: Callable[[str, str, str | None], None]  # filename, text, caption
    send_handle: Callable[[str, str, str | None], dict]  # handle, filename, caption
    platform: Callable[[], str | None]
    # Why the ovenmitt bridge couldn't send in this turn, or None when it can.
    bridge_problem: Callable[[], str | None]
    now: Callable[[], datetime]
    filename_for: Callable[[str, datetime], str]


def _caption(note: str | None = None) -> str:
    return f"Transcript. {note}" if note else "Transcript"


def check_path(path: str) -> str | None:
    p = Path(path)
    if not path or not p.is_file():
        return "no recording at that path"
    if p.suffix.lower() not in SUPPORTED:
        return f"unsupported audio type: {p.suffix or '(none)'}"
    if p.stat().st_size > MAX_BYTES:
        return "recording is too large"
    return None


def preflight(deps: Deps) -> str | None:
    """Why escalation can't safely happen right now, or None if it can."""
    if not deps.papareo.configured:
        return deps.papareo.why_not()
    bridge_problem = deps.bridge_problem()
    if bridge_problem:
        return bridge_problem
    platform = deps.platform()
    if not platform:
        return "this conversation has no chat to send a protected file to"
    try:
        vault = deps.vault()
        vault.health()
        names = {d.name for d in vault.destinations(VAULT) if d.kind == "send"}
    except Exception:  # noqa: BLE001 - daemon down, vault missing, socket absent
        return "the protected-output vault is not reachable"
    if platform not in names:
        return f"the operator has not allowed protected files on {platform}"
    return None


def run(path: str, engine: str, deps: Deps) -> dict[str, Any]:
    if engine not in ENGINES:
        return {"sent": False, "error": f"engine must be one of {', '.join(ENGINES)}"}
    problem = check_path(path)
    if problem:
        return {"sent": False, "error": problem}
    path = os.path.abspath(path)
    filename = deps.filename_for(path, deps.now())

    local: LocalResult | None = None
    try:
        local = deps.local(path)
    except Exception:  # noqa: BLE001
        logger.exception("papareo_transcribe: local transcription failed")

    if engine == "local":
        if local is None:
            return {"sent": False, "error": "local transcription failed"}
        return _deliver_local(local, filename, None, deps)

    if engine == "auto":
        if local is None:
            escalate, reasons = True, ("local-failed",)
        else:
            try:
                decision = deps.detect(local.text, local.language)
                escalate, reasons = decision.escalate, decision.reasons
                logger.info(
                    "papareo_transcribe: detector escalate=%s reasons=%s score=%.3f",
                    decision.escalate, ",".join(decision.reasons), decision.score,
                )
            except Exception:  # noqa: BLE001 - biased to fire: a broken detector escalates
                logger.exception("papareo_transcribe: DETECTOR ERROR, escalating (this is a bug)")
                escalate, reasons = True, ("detector-error",)
        if not escalate:
            return _deliver_local(local, filename, None, deps)  # type: ignore[arg-type]
    else:
        reasons = ("asked",)

    why_not = preflight(deps)
    if why_not:
        logger.info("papareo_transcribe: escalation skipped before any spend: %s", why_not)
        return _fallback(local, filename, f"It may contain te reo Māori, but Papa Reo wasn't used: {why_not}.", deps)

    duration = local.duration if local is not None else None
    try:
        text = deps.papareo.transcribe(path, duration)
    except (NotConfigured, PapareoError) as e:
        logger.warning("papareo_transcribe: Papa Reo failed: %s", e)
        return _fallback(
            local, filename,
            f"Papa Reo didn't complete ({e}), so this is the local transcript. Ask again to retry Papa Reo.",
            deps,
        )
    except Exception as e:  # noqa: BLE001 - any other failure still gets the user the local transcript
        # Only the type: an unexpected error's message is not ours to vouch for.
        logger.warning("papareo_transcribe: Papa Reo failed unexpectedly: %s", type(e).__name__)
        return _fallback(
            local, filename,
            "Papa Reo didn't complete (an unexpected error), so this is the local transcript. "
            "Ask again to retry Papa Reo.",
            deps,
        )

    body = with_footer(text)
    del text
    try:
        info = deps.vault().put(
            body, VAULT, name=filename, mime="text/plain; charset=utf-8", labels=[filename]
        )
    except Exception:  # noqa: BLE001
        logger.exception("papareo_transcribe: vault put failed after a Papa Reo call")
        return _fallback(
            local, filename,
            "Papa Reo transcribed it, but storing the protected transcript failed, so nothing "
            "was kept. This is the local transcript instead.",
            deps,
            paid=True,
        )
    finally:
        del body

    _export_copy(info.handle, filename, deps)

    # From here on Papa Reo is paid for and the transcript is stored: whatever
    # the send does, the agent must leave with the handle.
    try:
        result = deps.send_handle(info.handle, filename, _caption())
    except Exception:  # noqa: BLE001
        logger.exception("papareo_transcribe: the send tool raised")
        result = {}
    if not (isinstance(result, dict) and result.get("sent")):
        code = _send_error_code(result)
        return {
            "sent": False,
            "engine": "papareo",
            "handle": info.handle,
            "filename": filename,
            "papareo_charged": True,
            "note": (
                f"The Papa Reo transcript is safe in the vault but couldn't be sent ({code}). "
                f"{_RESEND_HINT.get(code, 'It can be resent with ovenmitt_vault_send.')}"
            ),
        }
    logger.info("papareo_transcribe: Papa Reo transcript sent as %s (%s)", info.handle, ",".join(reasons))
    return {"sent": True, "engine": "papareo", "handle": info.handle, "filename": filename}


def _export_copy(handle: str, filename: str, deps: Deps) -> None:
    """Also write the transcript to the operator's export folder, when the
    vault has one. Best effort: the chat send is what the person asked for."""
    try:
        vault = deps.vault()
        if any(d.name == FOLDER and d.kind == "fs" for d in vault.destinations(VAULT)):
            vault.export(handle, FOLDER, filename)
    except Exception as e:  # noqa: BLE001 - EXISTS, an unwritable folder, the daemon gone
        logger.warning("papareo_transcribe: export to the folder failed: %s", getattr(e, "code", type(e).__name__))


# What has to change before ovenmitt_vault_send can succeed, by refusal code.
_RESEND_HINT = {
    "DESTINATION_NOT_FOUND": "It can be resent with ovenmitt_vault_send once the operator allows this platform.",
    "TARGET_MISMATCH": "It can be resent with ovenmitt_vault_send once the operator allows this platform.",
    "NOT_CONFIGURED": "It can be resent with ovenmitt_vault_send once the operator sets up sending on this agent.",
    "UNSUPPORTED": "This chat can't receive files; it can be resent with ovenmitt_vault_send from a chat that can.",
    "SEND_TIMEOUT": "The file may still have arrived; check before resending with ovenmitt_vault_send.",
}


def _send_error_code(result: Any) -> str:
    """The refusal code from the send tool's reply, whatever its shape: the
    bridge's {"error": {"code": ...}}, the harness's {"error": "<text>"} when a
    tool raised or is missing, or nothing usable at all."""
    err = result.get("error") if isinstance(result, dict) else None
    if isinstance(err, dict) and isinstance(err.get("code"), str) and err["code"]:
        return err["code"]
    return "SEND_FAILED"


def _deliver_local(local: LocalResult, filename: str, note: str | None, deps: Deps) -> dict[str, Any]:
    out: dict[str, Any] = {"engine": "local", "filename": filename, "text": local.text}
    if note:
        out["note"] = note
    try:
        deps.send_local(filename, local.text, _caption(note))
        out["sent"] = True
    except Exception as e:  # noqa: BLE001
        logger.warning("papareo_transcribe: couldn't attach the local transcript: %s", e)
        out["sent"] = False
        out["note"] = ((note + " ") if note else "") + "The file couldn't be attached; the text is above."
    return out


def _fallback(
    local: LocalResult | None, filename: str, note: str, deps: Deps, paid: bool = False
) -> dict[str, Any]:
    if local is None:
        return {"sent": False, "engine": "none", "note": note + " The local transcription failed too."}
    out = _deliver_local(local, filename, note, deps)
    if paid:
        out["papareo_charged"] = True
    return out
