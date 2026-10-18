"""papareo_transcribe: transcripts of recordings, escalated to Papa Reo when
they may contain te reo Māori.

One tool, ``transcribe_recording(path, engine)``, and the agent asks the
person which engine first. "local" runs whisper and sends the transcript to
this chat. "papareo", when a safe way out exists, sends the original audio
to Papa Reo (whisper doesn't run), puts the result in the ovenmitt vault and
has the ovenmitt bridge send it here. Papa Reo text never reaches the model.
Automatic detection of te reo (the may_contain_mi project) is not wired in.

The Papa Reo key arrives in the process environment (a sealed key: compose
``environment:``, never ``.env``) and is popped out of it as this module's
very first statement, before any import that could fail, so terminal
children never inherit it even when the plugin doesn't load.
"""

from __future__ import annotations

import os

# First, before anything that can raise. On importlib.reload the module dict
# survives, so a second import keeps the value the first one popped.
_POPPED_KEY: str | None = os.environ.pop("PAPAREO_API_KEY", None) or globals().get("_POPPED_KEY")

import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from typing import Any  # noqa: E402

from . import deliver, transcribe  # noqa: E402
from .local import LocalWhisper, has_audio, probe_duration  # noqa: E402
from .papareo import PapareoClient  # noqa: E402

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

KEY_ENV = "PAPAREO_API_KEY"
SOCKET_ENV = "OVENMITT_SOCKET"
DEFAULT_SOCKET = "/run/ovenmitt/ovenmitt.sock"
SEND_TOOL = "ovenmitt_vault_send"
COMPLIANT_ENV = "PAPAREO_KAITIAKITANGA_COMPLIANT"

SCHEMA: dict[str, Any] = {
    "name": "transcribe_recording",
    "description": (
        "Transcribe a recording (a voice note or an audio file the user sent) and send the "
        "transcript to this chat as a .txt file. Ask the person which engine first, unless they "
        "already said. engine='local': whisper on this machine, free, and you get the text back. "
        "engine='papareo': Papa Reo, Te Hiku Media's speech recognition for te reo Māori, a paid "
        "call; you get only a handle, because the transcript is protected and you must not try to "
        "read it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the recording, as the message gave it."},
            "engine": {"type": "string", "enum": ["local", "papareo"]},
        },
        "required": ["path", "engine"],
    },
}


NOT_SET_UP = "protected outputs (ovenmitt) are not set up on this agent"


def _bridge_problem() -> str | None:
    """Why the ovenmitt bridge couldn't send in this turn, or None when it can.

    Asked before any money is spent. The bridge's own send_readiness() knows
    whether it holds a credential and whether this turn is bound to a chat
    whose adapter can take a file; its tool merely existing proves neither.
    """
    try:
        from tools.registry import registry  # type: ignore[import-not-found]

        if registry.get_entry(SEND_TOOL) is None:
            return NOT_SET_UP
    except Exception:  # noqa: BLE001
        return NOT_SET_UP
    bridge = next(
        (m for name, m in list(sys.modules.items())
         if name.rsplit(".", 1)[-1] == "ovenmitt_bridge" and hasattr(m, "send_readiness")),
        None,
    )
    if bridge is None:
        return "the ovenmitt bridge can't confirm it can send (it needs version 0.2.0 or later)"
    try:
        return bridge.send_readiness()
    except Exception:  # noqa: BLE001
        logger.exception("papareo_transcribe: the bridge's send_readiness failed")
        return "the ovenmitt bridge couldn't check this chat"


def register(ctx: Any) -> None:
    """Hermes plugin entry point."""
    global _POPPED_KEY
    _POPPED_KEY = os.environ.pop(KEY_ENV, None) or _POPPED_KEY
    papareo = PapareoClient(_POPPED_KEY)
    socket_path = os.environ.get(SOCKET_ENV) or DEFAULT_SOCKET
    whisper = LocalWhisper()
    # Set by Swarm Map when the operator declares this agent compliant with
    # the Kaitiakitanga License. Not a secret, so not popped.
    compliant = os.environ.get(COMPLIANT_ENV) == "1"
    vault_client: list[Any] = []

    def vault() -> Any:
        if not vault_client:
            from ovenmitt import VaultClient

            vault_client.append(VaultClient(socket_path))
        return vault_client[0]

    def send_handle(handle: str, filename: str, caption: str | None) -> dict:
        raw = ctx.dispatch_tool(SEND_TOOL, {"handle": handle, "filename": filename, "caption": caption})
        try:
            return json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            return {"error": {"code": "SEND_FAILED", "message": "unreadable reply from the send tool"}}

    deps = transcribe.Deps(
        local=whisper.transcribe,
        duration=probe_duration,
        is_audio=has_audio,
        share_papareo_text=compliant,
        papareo=papareo,
        vault=vault,
        send_local=deliver.send_local,
        send_handle=send_handle,
        platform=deliver.session_platform,
        bridge_problem=_bridge_problem,
        now=deliver.now,
        filename_for=deliver.filename_for,
    )

    def handle(args: dict[str, Any], **_: Any) -> str:
        path = str(args.get("path") or "")
        engine = str(args.get("engine") or "auto")  # missing: refused with ASK_FIRST
        return json.dumps(transcribe.run(path, engine, deps), ensure_ascii=False)

    ctx.register_tool(
        name="transcribe_recording",
        toolset="papareo_transcribe",
        schema=SCHEMA,
        handler=handle,
        description=SCHEMA["description"],
        emoji="📝",
    )
    logger.info(
        "papareo_transcribe %s: registered transcribe_recording (papareo: %s)",
        __version__,
        "ready" if papareo.configured else papareo.why_not(),
    )
    if compliant:
        logger.info("papareo_transcribe: declared Kaitiakitanga-compliant; Papa Reo text is returned to the agent")
