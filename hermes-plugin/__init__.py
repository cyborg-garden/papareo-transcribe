"""papareo_transcribe: transcripts of recordings, escalated to Papa Reo when
they may contain te reo Māori.

One tool, ``transcribe_recording(path, engine)``. It runs local whisper and
sends the local transcript to this chat, or, when the person asks for Papa
Reo (engine="papareo") and a safe way out exists, sends the original audio
to Papa Reo, puts the result in the ovenmitt vault and has the ovenmitt
bridge send it here. Papa Reo text never reaches the model. Automatic
detection of te reo (the may_contain_mi project) is not wired in yet.

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
from .local import LocalWhisper  # noqa: E402
from .papareo import PapareoClient  # noqa: E402

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

KEY_ENV = "PAPAREO_API_KEY"
SOCKET_ENV = "OVENMITT_SOCKET"
DEFAULT_SOCKET = "/run/ovenmitt/ovenmitt.sock"
SEND_TOOL = "ovenmitt_vault_send"

SCHEMA: dict[str, Any] = {
    "name": "transcribe_recording",
    "description": (
        "Transcribe a recording (a voice note or an audio file the user sent) and send the "
        "transcript to this chat as a .txt file. By default it is transcribed locally and you get "
        "the text back. With engine='papareo' it goes to Papa Reo instead (te reo Māori speech "
        "recognition, a paid call) and you get only a handle: the transcript is protected, and you "
        "must not try to read it. engine: 'auto' (default, local for now); 'papareo' when the user "
        "asks for Papa Reo or says it is in Māori; 'local' for a plain NZ English transcript."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the recording, as the message gave it."},
            "engine": {"type": "string", "enum": list(transcribe.ENGINES), "default": "auto"},
        },
        "required": ["path"],
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
        engine = str(args.get("engine") or "auto")
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
