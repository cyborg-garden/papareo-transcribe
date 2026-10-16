"""papareo_transcribe: transcripts of recordings, escalated to Papa Reo when
they may contain te reo Māori.

One tool, ``transcribe_recording(path, engine)``. It runs local whisper,
asks the detector, and either sends the local transcript to this chat or,
when the recording may contain Māori and a safe way out exists, sends the
original audio to Papa Reo, puts the result in the ovenmitt vault and has
the ovenmitt bridge send it here. Papa Reo text never reaches the model.

The Papa Reo key arrives in the process environment (a sealed key: compose
``environment:``, never ``.env``) and is popped into this module's closure
at ``register()``, so terminal children never inherit it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from . import deliver, detector, transcribe
from .local import LocalWhisper
from .papareo import PapareoClient

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
        "transcript to this chat as a .txt file. English is transcribed locally and you get the "
        "text back. If the recording may contain te reo Māori, it goes to Papa Reo instead and "
        "you get only a handle: the transcript is protected, and you must not try to read it. "
        "engine: 'auto' (default) decides; 'papareo' when the user asks for Papa Reo or says it is "
        "in Māori; 'local' when the user asks for a plain NZ English transcript."
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


def _bridge_available() -> bool:
    try:
        from tools.registry import registry  # type: ignore[import-not-found]

        return registry.get_entry(SEND_TOOL) is not None
    except Exception:  # noqa: BLE001
        return False


def register(ctx: Any) -> None:
    """Hermes plugin entry point."""
    papareo = PapareoClient(os.environ.pop(KEY_ENV, None) or None)
    socket_path = os.environ.get(SOCKET_ENV) or DEFAULT_SOCKET
    whisper = LocalWhisper()
    model = detector.TrigramModel.load()
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
        detect=lambda text, language: detector.decide(text, language, model),
        papareo=papareo,
        vault=vault,
        send_local=deliver.send_local,
        send_handle=send_handle,
        platform=deliver.session_platform,
        bridge_available=_bridge_available,
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
