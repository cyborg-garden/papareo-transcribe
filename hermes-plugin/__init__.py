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
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from . import deliver, fetch, transcribe  # noqa: E402
from .local import LocalWhisper, has_audio, probe_duration  # noqa: E402
from . import longjob  # noqa: E402
from .papareo import PapareoClient  # noqa: E402
from .speakers import Diarizer  # noqa: E402

logger = logging.getLogger(__name__)

__version__ = "0.5.0"

KEY_ENV = "PAPAREO_API_KEY"
SOCKET_ENV = "OVENMITT_SOCKET"
DEFAULT_SOCKET = "/run/ovenmitt/ovenmitt.sock"
SEND_TOOL = "ovenmitt_vault_send"
COMPLIANT_ENV = "PAPAREO_KAITIAKITANGA_COMPLIANT"
# Comma-separated platforms where, with no ovenmitt vault, Papa Reo
# transcripts are attached to the chat directly (e.g. "signal"). Not a secret.
DIRECT_ENV = "PAPAREO_DIRECT_PLATFORMS"
MAX_SPEAKERS = 32
DIRECT_ONLY = "long recordings can only be sent back where Papa Reo files may be attached directly"
CANT_MEASURE = "I couldn't tell how long that recording is, so nothing was spent. It may be damaged: ask them to check it plays"

SCHEMA: dict[str, Any] = {
    "name": "transcribe_recording",
    "description": (
        "Transcribe a recording (a voice note or an audio file the user sent) and send the "
        "transcript to this chat as a .txt file. Ask the person which engine first, unless they "
        "already said, and set chosen_by_person=true only when the person picked it: otherwise "
        "nothing runs. engine='local': whisper on this machine, free, and you get the text back. "
        "engine='papareo': Papa Reo, Te Hiku Media's speech recognition for te reo Māori, a paid "
        "call; the transcript is protected: you get only a handle (or nothing), and you must not try "
        "to read it. Give path for a file they sent, or url for a share link to one file (direct, "
        "OneDrive, Google Drive, Dropbox). speakers=true labels who spoke when, on this machine. "
        "voice_samples names speakers from short clips of each person; only use it when each person "
        "has agreed, and nothing about their voice is kept. Papa Reo recordings over an hour run in the "
        "background in parts: the first call explains what to confirm with the person."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the recording, as the message gave it."},
            "url": {"type": "string", "description": "A share link to one recording, instead of path."},
            "speakers": {"type": "boolean", "description": "True if the person wants speakers separated."},
            "num_speakers": {"type": "integer", "description": "How many people speak, if the person said."},
            "voice_samples": {
                "type": "object",
                "description": "Person's name -> path of a short clip of them, if the person sent clips.",
                "additionalProperties": {"type": "string"},
            },
            "long_job_ok": {"type": "boolean", "description": "True only once the person has agreed to a long job."},
            "resume_job": {"type": "boolean", "description": "True to carry on this chat's stopped long job. Nothing else is needed."},
            "discard_job": {"type": "boolean", "description": "True only if the person says to give up this chat's stopped long job, which deletes it. Parts already paid for are lost."},
            "engine": {"type": "string", "enum": ["local", "papareo"]},
            "chosen_by_person": {
                "type": "boolean",
                "description": "True only if the person told you which engine (in this conversation). "
                "Never guess or choose for them.",
            },
        },
        "required": ["engine", "chosen_by_person"],
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


def _voice_caches() -> list[Path]:
    """Where the gateway keeps what people send: voice notes in one place, and audio
    files such as .opus or .flac sent as documents in another."""
    caches = []
    for getter in ("get_audio_cache_dir", "get_document_cache_dir"):
        try:
            from gateway.platforms import base  # type: ignore[import-not-found]

            caches.append(Path(getattr(base, getter)()).resolve())
        except Exception:  # noqa: BLE001 - outside the harness
            continue
    return caches


def _forget_samples(samples: dict[str, str], recording: str = "") -> None:
    """Voice clips leave the gateway's caches once they've been used, and only those
    caches: a file somewhere else is the person's own. The recording itself is not
    a clip, even if it was also passed as one."""
    caches = _voice_caches()
    if not caches:
        return
    keep = Path(recording).resolve() if recording else None
    for sample in samples.values():
        try:
            resolved = Path(sample).resolve()
            if resolved != keep and any(resolved.is_relative_to(cache) for cache in caches):
                resolved.unlink(missing_ok=True)
        except OSError:
            logger.warning("papareo_transcribe: couldn't remove a voice clip from the cache")


def _read_samples(raw: Any) -> tuple[dict[str, str], str | None]:
    """voice_samples as {name: path}, also when it arrives as a JSON string."""
    if raw in (None, "", {}):
        return {}, None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    bad = "voice_samples must map each person's name to the path of their clip"
    if not isinstance(raw, dict):
        return {}, bad
    samples: dict[str, str] = {}
    for name, path in raw.items():
        if not (isinstance(name, str) and isinstance(path, str) and path):
            return {}, bad
        # A name goes on a line of its own in the transcript: it can't add lines.
        clean = re.sub(r"[\x00-\x1f\x7f\u2028\u2029]+", " ", name).strip()[:60]
        clean = clean.encode("utf-8", "replace").decode("utf-8")
        if not clean or clean in samples:
            return {}, bad
        samples[clean] = path
    return samples, None


def _read_count(raw: Any) -> tuple[int | None, str | None]:
    if raw in (None, "", 0):
        return None, None
    try:
        count = int(str(raw).strip()) if not isinstance(raw, bool) else 0
    except ValueError:
        count = 0
    if not 1 <= count <= MAX_SPEAKERS:
        return None, f"num_speakers must be a whole number from 1 to {MAX_SPEAKERS}"
    return count, None


def _this_chat() -> str:
    try:
        return deliver.bind().chat_id
    except Exception:  # noqa: BLE001 - no chat, so nothing can be waiting in it
        return ""


def _length(path: str) -> float | None:
    """A recording's length in seconds, by decoding it if its header won't say: a long
    file must not slip into the short flow and be paid for without agreement."""
    try:
        return probe_duration(path) or longjob.measure(path)
    except Exception:  # noqa: BLE001 - can't tell: the short flow copes
        return None


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
    direct = frozenset(p.strip().lower() for p in (os.environ.get(DIRECT_ENV) or "").split(",") if p.strip())
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
        direct_platforms=direct,
        papareo=papareo,
        vault=vault,
        send_local=deliver.send_local,
        send_handle=send_handle,
        platform=deliver.session_platform,
        bridge_problem=_bridge_problem,
        now=deliver.now,
        filename_for=deliver.filename_for,
        diarize=Diarizer(),
    )

    def job_env(samples: dict[str, str], attach: Any, say: Any, recording: str = "") -> longjob.Env:
        return longjob.Env(
            papareo=papareo, diarize=deps.diarize, attach=attach, say=say, samples=samples,
            samples_used=lambda: _forget_samples(samples, recording),
        )

    def run_long(job: longjob.Job, env: longjob.Env) -> None:
        longjob.run_job(job, env, transcribe.lay_out, transcribe.long_caption())

    def discard() -> dict:
        try:
            chat = deliver.bind()
        except Exception:  # noqa: BLE001
            return {"sent": False, "error": "this conversation has no chat to send to"}
        if longjob.is_busy():
            return {"sent": False, "error": longjob.BUSY}
        job = longjob.latest_for(chat.platform, chat.chat_id)
        if job is None:
            return {"sent": False, "error": "there is no stopped long job for this chat"}
        longjob.forget(job)
        return {"sent": False, "discarded": True}

    def resume() -> dict:
        if not transcribe.direct_ok(deps):
            return {"sent": False, "error": DIRECT_ONLY}
        try:
            chat = deliver.bind()
        except Exception:  # noqa: BLE001
            return {"sent": False, "error": "this conversation has no chat to send to"}
        if longjob.is_busy():
            return {"sent": False, "error": longjob.BUSY}
        if longjob.latest_for(chat.platform, chat.chat_id) is None:
            return {"sent": False, "error": "there is no stopped long job for this chat"}

        def work() -> None:
            # Read again inside the lock: it may have finished in between.
            job = longjob.latest_for(chat.platform, chat.chat_id)
            if job is not None:
                run_long(job, job_env({}, chat.attach, chat.say))

        if not longjob.start(work):
            return {"sent": False, "error": longjob.BUSY}
        return {"sent": False, "started": True, "next": longjob.NEXT_STARTED}

    def begin_long(
        path: str, duration: float, speakers: bool, count: int | None, samples: dict[str, str], agreed: bool, ours: bool
    ) -> dict:
        problem = longjob.refusal(path, duration, speakers, count, deps.diarize, transcribe.direct_ok(deps))
        if problem:
            return {"sent": False, "error": problem}
        if longjob.is_busy():
            return {"sent": False, "error": longjob.BUSY}
        waiting = longjob.latest_for(deps.platform() or "", _this_chat())
        if waiting is not None:
            return {"sent": False, "error": longjob.WAITING}
        if not agreed:
            return longjob.confirmation(duration)
        try:
            chat = deliver.bind()
        except Exception:  # noqa: BLE001
            return {"sent": False, "error": "this conversation has no chat to send to"}
        job = longjob.reserve(
            path, duration, chat.platform, chat.chat_id, deliver.filename_for(path, deps.now()), speakers, count,
            shutil.move if ours else shutil.copyfile, chat.thread_id, chat.profile, bool(samples),
        )
        env = job_env(samples, chat.attach, chat.say, path)
        if not longjob.start(lambda: run_long(job, env)):
            longjob.forget(job)
            return {"sent": False, "error": longjob.BUSY}
        return {"sent": False, "started": True, "next": longjob.NEXT_STARTED}

    longjob.recover(
        lambda job: deliver.bind_chat(job.platform, job.chat_id, job.thread_id, job.profile),
        lambda job, attach, say: run_long(job, job_env({}, attach, say)),
        lambda job: papareo.configured and job.platform in direct,
    )

    def handle(args: dict[str, Any], **_: Any) -> str:
        if args.get("resume_job") is True or str(args.get("resume_job")).lower() == "true":
            return json.dumps(resume(), ensure_ascii=False)
        if args.get("discard_job") is True or str(args.get("discard_job")).lower() == "true":
            return json.dumps(discard(), ensure_ascii=False)
        path = str(args.get("path") or "")
        engine = str(args.get("engine") or "auto")  # missing: refused with ASK_FIRST
        chosen = args.get("chosen_by_person") is True or str(args.get("chosen_by_person")).lower() == "true"
        url = str(args.get("url") or "").strip()
        wanted = args.get("speakers") is True or str(args.get("speakers")).lower() == "true"
        samples, bad = _read_samples(args.get("voice_samples"))
        count, bad_count = _read_count(args.get("num_speakers"))
        problem = bad or bad_count
        if problem:
            # Before anything is spent: dropping it quietly would lose the labels they asked for.
            return json.dumps({"sent": False, "error": problem}, ensure_ascii=False)
        tmp = None
        used = False
        try:
            refused = transcribe.refusal(engine, deps, chosen, samples)
            if refused:
                return json.dumps(refused, ensure_ascii=False)
            if url:
                tmp = tempfile.mkdtemp(prefix="recording-")  # mode 0700
                try:
                    path = fetch.fetch(url, tmp)
                except fetch.FetchError as e:
                    return json.dumps({"sent": False, "error": str(e)}, ensure_ascii=False)
                # Whatever its name says: a page saved as .mp3 must not be paid for.
                try:
                    audio = has_audio(path)
                except Exception:  # noqa: BLE001 - can't tell: not paid for
                    audio = False
                if not audio:
                    return json.dumps({"sent": False, "error": fetch.NEEDS_LINK}, ensure_ascii=False)

            if engine == "papareo" and not transcribe.check_path(path, has_audio, longjob.MAX_BYTES):
                seconds = _length(path)
                if seconds is None:
                    # It might be long, and long ones need agreement, so it isn't paid for.
                    return json.dumps({"sent": False, "error": CANT_MEASURE}, ensure_ascii=False)
                if seconds > longjob.LONG_AFTER_S:
                    agreed = args.get("long_job_ok") is True or str(args.get("long_job_ok")).lower() == "true"
                    return json.dumps(
                        begin_long(path, seconds, wanted or bool(samples) or bool(count), count, samples, agreed, bool(tmp)),
                        ensure_ascii=False,
                    )

            def mark_used() -> None:
                nonlocal used
                used = True

            result = transcribe.run(
                path, engine, deps, chosen_by_person=chosen, speakers=wanted, num_speakers=count,
                voice_samples=samples, samples_used=mark_used,
            )
            return json.dumps(result, ensure_ascii=False)
        finally:
            # Clips that were never used stay: the agent asks, then calls again.
            if used:
                _forget_samples(samples, path)
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

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
    if direct:
        logger.info("papareo_transcribe: Papa Reo files may be attached directly on: %s", ",".join(sorted(direct)))
    if compliant:
        logger.info("papareo_transcribe: declared Kaitiakitanga-compliant; Papa Reo text is returned to the agent")
