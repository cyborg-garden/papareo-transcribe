"""transcribe_recording: local pass, escalation and delivery.

One engine per call, chosen by the person: the agent asks first. "local"
runs whisper here; "papareo" sends the original audio to Papa Reo and never
runs whisper. Automatic detection ("may this contain te reo Māori?") lives
in its own project, may_contain_mi, and is not wired in yet: ``Deps.detect``
is the seam for it, and engine="auto" is refused until it is set.

The rule for escalation: check there is a safe way out before spending
money on Papa Reo. The safe way out is the vault plus a send destination
the operator has declared for this chat's platform. If either is missing,
nothing is spent and the user gets the local transcript with a note.

What the agent gets back: for a local transcript, the text (it is
unprotected, so the agent may discuss it); for a Papa Reo transcript, only
a handle. Papa Reo text never appears in a log line, and appears in a return
value only when the operator has declared the agent compliant with the
Kaitiakitanga License (``Deps.share_papareo_text``: Swarm Map's
PAPAREO_KAITIAKITANGA_COMPLIANT=1, set on the Transcription card).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .local import LocalResult
from .papareo import NotConfigured, PaidButLost, PapareoClient, PapareoError, with_footer
from .speakers import DiarizeError, Diarized, heard, render, split_at_turns

logger = logging.getLogger(__name__)

# Swarm Map's Transcription card sets this vault up; transcripts go nowhere else.
VAULT = "papareo"
# More voices than this, when nobody said how many, is the clustering splitting
# people up, not a crowd: real speech has given 50 "speakers". Labels like that
# are worse than none, and it's known before anything is paid for.
MAX_UNASKED_SPEAKERS = 8
# The operator's export folder for this vault (an `fs` destination), if any.
FOLDER = "folder"
ENGINES = ("auto", "papareo", "local")
SUPPORTED = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".opus",
    ".aac", ".flac", ".amr", ".3gp", ".caf", ".mov",
}
MAX_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class Decision:
    """What a detector says about a local transcript (the shape of
    may_contain_mi's Decision, without depending on it)."""

    escalate: bool
    reasons: tuple[str, ...]
    score: float


@dataclass
class Deps:
    local: Callable[[str], LocalResult]
    papareo: PapareoClient
    vault: Callable[[], Any]  # -> ovenmitt.VaultClient
    send_local: Callable[[str, str, str | None], None]  # filename, text, caption
    send_handle: Callable[[str, str, str | None], dict]  # handle, filename, caption
    platform: Callable[[], str | None]
    # Why the ovenmitt bridge couldn't send in this turn, or None when it can.
    bridge_problem: Callable[[], str | None]
    now: Callable[[], datetime]
    filename_for: Callable[[str, datetime], str]
    # Seconds of audio, for sizing the speaker separation timeout when whisper
    # didn't run (engine="papareo"). None if unknown.
    duration: Callable[[str], float | None] | None = None
    # Whether a file with an unknown extension (.bin, none) holds audio.
    is_audio: Callable[[str], bool] | None = None
    # The operator declared this agent compliant with the Kaitiakitanga
    # License, so Papa Reo text may be returned to it.
    share_papareo_text: bool = False
    # Platforms where, without an ovenmitt vault, the plugin may attach a Papa
    # Reo transcript to the chat itself (PAPAREO_DIRECT_PLATFORMS). The
    # operator's call that the platform itself is a fit for the licence (e.g.
    # Signal: end-to-end encrypted, no server copy). The text still never
    # reaches the model unless share_papareo_text is set.
    direct_platforms: frozenset[str] = frozenset()
    # (text, whisper language) -> Decision. None until may_contain_mi is
    # wired in; until then engine="auto" is refused (ask the person instead).
    detect: Callable[[str, str], Decision] | None = None
    # (path, number of speakers if known, name -> voice sample path, seconds).
    diarize: Callable[..., Diarized] | None = None


PAPAREO_SOURCE = "Papa Reo, Te Hiku Media"


def local_source(local: LocalResult) -> str:
    return f"whisper {local.model}, run locally" if local.model else "whisper, run locally"


def _caption(source: str, note: str | None = None) -> str:
    return f"Transcript ({source}). {note}" if note else f"Transcript ({source})"


def check_path(path: str, is_audio: Callable[[str], bool] | None = None, max_bytes: int = MAX_BYTES) -> str | None:
    p = Path(path)
    if not path or not p.is_file():
        return "no recording at that path"
    # Chat platforms often save attachments as .bin or with no extension, so
    # an unknown extension is checked by content before it is refused.
    if p.suffix.lower() not in SUPPORTED and not (is_audio and _safe(is_audio, path)):
        return f"not a recording this can read ({p.suffix or 'no extension'}, and no audio found in it)"
    if p.stat().st_size > max_bytes:
        return "recording is too large"
    return None


def long_caption() -> str:
    return _caption(PAPAREO_SOURCE)


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


ASK_FIRST = (
    "Nothing was transcribed. Ask the person first, warmly and in one message, for example: "
    "\"Happy to! Would you like a local transcript (free), or one from Papa Reo, which is best for "
    "te reo Māori (it costs a little)?\" Then call again with their answer as engine ('local' or "
    "'papareo') and chosen_by_person=true."
)


def _separate(
    path: str,
    num_speakers: int | None,
    samples: Mapping[str, str] | None,
    duration: float | None,
    deps: Deps,
    samples_used: Callable[[], None] | None = None,
) -> tuple[Diarized | None, str | None]:
    """Speakers for this recording, or why not. Local and free: it runs before
    any money is spent, so a failure costs nothing."""
    if deps.diarize is None:
        return None, "speaker separation is not installed on this agent"
    try:
        diarized = deps.diarize(path, num_speakers, samples, duration)
    except DiarizeError as e:
        logger.warning("papareo_transcribe: speaker separation failed: %s", e)
        if e.clips_used and samples_used:
            samples_used()
        return None, str(e)
    except Exception as e:  # noqa: BLE001 - it may have had them: lean towards not keeping them
        logger.warning("papareo_transcribe: speaker separation failed: %s", type(e).__name__)
        if samples_used:
            samples_used()
        return None, "speaker separation failed"
    if samples_used:
        samples_used()
    found = len({t.speaker for t in diarized.turns})
    if found > (num_speakers or MAX_UNASKED_SPEAKERS):
        logger.warning("papareo_transcribe: separation heard an unlikely %d voices", found)
        # One clause, read by the person too: a local transcript carries it as its
        # caption, and both callers add their own punctuation around it.
        clips = ", and the voice clips were deleted, so they'd need sending again" if samples else ""
        if num_speakers:
            return None, f"it heard {found} voices when told {num_speakers}, so this recording can't be separated reliably{clips}"
        return None, (
            f"it heard {found} voices, which is almost certainly wrong, "
            f"so it needs to know how many people are in the recording{clips}"
        )
    return diarized, None


def lay_out(cues: Any, text: str, diarized: Diarized | None, hints: Any = ()) -> tuple[str, Any, bool]:
    """The transcript as it will be delivered. Layout must never cost the words:
    if it fails for any reason, the plain text goes out. Never logs the text."""
    if not cues:
        return text, (), diarized is not None
    try:
        split = split_at_turns(cues, diarized.turns, hints) if diarized else cues
        return render(split, diarized), split, False
    except Exception as e:  # noqa: BLE001
        logger.warning("papareo_transcribe: layout failed: %s", type(e).__name__)
        return text, (), diarized is not None


def _finish(result: dict[str, Any], extra: dict[str, Any], no_labels: bool) -> dict[str, Any]:
    """Add the speaker fields without losing a note that is already there."""
    out = {**result, **extra}
    if no_labels:
        missing = "Papa Reo didn't return timings this time, so the transcript has no speaker labels."
        out["note"] = f"{result['note']} {missing}" if result.get("note") else missing
    return out


def _speaker_fields(cues: Any, diarized: Diarized | None) -> dict[str, Any]:
    if diarized is None:
        return {}
    speakers = heard(cues, diarized) if cues else set()
    named = {diarized.names[s] for s in speakers if s in diarized.names}
    brief = set(diarized.names.values()) - named
    out: dict[str, Any] = {"speakers": len(speakers)}
    if named:
        out["named_speakers"] = sorted(named)
    if diarized.unmatched:
        out["samples_not_matched"] = sorted(diarized.unmatched)
    if brief:
        out["samples_matched_too_briefly"] = sorted(brief)
    return out


def refusal(
    engine: str, deps: Deps, chosen_by_person: bool, voice_samples: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """Why this call would be refused whatever the recording: asked before any
    download as well as before any run."""
    if engine not in ENGINES:
        return {"sent": False, "error": f"engine must be one of {', '.join(ENGINES)}"}
    if (engine == "auto" and deps.detect is None) or (engine != "auto" and not chosen_by_person):
        return {"sent": False, "error": ASK_FIRST}
    for sample in (voice_samples or {}).values():
        problem = check_path(sample, deps.is_audio)
        if problem:
            return {"sent": False, "error": f"voice sample: {problem}"}
    return None


def run(
    path: str,
    engine: str,
    deps: Deps,
    chosen_by_person: bool = True,
    speakers: bool = False,
    num_speakers: int | None = None,
    voice_samples: Mapping[str, str] | None = None,
    samples_used: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """chosen_by_person: the person said which engine. The tool passes what
    the agent claims; without it nothing runs and the agent is told to ask."""
    speakers = speakers or bool(voice_samples) or bool(num_speakers)
    refused = refusal(engine, deps, chosen_by_person, voice_samples)
    if refused:
        return refused
    problem = check_path(path, deps.is_audio)
    if problem:
        return {"sent": False, "error": problem}
    path = os.path.abspath(path)
    filename = deps.filename_for(path, deps.now())

    # Whisper runs only when its transcript is wanted: never on the Papa Reo path.
    local: LocalResult | None = None
    if engine != "papareo":
        try:
            local = deps.local(path)
        except Exception:  # noqa: BLE001
            logger.exception("papareo_transcribe: local transcription failed")

    if engine == "local":
        if local is None:
            return {"sent": False, "error": "local transcription failed"}
        diarized, problem = (
            _separate(path, num_speakers, voice_samples, local.duration, deps, samples_used)
            if speakers else (None, None)
        )
        note = f"Speakers weren't separated: {problem}." if problem else None
        text, cues, _ = lay_out(local.cues, local.text, diarized)
        local = replace(local, text=text)
        return {**_deliver_local(local, filename, note, deps), **_speaker_fields(cues, diarized)}

    if engine == "auto":
        if local is None:
            escalate, reasons = True, ("local-failed",)
        else:
            try:
                decision = deps.detect(local.text, local.language)  # type: ignore[misc]
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
    direct = bool(why_not) and direct_ok(deps)
    if direct:
        why_not = None
    if why_not:
        logger.info("papareo_transcribe: escalation skipped before any spend: %s", why_not)
        return _fallback(local, filename, f"Papa Reo wasn't used, and nothing was spent: {why_not}.", deps)

    duration = local.duration if local is not None else _probe(deps, path)
    diarized: Diarized | None = None
    if speakers:
        diarized, problem = _separate(path, num_speakers, voice_samples, duration, deps, samples_used)
        if diarized is None:
            return {
                "sent": False,
                "engine": "none",
                "note": (
                    f"Speakers couldn't be separated ({problem}), so Papa Reo wasn't used and nothing was "
                    "spent. Ask whether they'd like the transcript without speaker labels."
                ),
            }
    try:
        timed = deps.papareo.transcribe_timed(path)
    except PaidButLost as e:
        logger.warning("papareo_transcribe: %s", e)
        return _fallback(local, filename, f"{e}. Running it again costs again, so ask first.", deps, paid=True)
    except (NotConfigured, PapareoError) as e:
        logger.warning("papareo_transcribe: Papa Reo failed: %s", e)
        return _fallback(local, filename, f"Papa Reo didn't complete ({e}). Ask again to retry Papa Reo.", deps)
    except Exception as e:  # noqa: BLE001 - any other failure still gets a clean answer
        # Only the type: an unexpected error's message is not ours to vouch for.
        logger.warning("papareo_transcribe: Papa Reo failed unexpectedly: %s", type(e).__name__)
        return _fallback(
            local, filename, "Papa Reo didn't complete (an unexpected error). Ask again to retry Papa Reo.", deps
        )

    text, cues, no_labels = lay_out(timed.cues, timed.text, diarized, timed.hints)
    extra = _speaker_fields(cues, diarized)
    del timed, cues
    body = with_footer(text)
    shared = text.strip() if deps.share_papareo_text else None
    del text
    if direct:
        return _finish(_deliver_direct(body, filename, reasons, shared, deps), extra, no_labels)
    try:
        info = deps.vault().put(
            body, VAULT, name=filename, mime="text/plain; charset=utf-8", labels=[filename]
        )
    except Exception:  # noqa: BLE001
        logger.exception("papareo_transcribe: vault put failed after a Papa Reo call")
        return {
            **_fallback(
                local, filename,
                "Papa Reo transcribed it, but storing the protected transcript failed, so nothing was kept.",
                deps,
                paid=True,
            ),
            **_shared(shared),
        }
    finally:
        del body

    _export_copy(info.handle, filename, deps)

    # From here on Papa Reo is paid for and the transcript is stored: whatever
    # the send does, the agent must leave with the handle.
    try:
        result = deps.send_handle(info.handle, filename, _caption(PAPAREO_SOURCE))
    except Exception:  # noqa: BLE001
        logger.exception("papareo_transcribe: the send tool raised")
        result = {}
    if not (isinstance(result, dict) and result.get("sent")):
        code = _send_error_code(result)
        return _finish({
            "sent": False,
            "engine": "papareo",
            "handle": info.handle,
            "filename": filename,
            "papareo_charged": True,
            "note": (
                f"The Papa Reo transcript is safe in the vault but couldn't be sent ({code}). "
                f"{_RESEND_HINT.get(code, 'It can be resent with ovenmitt_vault_send.')}"
            ),
            **_shared(shared),
        }, extra, no_labels)
    logger.info("papareo_transcribe: Papa Reo transcript sent as %s (%s)", info.handle, ",".join(reasons))
    return _finish({
        "sent": True,
        "engine": "papareo",
        "transcribed_by": PAPAREO_SOURCE,
        "handle": info.handle,
        "filename": filename,
        **_shared(shared),
        "next": NEXT_PAPAREO,
    }, extra, no_labels)


# What to do next, in every result: the skill may never have been read.
NEXT_LOCAL = (
    "The file has already been sent to this chat: don't send it again. In one short message, say it's "
    "attached and what made it (transcribed_by). Don't summarise, translate, quote or save copies of it "
    "unless they ask."
)
NEXT_PAPAREO = (
    "The file has already been sent to this chat: don't send it again (no ovenmitt_vault_send). In one "
    "short message, say it's attached and that it's a Papa Reo transcript. The file's footer asks them to "
    "follow the Kaitiakitanga License, so they don't need an explicit reminder; but if they ask about the "
    "licence, or look about to break its terms, explain it and give the link as a plain URL, "
    "https://papareo.io/kaitiakitanga. Don't repeat, summarise or save the text unless they ask."
)


NEXT_PAPAREO_DIRECT = (
    "The file has already been sent to this chat: don't send it again. In one short message, say it's "
    "attached and that it's a Papa Reo transcript. The file's footer asks them to follow the Kaitiakitanga "
    "License, so they don't need an explicit reminder; but if they ask about the licence, or look about to "
    "break its terms, explain it and give the link as a plain URL, https://papareo.io/kaitiakitanga. Don't "
    "repeat, summarise or save the text unless they ask."
)


def direct_ok(deps: Deps) -> bool:
    """No vault, but the operator allows attaching Papa Reo files directly on
    this chat's platform, and the key is there."""
    if not deps.direct_platforms or not deps.papareo.configured:
        return False
    try:
        platform = deps.platform()
    except Exception:  # noqa: BLE001
        return False
    return bool(platform) and platform in deps.direct_platforms


def _deliver_direct(
    body: bytes, filename: str, reasons: tuple[str, ...], shared: str | None, deps: Deps
) -> dict[str, Any]:
    """Attach the Papa Reo transcript to this chat with the chat's own
    adapter; nothing is kept. The text goes to the model only when shared."""
    try:
        deps.send_local(filename, body.decode("utf-8"), _caption(PAPAREO_SOURCE))
    except Exception as e:  # noqa: BLE001
        # Only the type: an adapter's message could echo what it was sent.
        logger.warning("papareo_transcribe: couldn't attach the Papa Reo transcript: %s", type(e).__name__)
        return {
            "sent": False,
            "engine": "papareo",
            "delivery": "direct",
            "filename": filename,
            "papareo_charged": True,
            "note": (
                "Papa Reo transcribed it (that was paid for), but the file couldn't be attached to this chat, "
                "and no copy was kept. A group send can reach some people and still report a failure, so ask "
                "whether it arrived anyway. Ask before running Papa Reo again."
            ),
            **_shared(shared),
        }
    finally:
        del body
    logger.info("papareo_transcribe: Papa Reo transcript attached directly (%s)", ",".join(reasons))
    return {
        "sent": True,
        "engine": "papareo",
        "delivery": "direct",
        "transcribed_by": PAPAREO_SOURCE,
        "filename": filename,
        **_shared(shared),
        "next": NEXT_PAPAREO_DIRECT,
    }


def _shared(text: str | None) -> dict[str, Any]:
    """The Papa Reo text, for an agent declared Kaitiakitanga-compliant."""
    return {} if text is None else {"text": text, "kaitiakitanga_compliant": True}


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
    source = local_source(local)
    out: dict[str, Any] = {
        "engine": "local",
        "transcribed_by": source,
        "filename": filename,
        "text": local.text,
        "next": NEXT_LOCAL,
    }
    if note:
        out["note"] = note
    try:
        # The file says where it came from, as Papa Reo's footer does.
        deps.send_local(filename, f"{local.text.strip()}\n\nTranscribed by {source}.\n", _caption(source, note))
        out["sent"] = True
    except Exception as e:  # noqa: BLE001
        logger.warning("papareo_transcribe: couldn't attach the local transcript: %s", e)
        out["sent"] = False
        out["note"] = ((note + " ") if note else "") + "The file couldn't be attached; the text is above."
    return out


def _safe(is_audio: Callable[[str], bool], path: str) -> bool:
    try:
        return bool(is_audio(path))
    except Exception:  # noqa: BLE001 - can't tell: treat as not audio
        return False


def _probe(deps: Deps, path: str) -> float | None:
    try:
        return deps.duration(path) if deps.duration else None
    except Exception:  # noqa: BLE001 - unknown length: the shortest timeout applies
        return None


def _fallback(
    local: LocalResult | None, filename: str, note: str, deps: Deps, paid: bool = False
) -> dict[str, Any]:
    """Papa Reo didn't deliver. With a local transcript in hand (engine="auto",
    later), send that; otherwise say so and leave the choice to the person."""
    if local is None:
        out: dict[str, Any] = {
            "sent": False,
            "engine": "none",
            "note": note + " No local transcript was made; ask whether they want one (engine='local').",
        }
        if paid:
            out["papareo_charged"] = True
        return out
    out = _deliver_local(local, filename, note + " This is the local transcript instead.", deps)
    if paid:
        out["papareo_charged"] = True
    return out
