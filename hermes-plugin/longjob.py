"""Recordings too long for one go: cut at quiet moments, separate speakers across
the parts, transcribe each part, and send one file.

It runs in the background, so nothing depends on one chat turn lasting for hours.
Speakers are separated before anything is paid for, as they are for short
recordings. Voice vectors stay in memory for that stage and are then dropped.

On disk, in a private folder until the job ends: the recording, its cut points,
Papa Reo task ids, and who spoke when. Transcript text and voice vectors never go
there: a resumed job downloads the text again by task id, which costs nothing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .papareo import PapareoError, TaskFailed, Timed, with_footer
from .speakers import AUDIO_DEMUXERS, Cue, DiarizeError, Diarized, NoSpeech, Turn

logger = logging.getLogger(__name__)

LONG_AFTER_S = 60 * 60.0
CHUNK_S = 30 * 60.0
CUT_WINDOW_S = 3 * 60.0
MAX_JOB_S = 48 * 60 * 60.0
MAX_BYTES = 4 * 1024 * 1024 * 1024
KEEP_S = 7 * 24 * 60 * 60.0
# Cosine similarity for "the same person as in an earlier part".
LINK_THRESHOLD = 0.45
SILENCE_FILTER = "silencedetect=noise=-35dB:d=0.4"
TOOL_TIMEOUT_S = 60 * 60.0
HOME_ENV = "PAPAREO_JOBS_HOME"
JOB_FILE = "job.json"
RESUME_HINT = "Ask me to resume the transcript and I'll carry on. Parts already done aren't paid for again."

_SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)")
_busy = threading.Lock()


def jobs_home() -> Path:
    configured = os.environ.get(HOME_ENV)
    if configured:
        return Path(configured)
    return Path(os.environ.get("HERMES_HOME") or "/opt/data") / "papareo-jobs"


@dataclass
class Job:
    id: str
    platform: str
    chat_id: str
    filename: str
    source: str
    speakers: bool
    num_speakers: int | None
    parts: list[tuple[float, float]]
    tasks: list[str | None]
    turns: list[tuple[float, float, int]] | None = None
    names: dict[str, str] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)


@dataclass
class Env:
    """What a running job needs from the outside. Bound while the turn still has its
    chat, because the job outlives the turn."""

    papareo: Any
    diarize: Callable[..., Diarized] | None
    attach: Callable[[str, str, str | None], None]
    say: Callable[[str], None]
    samples: Mapping[str, str] = field(default_factory=dict)
    samples_used: Callable[[], None] | None = None
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run


class Stopped(Exception):
    """The job can't go on for now. The message is for the person, and it never
    carries transcript text."""


def silences(source: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[float]:
    """The middle of each quiet moment, in seconds. Empty if they can't be found:
    the parts are then cut at fixed times, which may split a word."""
    try:
        done = run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-protocol_whitelist", "file", "-format_whitelist",
             AUDIO_DEMUXERS, "-i", source, "-vn", "-af", SILENCE_FILTER, "-f", "null", "-"],
            capture_output=True, timeout=TOOL_TIMEOUT_S, env=_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    text = done.stderr.decode("utf-8", "replace") if isinstance(done.stderr, bytes) else str(done.stderr or "")
    return [float(end) - float(length) / 2 for end, length in _SILENCE_END.findall(text)]


def plan_parts(duration: float, quiet: Sequence[float]) -> list[tuple[float, float]]:
    """Back-to-back parts covering the whole recording, about CHUNK_S each, cut at
    the quiet moment nearest each target. No gaps, and no part left tiny at the end."""
    parts: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > CHUNK_S + CUT_WINDOW_S:
        target = start + CHUNK_S
        near = [q for q in quiet if abs(q - target) <= CUT_WINDOW_S]
        cut = min(near, key=lambda q: abs(q - target)) if near else target
        parts.append((start, cut))
        start = cut
    parts.append((start, duration))
    return parts


class VoiceLinker:
    """Decides which voice in a part is which person so far. Kept in memory for
    one separation stage only."""

    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._people: list[tuple[list[float], int]] = []

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        top = sum(x * y for x, y in zip(a, b))
        size = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return top / size if size else 0.0

    def person(self, vector: Sequence[float]) -> int:
        scores = [self._cosine(vector, total) for total, _ in self._people]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        full = self._limit is not None and len(self._people) >= self._limit
        if best is None or (scores[best] < LINK_THRESHOLD and not full):
            self._people.append((list(vector), 1))
            return len(self._people) - 1
        total, count = self._people[best]
        self._people[best] = ([x + y for x, y in zip(total, vector)], count + 1)
        return best


def _env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(jobs_home())}


def _folder(job: Job) -> Path:
    return jobs_home() / job.id


def save(job: Job) -> None:
    """Written whole and then swapped in: a crash must not lose a paid task id."""
    target = _folder(job) / JOB_FILE
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(job)), encoding="utf-8")
    temporary.replace(target)


def load(folder: Path) -> Job | None:
    try:
        raw = json.loads((folder / JOB_FILE).read_text(encoding="utf-8"))
        raw["parts"] = [tuple(p) for p in raw["parts"]]
        if raw.get("turns") is not None:
            raw["turns"] = [tuple(t) for t in raw["turns"]]
        return Job(**raw)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def forget(job: Job) -> None:
    shutil.rmtree(_folder(job), ignore_errors=True)


def sweep(now: Callable[[], float] = time.time) -> None:
    """Abandoned jobs don't keep someone's recording for ever."""
    home = jobs_home()
    if not home.is_dir():
        return
    for folder in home.iterdir():
        try:
            if folder.is_dir() and now() - folder.stat().st_mtime > KEEP_S:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            continue


def latest_for(platform: str, chat_id: str) -> Job | None:
    home = jobs_home()
    if not home.is_dir():
        return None
    folders = sorted(home.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    for folder in folders:
        job = load(folder)
        if job and job.platform == platform and job.chat_id == chat_id:
            return job
    return None


def _cut(job: Job, index: int, run: Callable[..., subprocess.CompletedProcess]) -> Path:
    """One part as 16 kHz mono FLAC: lossless at the rate the recogniser reads."""
    start, end = job.parts[index]
    dest = _folder(job) / f"part-{index}.flac"
    if dest.is_file():
        return dest
    temporary = dest.with_suffix(".tmp.flac")
    try:
        done = run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file", "-format_whitelist",
             AUDIO_DEMUXERS, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i",
             str(_folder(job) / job.source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(temporary)],
            capture_output=True, timeout=TOOL_TIMEOUT_S, env=_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Stopped(f"I couldn't cut part {index + 1} of the recording. {RESUME_HINT}") from e
    if done.returncode != 0:
        raise Stopped(f"I couldn't cut part {index + 1} of the recording. {RESUME_HINT}")
    temporary.replace(dest)
    return dest


def _separate(job: Job, env: Env) -> None:
    """Speakers across every part, before anything is paid for."""
    if env.diarize is None:
        raise DiarizeError("speaker separation is not installed on this agent")
    linker = VoiceLinker(job.num_speakers)
    turns: list[tuple[float, float, int]] = []
    names: dict[str, str] = {}
    try:
        for index, (start, end) in enumerate(job.parts):
            try:
                found = env.diarize(str(_cut(job, index, env.run)), job.num_speakers, env.samples, end - start, True)
            except NoSpeech:
                continue
            people = {local: linker.person(vector) for local, vector in found.voices.items()}
            for turn in found.turns:
                # Too little of this voice to compare: its time goes to its neighbours.
                if turn.speaker in people:
                    turns.append((turn.start + start, turn.end + start, people[turn.speaker]))
            for local, name in found.names.items():
                if local in people:
                    names.setdefault(str(people[local]), name)
    finally:
        del linker
    taken = set(names.values())
    job.turns = turns
    job.names = names
    job.unmatched = [name for name in env.samples if name not in taken]
    save(job)


def _transcribe(job: Job, env: Env) -> list[Timed]:
    """Part by part, resuming from any task id already paid for. The text stays in
    memory. Each task id is saved before waiting on it."""
    results: list[Timed] = []
    for index in range(len(job.parts)):
        total = len(job.parts)
        if job.tasks[index] is None:
            part = _cut(job, index, env.run)
            try:
                job.tasks[index] = env.papareo.start_task(str(part))
            except PapareoError as e:
                raise Stopped(f"Papa Reo wouldn't take part {index + 1} of {total}. {RESUME_HINT}") from e
            save(job)
        try:
            results.append(env.papareo.finish_task(job.tasks[index]))
        except TaskFailed as e:
            job.tasks[index] = None
            save(job)
            raise Stopped(f"Papa Reo couldn't do part {index + 1} of {total}. {RESUME_HINT}") from e
        except PapareoError as e:
            raise Stopped(f"I lost touch with Papa Reo on part {index + 1} of {total}. {RESUME_HINT}") from e
        (_folder(job) / f"part-{index}.flac").unlink(missing_ok=True)
    return results


def _join(job: Job, results: Sequence[Timed]) -> tuple[str, tuple[Cue, ...], tuple[Cue, ...]]:
    """One timeline: every part's timings moved along by where the part starts."""
    words: list[str] = []
    cues: list[Cue] = []
    hints: list[Cue] = []
    for (start, end), result in zip(job.parts, results):
        if not result.text.strip():
            continue
        words.append(result.text.strip())
        if result.cues:
            cues.extend(Cue(c.start + start, c.end + start, c.text) for c in result.cues)
        else:
            cues.append(Cue(start, end, result.text.strip()))
        hints.extend(Cue(c.start + start, c.end + start, c.text) for c in result.hints)
    return " ".join(words), tuple(cues), tuple(hints)


def run_job(job: Job, env: Env, lay_out: Callable[..., tuple[str, Any, bool]], caption: str) -> None:
    """The whole job. Tells the person how it ended, and logs counts only."""
    try:
        if job.speakers and job.turns is None:
            try:
                _separate(job, env)
            except DiarizeError as e:
                logger.warning("papareo_transcribe: long job separation failed: %s", e)
                if env.samples_used and e.clips_used:
                    env.samples_used()
                forget(job)
                env.say(f"Speakers couldn't be separated ({e}), so Papa Reo wasn't used and nothing was spent.")
                return
            if env.samples_used:
                env.samples_used()
        results = _transcribe(job, env)
        text, cues, hints = _join(job, results)
        del results
        if not text:
            forget(job)
            env.say("Papa Reo found no words in that recording.")
            return
        diarized = Diarized(
            tuple(Turn(*t) for t in job.turns or ()),
            {int(k): v for k, v in job.names.items()},
            tuple(job.unmatched),
        ) if job.speakers and job.turns else None
        laid_out, _, _ = lay_out(cues, text, diarized, hints)
        del text, cues, hints
        try:
            env.attach(job.filename, with_footer(laid_out).decode("utf-8"), caption)
        except Exception as e:  # noqa: BLE001 - only the type: it could echo what it was sent
            logger.warning("papareo_transcribe: long job couldn't attach: %s", type(e).__name__)
            env.say(
                "The transcript is done, but sending the file reported a problem. A group send can reach "
                f"people and still report that, so check if it arrived. If it didn't: {RESUME_HINT}"
            )
            return
        logger.info("papareo_transcribe: long job sent (%d parts)", len(job.parts))
        forget(job)
    except Stopped as e:
        logger.warning("papareo_transcribe: long job stopped after %d parts", sum(t is not None for t in job.tasks))
        env.say(str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("papareo_transcribe: long job failed: %s", type(e).__name__)
        env.say(f"The transcript stopped because of an unexpected problem. {RESUME_HINT}")


BUSY = "another long transcript is running on this agent. Ask them to try again when it's finished"
NEXT_STARTED = (
    "It's now running in the background, and the file will be posted in this chat when it's done, which may "
    "take hours. Tell them that in one short message. Don't call this again, and don't send anything else."
)


def refusal(
    path: str, duration: float, speakers: bool, count: int | None, diarize: Any, direct: bool
) -> str | None:
    """Why this long recording can't start, before anything is spent."""
    if duration > MAX_JOB_S:
        return f"that recording is over {int(MAX_JOB_S // 3600)} hours, which is more than this can take in one job"
    if Path(path).stat().st_size > MAX_BYTES:
        return "that recording is over 4GB, which is more than this can take"
    if not direct:
        return "long recordings can only be sent back where Papa Reo files may be attached directly"
    if speakers and diarize is None:
        return "speaker separation is not installed on this agent"
    if speakers and not count:
        return (
            "this recording is too long to guess how many people there are. Ask how many people speak "
            "in it, then call again with num_speakers"
        )
    return None


def confirmation(duration: float) -> dict[str, Any]:
    hours = round(duration / 3600, 1)
    parts = len(plan_parts(duration, ()))
    return {
        "sent": False,
        "needs_agreement": True,
        "hours": hours,
        "parts": parts,
        "note": (
            f"Nothing has been spent. This recording is {hours} hours, so it would run in the background in about "
            f"{parts} parts, and Papa Reo charges for all of it. Tell them that, and that speaker numbers across "
            "parts are matched by machine and may slip. The recording stays on this machine until the job ends. "
            "If they agree, call again the same way with long_job_ok=true."
        ),
    }


def start(work: Callable[[], None], spawn: Callable[[Callable[[], None]], None] | None = None) -> bool:
    """Run work in the background, one job at a time. False if one is running."""
    if not _busy.acquire(blocking=False):
        return False

    def guarded() -> None:
        try:
            work()
        finally:
            _busy.release()

    try:
        if spawn:
            spawn(guarded)
        else:
            threading.Thread(target=guarded, name="papareo-long-job", daemon=True).start()
    except Exception:
        _busy.release()
        raise
    return True


def is_busy() -> bool:
    return _busy.locked()


def new_job(
    path: str, duration: float, platform: str, chat_id: str, filename: str, speakers: bool,
    num_speakers: int | None, take: Callable[[str, str], Any], run: Callable[..., subprocess.CompletedProcess],
) -> Job:
    sweep()
    home = jobs_home()
    home.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    folder = home / job_id
    folder.mkdir(mode=0o700)
    try:
        source = "source" + Path(path).suffix[:8]
        take(path, str(folder / source))
        parts = plan_parts(duration, silences(str(folder / source), run))
        job = Job(job_id, platform, chat_id, filename, source, speakers, num_speakers, parts, [None] * len(parts))
        save(job)
        return job
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
