"""Recordings too long for one go: cut at quiet moments, separate speakers across
the parts, transcribe each part, and send one file.

It runs in the background, so nothing depends on one chat turn lasting for hours.
Speakers are separated before anything is paid for, as they are for short
recordings. Voice vectors stay in memory for that stage and are then dropped.

The recording is first turned into one 16 kHz mono WAV, so every cut lands on an
exact sample whatever the original format was, and its real length is known.

On disk, in a private folder until the job ends: that WAV, its cut points, Papa
Reo task ids, and who spoke when. Transcript text and voice vectors never go
there: a resumed job downloads the text again by task id, which costs nothing.
"""

from __future__ import annotations

import fcntl
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
import wave
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
# A WAV stops at 4GB, which at this rate is a little over 37 hours.
MAX_JOB_S = 36 * 60 * 60.0
MAX_BYTES = 4 * 1024 * 1024 * 1024
KEEP_S = 7 * 24 * 60 * 60.0
# Cosine similarity for "the same person as in an earlier part".
LINK_THRESHOLD = 0.45
SILENCE_FILTER = "silencedetect=noise=-35dB:d=0.4"
TOOL_TIMEOUT_S = 2 * 60 * 60.0
SOURCE = "source.wav"
LOCK_FILE = ".running.lock"
HOME_ENV = "PAPAREO_JOBS_HOME"
JOB_FILE = "job.json"
RECOVER_TRIES = 60
RECOVER_WAIT_S = 10.0
RESUME_HINT = "Ask me to resume the transcript and I'll carry on. Parts already done aren't paid for again."

MEASURE_TIMEOUT_S = 20 * 60.0
_DECODED = re.compile(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)")


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
    # "running" comes back by itself after a restart; "stopped" waits to be asked.
    state: str = "running"
    thread_id: str = ""
    profile: str = ""
    had_samples: bool = False
    # What the header claimed. The parts come from the decoded length, once known.
    expected_s: float = 0.0


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


def measure(path: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> float | None:
    """A recording's length by decoding it, for files whose header doesn't say. None
    if that can't be told. Reads the whole file, so only use it when probing failed."""
    try:
        done = run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-protocol_whitelist", "file", "-format_whitelist",
             AUDIO_DEMUXERS, "-i", path, "-vn", "-f", "null", "-"],
            capture_output=True, timeout=MEASURE_TIMEOUT_S, env=_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = done.stderr.decode("utf-8", "replace") if isinstance(done.stderr, bytes) else str(done.stderr or "")
    times = _DECODED.findall(text)
    if done.returncode != 0 or not times:
        return None
    hours, minutes, seconds = times[-1]
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


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

    def assign(self, voices: Mapping[int, Sequence[float]]) -> dict[int, int]:
        """Which person each of one part's voices is.

        With a known number of people the separation is told that number, so
        every voice in a part is a different person: they are matched to people
        one-to-one, best match first. Voices in one room can sound alike enough
        to pass LINK_THRESHOLD, and without this they all collapsed into one
        person. Without a known number the separation may split one person in
        two, so each voice is linked on its own and may merge.

        A match made only because every place is taken (below LINK_THRESHOLD)
        still gets a label, but doesn't change that person's stored voice.
        """
        if self._limit is None:
            return {local: self.person(vector) for local, vector in voices.items()}
        existing = len(self._people)
        score = {
            (local, person): self._cosine(vector, self._people[person][0])
            for local, vector in voices.items() for person in range(existing)
        }
        out: dict[int, int] = {}
        confident: set[int] = set()
        used: set[int] = set()
        for (local, person), sim in sorted(score.items(), key=lambda kv: kv[1], reverse=True):
            if sim < LINK_THRESHOLD:
                break
            if local in out or person in used:
                continue
            out[local] = person
            used.add(person)
            confident.add(local)
        left = [local for local in voices if local not in out]
        room = max(0, self._limit - existing)
        free = [p for p in range(existing) if p not in used]

        def out_local(person: int) -> int:
            return next(l for l, p in out.items() if p == person)

        def best_free(local: int) -> float:
            return max((score[(local, p)] for p in free), default=-1.0)

        # The least alike of the leftover voices become new people, up to the
        # known number. The rest must share a place: closest free person
        # first, or (more voices than people) the closest person at all.
        left.sort(key=best_free, reverse=True)
        n_new = min(room, len(left))
        newcomers = left[len(left) - n_new:] if n_new else []
        forced = left[: len(left) - n_new]
        for local in newcomers:
            self._people.append(([0.0] * len(voices[local]), 0))
            out[local] = len(self._people) - 1
            used.add(out[local])
            confident.add(local)
        for local in forced:
            pool = [p for p in free if p not in used] or list(range(len(self._people)))
            out[local] = max(pool, key=lambda p: self._cosine(
                voices[local], self._people[p][0] if self._people[p][1] else voices[out_local(p)]
            ))
            used.add(out[local])
        for local, person in out.items():
            if local not in confident:
                continue
            total, count = self._people[person]
            self._people[person] = ([x + y for x, y in zip(total, voices[local])], count + 1)
        return out


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
    sweep()
    for job in sorted(_all_jobs(), key=_age):
        if job.platform == platform and job.chat_id == chat_id:
            return job
    return None


def _age(job: Job) -> float:
    try:
        return -(_folder(job) / JOB_FILE).stat().st_mtime
    except OSError:
        return 0.0


def _all_jobs() -> list[Job]:
    try:
        folders = list(jobs_home().iterdir())
    except OSError:
        return []
    return [job for job in (load(folder) for folder in folders) if job]


def _cut(job: Job, index: int, run: Callable[..., subprocess.CompletedProcess]) -> Path:
    """One part as WAV. The samples are copied out of the WAV by position, so parts
    neither lose nor repeat any, with no seeking inside ffmpeg.

    WAV, not FLAC: Papa Reo's /tuhi/transcribe/large accepts the upload of a
    FLAC file and then reports the task as failed, at any length (checked
    2026-11-12 with 1-, 10- and 20-minute slices). WAV and AAC succeed."""
    start, end = job.parts[index]
    dest = _folder(job) / f"part-{index}.wav"
    if dest.is_file():
        return dest
    piece = dest.with_suffix(".tmp.wav")
    last = index == len(job.parts) - 1
    try:
        with wave.open(str(_folder(job) / job.source), "rb") as source:
            rate = source.getframerate()
            first = int(start * rate)
            left = source.getnframes() - first if last else int(end * rate) - first
            source.setpos(min(first, source.getnframes()))
            with wave.open(str(piece), "wb") as out:
                out.setnchannels(source.getnchannels())
                out.setsampwidth(source.getsampwidth())
                out.setframerate(rate)
                while left > 0:
                    block = source.readframes(min(left, rate * 60))
                    if not block:
                        break
                    out.writeframes(block)
                    left -= len(block) // (source.getsampwidth() * source.getnchannels())
        piece.replace(dest)
    except (OSError, EOFError, wave.Error) as e:
        piece.unlink(missing_ok=True)
        raise Stopped(f"I couldn't cut part {index + 1} of the recording. {RESUME_HINT}") from e
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
                part = _cut(job, index, env.run)
                try:
                    found = env.diarize(str(part), job.num_speakers, env.samples, end - start, True)
                finally:
                    # Cutting again is cheap (no ffmpeg); keeping every WAV part
                    # until it is sent would double a long job's disk use.
                    part.unlink(missing_ok=True)
            except NoSpeech:
                continue
            people = linker.assign(found.voices)
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
                raise Stopped(
                    f"Sending part {index + 1} of {total} to Papa Reo failed. Ask me to resume and I'll send it again. "
                    "If Papa Reo had already received it, that part could be charged twice. Other finished parts aren't."
                ) from e
            save(job)
        try:
            results.append(env.papareo.finish_task(job.tasks[index]))
        except TaskFailed as e:
            job.tasks[index] = None
            save(job)
            raise Stopped(
                f"Papa Reo reports that part {index + 1} of {total} failed. Ask me to resume and I'll send it again, "
                "which may be charged. Other finished parts aren't."
            ) from e
        except PapareoError as e:
            raise Stopped(f"I lost touch with Papa Reo on part {index + 1} of {total}. {RESUME_HINT}") from e
        (_folder(job) / f"part-{index}.wav").unlink(missing_ok=True)
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


def _notes(job: Job, env: Env, results: Sequence[Timed]) -> str:
    """What the person should know about this file. Names and counts only."""
    notes = []
    if job.unmatched:
        notes.append(f"Couldn't match these voice clips to anyone: {', '.join(job.unmatched)}.")
    if job.had_samples and not env.samples and not job.names and not job.unmatched:
        notes.append("Voice clips were lost in a restart, so speakers are numbered, not named.")
    silent = sum(1 for r in results if not r.text.strip())
    if silent:
        notes.append(f"{silent} of {len(results)} parts came back with no words, which may just be silence.")
    found = job.parts[-1][1] if job.parts else 0.0
    if job.expected_s - found > max(60.0, job.expected_s * 0.02):
        notes.append(
            f"The recording claimed to be {job.expected_s / 3600:.1f} hours, but only {found / 3600:.1f} hours "
            "could be read, so it may be cut off."
        )
    rough = sum(1 for r in results if r.text.strip() and not r.cues)
    if job.speakers and rough:
        notes.append(f"Papa Reo gave no timings for {rough} part(s), so their speaker labels are rough.")
    return " ".join(notes)


def run_job(job: Job, env: Env, lay_out: Callable[..., tuple[str, Any, bool]], caption: str) -> None:
    """The whole job. Tells the person how it ended, and logs counts only."""
    try:
        job.state = "running"
        save(job)
        if not job.parts:
            try:
                prepare(job, env.run)
            except Stopped as e:
                forget(job)
                env.say(str(e))
                return
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
            if not job.turns:
                forget(job)
                env.say("No speech was found in that recording, so Papa Reo wasn't used and nothing was spent.")
                return
        results = _transcribe(job, env)
        text, cues, hints = _join(job, results)
        notes = _notes(job, env, results)
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
            env.attach(job.filename, with_footer(laid_out).decode("utf-8"), f"{caption} {notes}".strip())
        except Exception as e:  # noqa: BLE001 - only the type: it could echo what it was sent
            logger.warning("papareo_transcribe: long job couldn't attach: %s", type(e).__name__)
            job.state = "stopped"
            save(job)
            env.say(
                "The transcript is done, but sending the file reported a problem. A group send can reach "
                f"people and still report that, so check if it arrived. If it didn't: {RESUME_HINT}"
            )
            return
        logger.info("papareo_transcribe: long job sent (%d parts)", len(job.parts))
        forget(job)
    except Stopped as e:
        logger.warning("papareo_transcribe: long job stopped after %d parts", sum(t is not None for t in job.tasks))
        _mark_stopped(job)
        env.say(str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("papareo_transcribe: long job failed: %s", type(e).__name__)
        _mark_stopped(job)
        env.say(f"The transcript stopped because of an unexpected problem. {RESUME_HINT}")


BUSY = "another long transcript is running on this agent. Ask them to try again when it's finished"
WAITING = (
    "this chat already has a stopped long transcript ({name}), with parts that may already be paid for, and it may "
    "be someone else's in a group. Starting another would pay again if it's the same recording. Ask them: carry on "
    "(call with resume_job=true), or give it up (discard_job=true, and then send this one again)"
)
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


def _take_lock() -> Any | None:
    """One job at a time on this agent. A file lock, because the gateway re-imports
    plugins while running, which would give a module-level lock a second copy. Freed
    by the system if the process dies."""
    home = jobs_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        handle = open(home / LOCK_FILE, "w")  # noqa: SIM115 - held for the whole job
    except OSError:
        logger.warning("papareo_transcribe: the long job folder can't be written")
        return None
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError:
        handle.close()
        return None


def start(work: Callable[[], None], spawn: Callable[[Callable[[], None]], None] | None = None) -> bool:
    """Run work in the background. False if a job is already running."""
    lock = _take_lock()
    if lock is None:
        return False

    def guarded() -> None:
        try:
            work()
        finally:
            lock.close()

    try:
        if spawn:
            spawn(guarded)
        else:
            threading.Thread(target=guarded, name="papareo-long-job", daemon=True).start()
    except Exception:
        lock.close()
        raise
    return True


def _mark_stopped(job: Job) -> None:
    job.state = "stopped"
    try:
        save(job)
    except OSError:
        logger.warning("papareo_transcribe: couldn't save a stopped job")


def is_busy() -> bool:
    lock = _take_lock()
    if lock is None:
        return True
    lock.close()
    return False


def reserve(
    path: str, duration: float, platform: str, chat_id: str, filename: str, speakers: bool, num_speakers: int | None,
    take: Callable[[str, str], Any], thread_id: str = "", profile: str = "", had_samples: bool = False,
) -> Job:
    """A private folder holding the recording, and a job that says whose it is, so a
    restart before it's prepared can still carry on. Done during the turn, while the
    recording still exists."""
    sweep()
    home = jobs_home()
    home.mkdir(parents=True, exist_ok=True)
    job = Job(
        uuid.uuid4().hex, platform, chat_id, filename, "original" + Path(path).suffix[:8], speakers, num_speakers,
        [], [], thread_id=thread_id, profile=profile, had_samples=had_samples, expected_s=duration,
    )
    folder = _folder(job)
    folder.mkdir(mode=0o700)
    try:
        take(path, str(folder / job.source))
        save(job)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return job


def prepare(job: Job, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
    """Decode to one WAV, find its true length and quiet moments, and plan the parts.
    Raises Stopped if the recording can't be read or holds nothing."""
    unreadable = "I couldn't read that recording, so Papa Reo wasn't used and nothing was spent."
    folder = _folder(job)
    original = folder / job.source
    source = folder / SOURCE
    try:
        done = run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file", "-format_whitelist",
             AUDIO_DEMUXERS, "-i", str(original), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
             "-f", "wav", str(source)],
            capture_output=True, timeout=TOOL_TIMEOUT_S, env=_env(),
        )
        if done.returncode != 0:
            raise Stopped(unreadable)
        with wave.open(str(source), "rb") as decoded:
            frames, rate = decoded.getnframes(), decoded.getframerate()
    except (OSError, EOFError, wave.Error, subprocess.TimeoutExpired) as e:
        raise Stopped(unreadable) from e
    actual = frames / rate if rate else 0.0
    if actual <= 0:
        raise Stopped("There's no audio in that recording, so Papa Reo wasn't used and nothing was spent.")
    if actual > MAX_JOB_S:
        raise Stopped(f"That recording is over {int(MAX_JOB_S // 3600)} hours, so nothing was spent.")
    job.parts = plan_parts(actual, silences(str(source), run))
    job.tasks = [None] * len(job.parts)
    job.source = SOURCE
    save(job)
    original.unlink(missing_ok=True)


def running_jobs() -> list[Job]:
    """Jobs that a restart interrupted."""
    return [job for job in _all_jobs() if job.state == "running"]


def recover(
    bind: Callable[[Job], tuple[Callable[..., None], Callable[[str], None]]],
    work: Callable[[Job, Callable[..., None], Callable[[str], None]], None],
    allowed: Callable[[Job], bool],
    spawn: Callable[[Callable[[], None]], None] | None = None,
    wait: Callable[[float], None] = time.sleep,
) -> None:
    """Carry on after a restart: the person was told the file would come. A plugin
    reload also lands here, so a job already running is left alone, and each job is
    read again once the lock is held. Only a process with a live chat adapter gets
    further than bind, which a passing CLI command doesn't."""
    sweep()
    if not running_jobs() or is_busy():
        return

    def go() -> None:
        while True:
            jobs = running_jobs()
            if not jobs:
                return
            senders = None
            for _ in range(RECOVER_TRIES):
                try:
                    senders = bind(jobs[0])
                    break
                except Exception:  # noqa: BLE001 - the gateway isn't up yet, or never will be
                    wait(RECOVER_WAIT_S)
            if senders is None:
                return
            lock = _take_lock()
            if lock is None:
                return
            try:
                job = load(_folder(jobs[0]))
                if job is None or job.state != "running":
                    continue
                attach, say = senders
                if allowed(job):
                    say("The agent restarted, so I'm carrying on with your transcript. Parts already done aren't paid for again.")
                    work(job, attach, say)
                else:
                    _mark_stopped(job)
                    say("I can't carry on with your transcript here any more, so it has stopped. Nothing more was spent.")
            finally:
                lock.close()
            # Never round again on the same job, whatever just happened to it.
            after = load(_folder(job))
            if after and after.state == "running":
                return

    if spawn:
        spawn(go)
    else:
        threading.Thread(target=go, name="papareo-recover", daemon=True).start()
