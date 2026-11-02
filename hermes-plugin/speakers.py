"""Who spoke when, and tidy paragraphs. No model ever reads the transcript here.

Both jobs are plain arithmetic on timings, so they are safe for Papa Reo text:
the text is laid out in memory and stays out of logs and model context.

Speaker separation runs on this machine, in a separate process and its own
environment (install_speakers.sh), so the agent image needs nothing new
and nothing it holds is inherited. Audio goes to a private temp dir that is
removed afterwards. Voice samples become numbers in the worker's memory for the
length of one call: no voiceprint is ever written anywhere.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

HOME_ENV = "PAPAREO_SPEAKERS_HOME"
DEFAULT_HOME = "/opt/data/papareo-speakers"
PARAGRAPH_GAP_S = 1.2
SOFT_PARAGRAPH_CHARS = 400
HARD_PARAGRAPH_CHARS = 700
NEAREST_TURN_S = 1.5
MIN_PIECE_S = 0.7
HINT_NEAR_S = 1.0
# The only containers ffmpeg may open for us. A playlist (HLS, concat) would make
# it read other files on this machine on behalf of whoever wrote the playlist.
AUDIO_DEMUXERS = (
    "mov,mp4,m4a,3gp,3g2,mj2,mp3,wav,matroska,webm,ogg,aac,flac,amr,amrnb,amrwb,caf,aiff,mpeg,mpegts,asf,au,wv,ac3"
)
MIN_TIMEOUT_S = 300.0
LABEL_NOTE = "Speaker labels were added automatically and may be wrong."

_SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d+)\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d+)")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int


@dataclass(frozen=True)
class Diarized:
    turns: tuple[Turn, ...]
    # Speaker number -> the name given with a matching voice sample.
    names: Mapping[int, str] = field(default_factory=dict)
    # Names whose sample matched nobody in the recording.
    unmatched: tuple[str, ...] = ()


class DiarizeError(Exception):
    """Speaker separation could not run. Messages never carry audio or text."""

    def __init__(self, message: str, clips_used: bool = False) -> None:
        super().__init__(message)
        self.clips_used = clips_used  # the worker had the voice clips


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    # Papa Reo doesn't pad milliseconds: ",35" is 35 ms (its JSON says 0.035),
    # not 350. So the digits are a count of milliseconds, never a fraction.
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(body: str) -> list[Cue]:
    """Every line after a timecode belongs to that cue until the next timecode, so
    a stray blank line or a missing one loses no words."""
    lines = body.replace("\r\n", "\n").split("\n")
    marks = [(i, m) for i, m in ((i, _SRT_TIME.search(line)) for i, line in enumerate(lines)) if m]
    cues: list[Cue] = []
    for index, (at, match) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(lines)
        text = [line.strip() for line in lines[at + 1 : stop] if line.strip()]
        # The number of the next cue, not part of this one.
        if index + 1 < len(marks) and text and text[-1].isdigit():
            text.pop()
        joined = " ".join(text)
        if not joined:
            continue
        g = match.groups()
        try:
            start, end = _seconds(*g[:4]), _seconds(*g[4:])
        except (ValueError, OverflowError):  # an absurd number is no cue at all
            continue
        cues.append(Cue(start, max(start, end), joined))
    return cues


def parse_segments(body: str) -> tuple[Cue, ...]:
    """Papa Reo's JSON download: the same words as its plain text, with timings.
    (Its SRT has finer timings but changes and drops words, so the SRT only ever
    guides where to split, and its text is never delivered.)"""
    try:
        cues = []
        for segment in json.loads(body):
            text = segment["text"].strip()
            start, end = float(segment["start_time"]), float(segment["end_time"])
            if not (math.isfinite(start) and math.isfinite(end)):
                return ()
            if text:
                cues.append(Cue(start, max(start, end), text))
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
        return ()
    return tuple(cues)


def same_words(cues: Sequence[Cue], text: str) -> bool:
    """Whether the cues hold exactly the transcript. Losing paid words is worse
    than losing the layout."""
    return [w for c in cues for w in c.text.split()] == text.split()


def flatten(turns: Sequence[Turn]) -> tuple[Turn, ...]:
    """A timeline with no overlaps. Where two people overlap, whoever started later
    has the floor: otherwise someone interrupting inside a long turn would vanish."""
    points = sorted({p for t in turns for p in (t.start, t.end)})
    ordered = sorted(turns, key=lambda t: t.start)
    out: list[Turn] = []
    active: list[Turn] = []
    next_turn = 0
    for start, end in zip(points, points[1:]):
        while next_turn < len(ordered) and ordered[next_turn].start <= start:
            active.append(ordered[next_turn])
            next_turn += 1
        active = [t for t in active if t.end > start]
        if not active:
            continue
        speaker = active[-1].speaker
        if out and out[-1].speaker == speaker and out[-1].end == start:
            out[-1] = Turn(out[-1].start, end, speaker)
        else:
            out.append(Turn(start, end, speaker))
    return tuple(out)


def _pieces(cue: Cue, turns: Sequence[Turn]) -> list[Turn]:
    """The speakers inside one cue, in order, ignoring anything too short to place."""
    merged: list[Turn] = []
    for t in turns:
        start, end = max(cue.start, t.start), min(cue.end, t.end)
        if end > start:
            merged.append(Turn(start, end, t.speaker))
    kept: list[Turn] = []
    for piece in (p for p in merged if p.end - p.start >= MIN_PIECE_S):
        if kept and kept[-1].speaker == piece.speaker:
            kept[-1] = Turn(kept[-1].start, piece.end, piece.speaker)
        else:
            kept.append(piece)
    return kept


def _share_before(cue: Cue, at: float, hints: Sequence[Cue]) -> float:
    """How much of a cue's speech comes before a moment. A hint only helps if one
    of its edges falls near that moment; otherwise assume a steady pace."""
    inside = [h for h in hints if cue.start <= (h.start + h.end) / 2 <= cue.end]
    total = sum(len(h.text.split()) for h in inside)
    if total and any(min(abs(h.start - at), abs(h.end - at)) <= HINT_NEAR_S for h in inside):
        return sum(len(h.text.split()) for h in inside if (h.start + h.end) / 2 < at) / total
    span = cue.end - cue.start
    return (at - cue.start) / span if span > 0 else 0.0


def split_at_turns(cues: Sequence[Cue], turns: Sequence[Turn], hints: Sequence[Cue] = ()) -> tuple[Cue, ...]:
    """Split any cue that holds two speakers, so each gets their own words. Every
    word survives, in order, and everyone placed in a cue gets at least one word,
    as far as the words go. Where exactly a split falls is an estimate."""
    flat = flatten(turns)
    out: list[Cue] = []
    for cue in cues:
        pieces = _pieces(cue, flat)
        words = cue.text.split()
        if len(pieces) > len(words):
            longest = sorted(pieces, key=lambda p: p.end - p.start, reverse=True)[: len(words)]
            pieces = [p for p in pieces if p in longest]
        if len(pieces) < 2:
            out.append(cue)
            continue
        edge = 0
        for index, piece in enumerate(pieces):
            if index == len(pieces) - 1:
                stop = len(words)
            else:
                wanted = round(_share_before(cue, pieces[index + 1].start, hints) * len(words))
                stop = max(edge + 1, min(wanted, len(words) - (len(pieces) - index - 1)))
            out.append(Cue(piece.start, piece.end, " ".join(words[edge:stop])))
            edge = stop
    return tuple(out)


def _speaker_for(cue: Cue, turns: Sequence[Turn]) -> int | None:
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(cue.end, t.end) - max(cue.start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
    if best is not None:
        return best
    mid = (cue.start + cue.end) / 2
    near = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)), default=None)
    if near and min(abs(mid - near.start), abs(mid - near.end)) <= NEAREST_TURN_S:
        return near.speaker
    return None


def _clock(seconds: float) -> str:
    whole = int(seconds)
    h, m, s = whole // 3600, whole % 3600 // 60, whole % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _next_label(labels: Mapping[int, str], names: Mapping[int, str]) -> str:
    taken = {*labels.values(), *names.values()}
    number = 1
    while f"Speaker {number}" in taken:
        number += 1
    return f"Speaker {number}"


def render(cues: Sequence[Cue], diarized: Diarized | None = None) -> str:
    """Cues as readable text: paragraphs, and speakers when they are known."""
    turns = flatten(diarized.turns) if diarized else ()
    names = diarized.names if diarized else {}
    labels: dict[int, str] = {}
    current: int | None = None
    started = False
    blocks: list[str] = []
    paragraph: list[str] = []
    label: str | None = None
    length = 0
    previous_end: float | None = None

    def flush() -> None:
        nonlocal paragraph, length, label
        if paragraph:
            text = " ".join(paragraph)
            blocks.append(f"{label}\n{text}" if label else text)
            label = None
        paragraph, length = [], 0

    for cue in cues:
        speaker = _speaker_for(cue, turns) if turns else None
        if speaker is None:
            speaker = current
        changed = bool(turns) and (not started or speaker != current)
        gap = previous_end is not None and cue.start - previous_end >= PARAGRAPH_GAP_S
        full = length >= HARD_PARAGRAPH_CHARS or (
            length >= SOFT_PARAGRAPH_CHARS and paragraph[-1].endswith((".", "!", "?"))
        )
        if changed or gap or full:
            flush()
        if changed:
            if speaker is None:
                name = "Unknown speaker"
            else:
                if speaker not in labels:
                    labels[speaker] = names.get(speaker) or _next_label(labels, names)
                name = labels[speaker]
            label = f"[{_clock(cue.start)}] {name}"
        current, started = speaker, True
        paragraph.append(cue.text)
        length += len(cue.text) + 1
        previous_end = cue.end
    flush()
    body = "\n\n".join(blocks)
    return f"{LABEL_NOTE}\n\n{body}" if turns else body


def heard(cues: Sequence[Cue], diarized: Diarized) -> set[int]:
    """The speakers who end up with words in this transcript."""
    flat = flatten(diarized.turns)
    return {s for s in (_speaker_for(c, flat) for c in cues) if s is not None}


def speaker_count(cues: Sequence[Cue], diarized: Diarized) -> int:
    return len(heard(cues, diarized))


class Diarizer:
    """Runs speakers_worker.py in its own environment."""

    def __init__(self, home: str | None = None, run: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self._home = Path(home or os.environ.get(HOME_ENV) or DEFAULT_HOME)
        self._run = run

    @property
    def python(self) -> Path:
        return self._home / "venv" / "bin" / "python"

    @property
    def models(self) -> Path:
        return self._home / "models"

    def why_not(self) -> str | None:
        if not self.python.is_file() or not (self.models / "seg.onnx").is_file() or not (self.models / "emb.onnx").is_file():
            return "speaker separation is not installed on this agent (install_speakers.sh)"
        return None

    def _decode(self, source: str, dest: Path, timeout: float) -> None:
        done = self._run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file", "-format_whitelist", AUDIO_DEMUXERS, "-i", source, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(dest)],
            capture_output=True, timeout=timeout, env=self._env(dest.parent),
        )
        if done.returncode != 0:
            raise DiarizeError("the recording could not be decoded")

    @staticmethod
    def _env(home: Path) -> dict[str, str]:
        # Built from nothing, so no key held by the gateway reaches the worker.
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "OMP_NUM_THREADS": "2"}

    def __call__(
        self,
        path: str,
        num_speakers: int | None = None,
        samples: Mapping[str, str] | None = None,
        duration: float | None = None,
    ) -> Diarized:
        problem = self.why_not()
        if problem:
            raise DiarizeError(problem)
        timeout = max(MIN_TIMEOUT_S, (duration or 0.0) * 2)
        tmp = Path(tempfile.mkdtemp(prefix="speakers-"))  # mode 0700
        started = False
        try:
            wav = tmp / "recording.wav"
            self._decode(path, wav, timeout)
            sample_wavs: dict[str, str] = {}
            for index, (name, sample) in enumerate((samples or {}).items()):
                target = tmp / f"sample-{index}.wav"
                self._decode(sample, target, MIN_TIMEOUT_S)
                sample_wavs[name] = str(target)
            request = {
                "wav": str(wav),
                "models": str(self.models),
                "num_speakers": num_speakers,
                "samples": sample_wavs,
            }
            started = True
            done = self._run(
                [str(self.python), "-I", str(Path(__file__).with_name("speakers_worker.py"))],
                input=json.dumps(request).encode(),
                capture_output=True,
                timeout=timeout,
                env=self._env(tmp),
                cwd=str(tmp),
            )
        except subprocess.TimeoutExpired as e:
            raise DiarizeError("speaker separation took too long", started) from e
        except OSError as e:
            raise DiarizeError("speaker separation could not start") from e
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if done.returncode != 0:
            raise DiarizeError("speaker separation failed", True)
        try:
            answer = json.loads(done.stdout)
            turns = tuple(Turn(float(t["start"]), float(t["end"]), int(t["speaker"])) for t in answer["turns"])
            names = {int(k): str(v) for k, v in (answer.get("names") or {}).items()}
            unmatched = tuple(str(n) for n in (answer.get("unmatched") or []))
        except (ValueError, KeyError, TypeError) as e:
            raise DiarizeError("speaker separation answered with something unexpected", True) from e
        if not turns:
            raise DiarizeError("no speech was found in the recording", True)
        return Diarized(turns, names, unmatched)
