"""The worker's grouping, with fake voice fingerprints in place of sherpa-onnx.

The fake audio carries who is speaking as its sample values, and the fake
extractor turns that into a voice vector with a little noise, so the tests know
the truth and can check the grouping against it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

RATE = 16000
DIM = 32


def load_worker():
    path = Path(__file__).resolve().parents[1] / "hermes-plugin" / "speakers_worker.py"
    spec = importlib.util.spec_from_file_location("speakers_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_worker()


def voices(count, closeness=0.0, seed=1):
    """Unit voice vectors. closeness > 0 makes them sound alike."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=DIM)
    out = []
    for _ in range(count):
        v = rng.normal(size=DIM) + closeness * shared
        out.append(v / np.linalg.norm(v))
    return out


class FakeExtractor:
    # noise 0.15 puts a window about as close to its speaker's average voice as
    # real fingerprints are (cosine ~0.75).
    def __init__(self, bank, noise=0.15, seed=2):
        self.bank = bank
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    class Stream:
        def __init__(self):
            self.audio = np.zeros(0, dtype=np.float32)

        def accept_waveform(self, rate, audio):
            self.audio = np.concatenate([self.audio, audio])

        def input_finished(self):
            pass

    def create_stream(self):
        return self.Stream()

    def is_ready(self, stream):
        return len(stream.audio) > 0

    def compute(self, stream):
        ids = np.rint(stream.audio * 100).astype(int) - 1
        ids = ids[ids >= 0]
        if not len(ids):
            return list(self.rng.normal(size=DIM))
        who = np.bincount(ids).argmax()
        v = self.bank[who] + self.noise * self.rng.normal(size=DIM)
        return list(v)


def recording(script):
    """script: [(seconds, speaker), ...] back to back. Returns audio and truth."""
    pieces, truth, at = [], [], 0.0
    for seconds, who in script:
        pieces.append(np.full(int(seconds * RATE), (who + 1) / 100, dtype=np.float32))
        truth.append((at, at + seconds, who))
        at += seconds
    return np.concatenate(pieces), truth


def segmenter(truth):
    # sherpa's turns: one per stretch of speech, cut where the voice changes.
    return lambda audio: [(a, b) for a, b, _ in truth]


def who_at(turns, t):
    for turn in turns:
        if turn["start"] <= t < turn["end"]:
            return turn["speaker"]
    return None


def agreement(turns, truth):
    """Share of time each true speaker's label matches the best mapping."""
    from collections import Counter

    pairs = Counter()
    total = 0
    for a, b, who in truth:
        t = a
        while t < b:
            got = who_at(turns, t)
            if got is not None:
                pairs[(who, got)] += 1
            total += 1
            t += 0.5
    best = {}
    for (who, got), n in pairs.most_common():
        if who not in best and got not in best.values():
            best[who] = got
    return sum(n for (who, got), n in pairs.items() if best.get(who) == got) / total


def run(script, bank, noise=0.15, **request):
    audio, truth = recording(script)
    found = worker.separate(request, audio, segmenter(truth), FakeExtractor(bank, noise=noise))
    return found, truth


CONVERSATION = [(12, 0), (6, 1), (9, 0), (14, 2), (5, 1), (11, 2), (8, 0), (10, 1), (7, 2), (9, 0)] * 3


def test_it_finds_how_many_people_without_being_told():
    found, truth = run(CONVERSATION, voices(3))
    assert len({t["speaker"] for t in found["turns"]}) == 3
    assert agreement(found["turns"], truth) > 0.95


def test_a_count_is_a_ceiling_and_never_splits_one_person():
    # Told six, but only three talk: forcing six split one person into several.
    found, truth = run(CONVERSATION, voices(3), num_speakers=6)
    assert len({t["speaker"] for t in found["turns"]}) == 3
    assert agreement(found["turns"], truth) > 0.95


def test_a_count_still_caps_the_guess():
    found, _ = run(CONVERSATION, voices(3), num_speakers=2)
    assert len({t["speaker"] for t in found["turns"]}) <= 2


def test_moments_pin_names_to_the_right_people():
    found, truth = run(CONVERSATION, voices(3), anchors={"Moana": [[13, 17]], "Aroha": [[28, 34]]})
    names = {int(k): v for k, v in found["names"].items()}
    assert names[who_at(found["turns"], 14)] == "Moana"
    assert names[who_at(found["turns"], 30)] == "Aroha"
    assert found["unmatched"] == []


def test_moments_keep_alike_voices_apart():
    # Two voices about as alike as two people at one table: on their own they
    # merge, and a moment for each is enough to pull them apart.
    bank = voices(2, closeness=3.0, seed=7)
    script = [(10, 0), (10, 1)] * 6
    alone, truth = run(script, bank, noise=0.12, num_speakers=2)
    pinned, _ = run(script, bank, noise=0.12, num_speakers=2, anchors={"A": [[1, 9]], "B": [[11, 19]]})
    assert agreement(alone["turns"], truth) < 0.7
    assert agreement(pinned["turns"], truth) > 0.9
    names = {int(k): v for k, v in pinned["names"].items()}
    assert names[who_at(pinned["turns"], 45)] == "A"
    assert names[who_at(pinned["turns"], 55)] == "B"


def test_a_moment_in_silence_is_reported_unmatched():
    found, _ = run(CONVERSATION, voices(3), anchors={"Nobody": [[10_000, 10_005]]})
    assert found["unmatched"] == ["Nobody"]
    assert found["names"] == {}


def test_a_lone_flicker_inside_one_persons_talk_is_smoothed():
    spans = [(0, 2), (2, 4), (4, 5.5), (5.5, 7.5), (7.5, 9.5)]
    assert worker.smooth(spans, [0, 0, 1, 0, 0]) == [0, 0, 0, 0, 0]
    # A real turn, long enough, stays.
    assert worker.smooth([(0, 2), (2, 6), (6, 8)], [0, 1, 0]) == [0, 1, 0]


def test_speech_pieces_cut_at_every_voice_change():
    assert worker.speech_pieces([(0, 10), (4, 6), (12, 14)]) == [(0, 4), (4, 6), (6, 10), (12, 14)]


def test_windows_cover_each_piece_with_none_tiny():
    out = worker.windows_of([(0, 5.1)])
    assert out[0][0] == 0 and out[-1][1] == pytest.approx(5.1)
    assert min(b - a for a, b in out) > 1.5


def test_speakers_are_numbered_in_order_of_first_appearance():
    found, _ = run(CONVERSATION, voices(3))
    seen = []
    for t in found["turns"]:
        if t["speaker"] not in seen:
            seen.append(t["speaker"])
    assert seen == sorted(seen)
    assert set(found["voices"]) == set(seen)


def test_a_name_that_loses_every_moment_is_unmatched_not_a_crash():
    script = [(2, 0), (2, 1), (2, 0), (2, 1), (2, 0)]
    found, _ = run(script, voices(2), num_speakers=2, anchors={"A": [[0, 2]], "B": [[2.9, 3.1]], "C": [[2, 4]]})
    assert "B" in found["unmatched"] or "C" in found["unmatched"]
    names = {int(k): v for k, v in found["names"].items()}
    assert names.get(who_at(found["turns"], 1)) == "A"


def test_two_names_at_the_same_moment_name_nobody_there():
    found, _ = run(CONVERSATION, voices(3), anchors={"Moana": [[13, 17]], "Aroha": [[13, 17]]})
    assert found["names"] == {}
    assert sorted(found["unmatched"]) == ["Aroha", "Moana"]


@pytest.mark.parametrize("seconds", [16, 24, 40])
def test_a_short_recording_is_not_split_into_many(seconds):
    script = [(4, i % 2) for i in range(seconds // 4)]
    found, truth = run(script, voices(2))
    assert len({t["speaker"] for t in found["turns"]}) == 2


@pytest.mark.parametrize("seconds", [8, 16, 40])
def test_one_person_alone_stays_one_person(seconds):
    found, _ = run([(4, 0)] * (seconds // 4), voices(1))
    assert len({t["speaker"] for t in found["turns"]}) == 1
