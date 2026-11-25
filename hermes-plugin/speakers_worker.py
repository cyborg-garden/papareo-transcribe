"""Separate speakers in one recording. Runs alone, under install_speakers.sh's
environment, never imported by the plugin. Reads a request on stdin, writes turns
on stdout, and writes nothing to disk: voice samples exist only as numbers in this
process, for this one call.

How it works:
1. sherpa-onnx finds where people speak and where the voice changes.
2. That speech is cut into windows of about two seconds, and each window gets a
   voice fingerprint.
3. The fingerprints are grouped by spectral clustering. sherpa's own grouping
   sorts voices by which part of a show they were in, not by person.
4. A moment the person points to ("Moana talks at 1:20") pins that voice to that
   person, so the grouping has to keep them apart from everyone else.
5. A lone window whose neighbours on both sides agree takes their label, so one
   sentence isn't cut up between people."""

from __future__ import annotations

import json
import math
import sys
import wave

import numpy as np

RATE = 16000
# Only used to find speech and voice changes. The grouping is ours.
CLUSTER_THRESHOLD = 0.5
# Cosine similarity a sample needs to name a speaker. Below it the speaker stays
# "Speaker N": a missing name is better than the wrong one.
MATCH_THRESHOLD = 0.45
WINDOW_S = 2.0
# Shorter windows make poor fingerprints: they are labelled afterwards instead.
MIN_WINDOW_S = 0.8
MIN_EMBED_S = 0.3
# The most people guessed when nobody said how many.
MAX_AUTO_SPEAKERS = 8
# Below this many windows (about a minute of speech) the count is chosen by
# how well the groups hold together, not by the spectrum.
SMALL_N = 40
# Groups holding together worse than this are one person. Set on real clips:
# one person alone scored 0.23-0.26, two people talking 0.48.
SILHOUETTE_MIN = 0.3
# Two groups whose average voices are at least this alike (cosine) are one person.
SAME_VOICE = 0.75
# Each window keeps only its closest neighbours, so a loud room doesn't tie everyone together.
PRUNE_KEEP = 0.15
# ...but never so few that a short recording falls apart into many voices.
MIN_NEIGHBOURS = 6
KMEANS_TRIES = 8
KMEANS_ROUNDS = 60
# A lone window shorter than this, between two of the same speaker, joins them.
FLICKER_S = 2.5
JOIN_GAP_S = 0.5


def load(path: str) -> np.ndarray:
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def _unit(v: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    return v / norm if norm and math.isfinite(norm) else None


def embed(extractor, audio: np.ndarray) -> np.ndarray | None:
    if len(audio) < MIN_EMBED_S * RATE:
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(RATE, audio)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    return _unit(np.array(extractor.compute(stream), dtype=np.float64))


def embed_long(extractor, audio: np.ndarray) -> np.ndarray | None:
    """One fingerprint for a clip of any length: the average of its windows."""
    step = int(WINDOW_S * RATE)
    vectors = [v for v in (embed(extractor, audio[i : i + step]) for i in range(0, len(audio), step)) if v is not None]
    return _unit(np.mean(vectors, axis=0)) if vectors else None


# --- where people speak ---------------------------------------------------------


def speech_pieces(turns: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Stretches of speech with no voice change inside, from sherpa's turns."""
    edges = sorted({p for t in turns for p in t})
    pieces = []
    for a, b in zip(edges, edges[1:]):
        if b > a and any(s <= a and b <= e for s, e in turns):
            pieces.append((a, b))
    return pieces


def windows_of(pieces: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Each piece cut into windows of about WINDOW_S, none left tiny at the end."""
    out = []
    for a, b in pieces:
        count = max(1, round((b - a) / WINDOW_S))
        size = (b - a) / count
        out += [(a + i * size, a + (i + 1) * size) for i in range(count)]
    return out


# --- grouping -------------------------------------------------------------------


def affinity(E: np.ndarray) -> np.ndarray:
    A = np.clip(E @ E.T, 0.0, None)
    n = len(A)
    keep = min(n, max(2, min(MIN_NEIGHBOURS, n // 2), math.ceil(PRUNE_KEEP * n)))
    if keep < n:
        cut = np.partition(A, n - keep, axis=1)[:, n - keep][:, None]
        A = np.where(A >= cut, A, 0.0)
    return (A + A.T) / 2


def spectrum(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = A.sum(axis=1)
    inv = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 0.0)
    L = np.eye(len(A)) - inv[:, None] * A * inv[None, :]
    return np.linalg.eigh(L)


def choose_k(values: np.ndarray, low: int, high: int) -> int:
    """The number of groups where the spectrum jumps most (the eigengap)."""
    high = max(low, min(high, len(values) - 1))
    if low >= high:
        return low
    gaps = [values[k] - values[k - 1] for k in range(low, high + 1)]
    return low + int(np.argmax(gaps))


def kmeans(X: np.ndarray, k: int, fixed: dict[int, int], seed: int = 0) -> np.ndarray:
    """k-means where the rows in ``fixed`` always stay in their group. Groups
    0..a-1 start at the pinned rows; the rest start k-means++ style."""
    rng = np.random.default_rng(seed)
    pinned = sorted(set(fixed.values()))
    free = np.array([i for i in range(len(X)) if i not in fixed], dtype=int)
    pool = free if len(free) else np.arange(len(X))
    best, best_cost = None, math.inf
    for _try in range(KMEANS_TRIES):
        centres = [X[[i for i, c in fixed.items() if c == g]].mean(axis=0) for g in pinned]
        while len(centres) < k:
            if not centres:
                centres.append(X[rng.choice(pool)])
                continue
            dist = np.min([((X[pool] - c) ** 2).sum(axis=1) for c in centres], axis=0)
            total = dist.sum()
            centres.append(X[rng.choice(pool, p=dist / total)] if total > 0 else X[rng.choice(pool)])
        C = np.array(centres)
        labels = np.full(len(X), -1, dtype=int)
        for _round in range(KMEANS_ROUNDS):
            dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            new = dist.argmin(axis=1)
            for i, g in fixed.items():
                new[i] = g
            if np.array_equal(new, labels):
                break
            labels = new
            for g in range(k):
                if (labels == g).any():
                    C[g] = X[labels == g].mean(axis=0)
        cost = float(((X - C[labels]) ** 2).sum())
        if cost < best_cost:
            best, best_cost = labels.copy(), cost
    return best  # type: ignore[return-value]


def group(E: np.ndarray, wanted: int | None, fixed: dict[int, int]) -> np.ndarray:
    """A group for every fingerprint. ``wanted`` is the number of people if the
    person said, and is a ceiling, not a target: one part of a long job may not
    hear everyone, and forcing the number split one person into several."""
    n = len(E)
    pinned = len(set(fixed.values()))
    if n <= 2:
        return np.array([fixed.get(i, 0) for i in range(n)], dtype=int)
    values, vectors = spectrum(affinity(E))
    low = max(1, pinned)
    high = min(n, max(low, wanted or MAX_AUTO_SPEAKERS))

    def split(k: int) -> np.ndarray:
        if k <= 1:
            return np.zeros(n, dtype=int)
        U = vectors[:, :k]
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        labels = kmeans(U / np.where(norms > 0, norms, 1), k, fixed)
        return refine(E, labels, fixed) if fixed else labels

    if n >= SMALL_N:
        return merge_alike(E, split(min(choose_k(values, low, high), n)), pinned)
    # Too few windows for the spectrum to be trusted: try each count and keep
    # the one whose groups hold together best.
    best, best_score = split(low), -math.inf
    for k in range(max(2, low), high + 1):
        labels = split(k)
        score = silhouette(E, labels)
        if score > best_score:
            best, best_score = labels, score
    if low == 1 and best_score < SILHOUETTE_MIN:
        return np.zeros(n, dtype=int)
    return merge_alike(E, best, pinned)


def merge_alike(E: np.ndarray, labels: np.ndarray, pinned: int) -> np.ndarray:
    """Groups whose average voices are this alike are one person split in two.
    Groups below ``pinned`` are named people and never merge with each other."""
    labels = labels.copy()
    while True:
        groups = sorted(set(int(x) for x in labels))
        centres = {g: _unit(E[labels == g].mean(axis=0)) for g in groups}
        pairs = [
            (float(centres[a] @ centres[b]), a, b)
            for i, a in enumerate(groups) for b in groups[i + 1 :]
            if centres[a] is not None and centres[b] is not None and not (a < pinned and b < pinned)
        ]
        if not pairs:
            return labels
        score, a, b = max(pairs)
        if score < SAME_VOICE:
            return labels
        labels[labels == b] = a


def silhouette(E: np.ndarray, labels: np.ndarray) -> float:
    """How well each fingerprint sits in its own group rather than the nearest
    other one, from -1 to 1, by cosine distance."""
    groups = sorted(set(int(x) for x in labels))
    if len(groups) < 2:
        return -1.0
    D = 1 - E @ E.T
    scores = []
    for i in range(len(E)):
        own = labels == labels[i]
        if own.sum() <= 1:
            scores.append(0.0)
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == g].mean() for g in groups if g != labels[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def refine(E: np.ndarray, labels: np.ndarray, fixed: dict[int, int]) -> np.ndarray:
    """Settle each fingerprint with the nearest voice, in voice space, keeping the
    pinned ones where the person put them. The spectral step finds the groups;
    this lets a pinned moment pull in voices that sound like it, which matters
    most when two people sound alike."""
    labels = labels.copy()
    groups = sorted(set(int(x) for x in labels))
    for _round in range(KMEANS_ROUNDS):
        centres = np.array([E[labels == g].mean(axis=0) for g in groups])
        centres /= np.where(np.linalg.norm(centres, axis=1, keepdims=True) > 0, np.linalg.norm(centres, axis=1, keepdims=True), 1)
        new = np.array(groups)[(E @ centres.T).argmax(axis=1)]
        for i, g in fixed.items():
            new[i] = g
        if np.array_equal(new, labels):
            break
        labels = new
        groups = sorted(set(int(x) for x in labels))
    return labels


def smooth(spans: list[tuple[float, float]], labels: list[int]) -> list[int]:
    """A short window between two of the same speaker becomes theirs."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        a, b = spans[i]
        if out[i - 1] == out[i + 1] != out[i] and b - a < FLICKER_S:
            out[i] = out[i - 1]
    return out


def turns_of(spans: list[tuple[float, float]], labels: list[int]) -> list[dict]:
    turns: list[dict] = []
    for (a, b), g in sorted(zip(spans, labels)):
        if turns and turns[-1]["speaker"] == g and a - turns[-1]["end"] <= JOIN_GAP_S:
            turns[-1]["end"] = max(turns[-1]["end"], b)
        else:
            turns.append({"start": float(a), "end": float(b), "speaker": int(g)})
    return turns


def first_seen(labels: list[int]) -> dict[int, int]:
    """Groups renumbered 0, 1, 2... in order of first appearance."""
    order: dict[int, int] = {}
    for g in labels:
        order.setdefault(g, len(order))
    return order


# --- names ----------------------------------------------------------------------


def name_speakers(
    samples: dict[str, np.ndarray | None], voices: dict[int, np.ndarray], taken: set[int]
) -> tuple[dict[str, str], list[str]]:
    scores = []
    for name, vector in samples.items():
        if vector is not None:
            scores += [(float(vector @ v), name, s) for s, v in voices.items() if s not in taken]
    names: dict[str, str] = {}
    used = set()
    for score, name, speaker in sorted(scores, reverse=True):
        if score >= MATCH_THRESHOLD and name not in used and str(speaker) not in names:
            names[str(speaker)] = name
            used.add(name)
    return names, [n for n in samples if n not in used]


# --- the whole separation -------------------------------------------------------


def separate(request: dict, audio: np.ndarray, segment, extractor) -> dict:
    """``segment(audio)`` gives speech turns as (start, end); ``extractor`` makes
    fingerprints. Both are sherpa-onnx in real use."""
    duration = len(audio) / RATE
    spans = windows_of(speech_pieces(list(segment(audio))))
    anchors: dict[str, list[tuple[float, float]]] = {}
    for name, bands in (request.get("anchors") or {}).items():
        kept = [(max(0.0, float(a)), min(duration, float(b))) for a, b in bands]
        kept = [(a, b) for a, b in kept if b > a]
        if kept:
            anchors[name] = kept
        else:
            anchors[name] = []

    rows: list[np.ndarray] = []
    row_span: list[tuple[float, float] | None] = []
    short: list[tuple[float, float]] = []
    for a, b in spans:
        v = embed(extractor, audio[int(a * RATE) : int(b * RATE)]) if b - a >= MIN_WINDOW_S else None
        if v is None:
            short.append((a, b))
        else:
            rows.append(v)
            row_span.append((a, b))

    # Pin the moments the person pointed to. A moment that overlaps another
    # person's is dropped for both rather than guessed, so every pinned window
    # belongs to exactly one name.
    def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return min(a[1], b[1]) > max(a[0], b[0])

    clear = {
        name: [
            band for band in bands
            if not any(overlaps(band, other) for o, theirs in anchors.items() if o != name for other in theirs)
        ]
        for name, bands in anchors.items()
    }
    fixed: dict[int, int] = {}
    anchor_group: dict[str, int] = {}
    missing: list[str] = []
    for name, bands in clear.items():
        hits: list[int] = []
        for i, span in enumerate(row_span):
            if span is not None and i not in fixed and any(s <= (span[0] + span[1]) / 2 <= e for s, e in bands):
                hits.append(i)
        for s, e in bands:
            v = embed_long(extractor, audio[int(s * RATE) : int(e * RATE)])
            if v is not None:
                rows.append(v)
                row_span.append(None)
                hits.append(len(rows) - 1)
        if hits:
            g = len(anchor_group)  # groups 0..a-1, with no gaps: kmeans relies on it
            anchor_group[name] = g
            fixed.update({i: g for i in hits})
        else:
            missing.append(name)

    E = np.array(rows) if rows else np.zeros((0, 1))
    labels = group(E, request.get("num_speakers") or None, fixed)
    centres: dict[int, np.ndarray] = {}
    for g in sorted(set(int(x) for x in labels)):
        c = _unit(E[labels == g].mean(axis=0))
        if c is not None:
            centres[g] = c

    real = [i for i, s in enumerate(row_span) if s is not None]
    timed: list[tuple[float, float]] = [row_span[i] for i in real]  # type: ignore[misc]
    window_labels = [int(labels[i]) for i in real]
    # Short pieces go to the closest voice, or failing that the nearest window in time.
    placed_timed, placed = list(timed), list(window_labels)
    for a, b in short:
        v = embed(extractor, audio[int(a * RATE) : int(b * RATE)])
        if v is not None and centres:
            g = max(centres, key=lambda c: float(v @ centres[c]))
        elif timed:
            mid = (a + b) / 2
            g = window_labels[min(range(len(timed)), key=lambda i: abs((timed[i][0] + timed[i][1]) / 2 - mid))]
        else:
            continue
        placed_timed.append((a, b))
        placed.append(g)

    order = sorted(range(len(placed_timed)), key=lambda i: placed_timed[i][0])
    spans_sorted = [placed_timed[i] for i in order]
    smoothed = smooth(spans_sorted, [placed[i] for i in order])
    number = first_seen(smoothed)
    turns = turns_of(spans_sorted, [number[g] for g in smoothed])

    voices = {number[g]: c for g, c in centres.items() if g in number}
    names = {str(number[g]): name for name, g in anchor_group.items() if g in number}
    unmatched = list(missing) + [name for name, g in anchor_group.items() if g not in number]
    if request.get("samples"):
        clips = {name: embed_long(extractor, load(path)) for name, path in request["samples"].items()}
        taken = {int(k) for k in names}
        more, left = name_speakers({n: v for n, v in clips.items() if n not in anchor_group}, voices, taken)
        names.update(more)
        unmatched += left
    # One entry per name, and never both named and unmatched.
    named = set(names.values())
    unmatched = [n for n in dict.fromkeys(unmatched) if n not in named]
    return {"turns": turns, "names": names, "unmatched": unmatched, "voices": voices}


def sherpa_parts(models: str):
    import sherpa_onnx as so

    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=models + "/seg.onnx"), num_threads=2
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=models + "/emb.onnx", num_threads=2),
        clustering=so.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        sys.exit(2)
    diarizer = so.OfflineSpeakerDiarization(config)
    extractor = so.SpeakerEmbeddingExtractor(
        so.SpeakerEmbeddingExtractorConfig(model=models + "/emb.onnx", num_threads=2)
    )

    def segment(audio: np.ndarray) -> list[tuple[float, float]]:
        return [(float(r.start), float(r.end)) for r in diarizer.process(audio).sort_by_start_time()]

    return segment, extractor


def main() -> None:
    request = json.load(sys.stdin)
    audio = load(request["wav"])
    segment, extractor = sherpa_parts(request["models"])
    found = separate(request, audio, segment, extractor)
    answer = {"turns": found["turns"], "names": found["names"], "unmatched": found["unmatched"]}
    if request.get("want_voices"):
        answer["voices"] = {str(s): v.tolist() for s, v in found["voices"].items()}
    json.dump(answer, sys.stdout)


if __name__ == "__main__":
    main()
