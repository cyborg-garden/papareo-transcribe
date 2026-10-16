"""Train the detector's trigram model and set a provisional threshold.

    python -I scripts/fetch_corpus.py
    python -I scripts/train_detector.py

Positives are the reo-dominant utterances (70% or more) of the Hansard
corpus. Each is used twice: as written, and with macrons stripped, because
the trigram signal exists for transcripts where whisper has dropped them.
Negatives are the Mansfield stories. One tenth of each is held out.

The threshold written here is provisional. It is chosen so that nearly all
synthetic mixed passages (English with a macron-stripped Māori sentence
spliced in) trip; the real threshold comes from scripts/tune.py on real
recordings, because whisper's output is not Hansard.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hermes-plugin"))

import detector  # noqa: E402

CACHE = ROOT / ".cache"
OUT = ROOT / "hermes-plugin" / "model" / "trigram.json"
K = 0.5  # add-k smoothing
MIN_COUNT = 2  # trigrams seen fewer times in both classes are left to the unseen value
REO_MIN_PERCENT = 70.0
RECALL_TARGET = 0.98
SEED = 20261016

# A few words an English speaker in Aotearoa uses every day. Mixed into
# English passages to show the false-positive rate this design accepts.
EVERYDAY_REO = [
    "kia ora", "Aotearoa", "whanau", "Whanganui", "Tauranga", "Rotorua", "kai", "haka",
    "Ngai Tahu", "marae", "Te Papa", "kia kaha", "aroha", "Paekakariki", "hangi", "Whangarei",
]


def strip_macrons(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def held_out(key: str) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10 == 0


def load_reo() -> tuple[list[str], list[str]]:
    data = CACHE / "corpus" / "data"
    [path] = [p for p in data.iterdir() if unicodedata.normalize("NFC", p.name) == "hansardreomāori.csv"]
    train, test = [], []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                percent = float(row["percent"])
            except ValueError:
                continue
            text = row["text"].strip()
            if percent < REO_MIN_PERCENT or not text:
                continue
            (test if held_out(row["url"] + row["utterance"]) else train).append(text)
    return train, test


def load_english() -> tuple[list[str], list[str]]:
    train, test = [], []
    for p in sorted((CACHE / "english").glob("pg*.txt")):
        text = p.read_text(encoding="utf-8").replace("\r\n", "\n")
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        body = text[text.find("\n", start) + 1 : end] if start >= 0 and end > start else text
        paras = [re.sub(r"\s+", " ", para).strip() for para in body.split("\n\n")]
        for i, para in enumerate(paras):
            if len(para.split()) < 5:
                continue
            (test if held_out(f"{p.name}:{i}") else train).append(para)
    return train, test


def count(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for w in detector.words(t):
            c.update(detector.trigrams(w))
    return c


def log_probs(c: Counter, vocab: set[str]) -> tuple[dict[str, float], float]:
    total = sum(c.values()) + K * (len(vocab) + 1)
    probs = {g: math.log((c[g] + K) / total) for g in vocab}
    return probs, math.log(K / total)


def passages(texts: list[str], size: int, rng: random.Random) -> list[str]:
    """Memo-sized runs of English words (a voice memo is ~30-150 words)."""
    words = " ".join(texts).split()
    out = []
    for i in range(0, len(words) - size, size):
        out.append(" ".join(words[i : i + size]))
    rng.shuffle(out)
    return out


def splice(english: str, insert: str, rng: random.Random) -> str:
    ws = english.split()
    at = rng.randrange(len(ws))
    return " ".join(ws[:at] + insert.split() + ws[at:])


def main() -> None:
    rng = random.Random(SEED)
    reo_train, reo_test = load_reo()
    en_train, en_test = load_english()
    positives = reo_train + [strip_macrons(t) for t in reo_train]
    mi_c, en_c = count(positives), count(en_train)
    vocab = {g for g in set(mi_c) | set(en_c) if mi_c[g] + en_c[g] >= MIN_COUNT}
    mi, mi_unseen = log_probs(mi_c, vocab)
    en, en_unseen = log_probs(en_c, vocab)
    model = detector.TrigramModel(mi=mi, en=en, mi_unseen=mi_unseen, en_unseen=en_unseen, threshold=0.0)

    # Word level, held out.
    mi_words = [w for t in reo_test for w in detector.words(strip_macrons(t)) if len(w) >= 2]
    en_words = [w for t in en_test for w in detector.words(t) if len(w) >= 2]
    mi_acc = sum(model.word_llr(w) > 0 for w in mi_words) / len(mi_words)
    en_acc = sum(model.word_llr(w) <= 0 for w in en_words) / len(en_words)

    # Memo level, synthetic.
    english = passages(en_test, 60, rng)[:400]
    reo_sentences = [strip_macrons(t) for t in reo_test if 6 <= len(t.split()) <= 15]
    mixed = [splice(e, rng.choice(reo_sentences), rng) for e in english]
    named = [splice(e, " ".join(rng.sample(EVERYDAY_REO, 2)), rng) for e in english]
    s_en = sorted(model.score(t) for t in english)
    s_mixed = sorted(model.score(t) for t in mixed)
    s_named = sorted(model.score(t) for t in named)
    s_reo = sorted(model.score(strip_macrons(t)) for t in reo_test)

    cut = s_mixed[int((1 - RECALL_TARGET) * len(s_mixed))]
    threshold = round(max(0.0, cut - 0.005), 3)

    def rate(xs: list[float]) -> float:
        return sum(x > threshold for x in xs) / len(xs)

    report = {
        "words": {"maori_heldout_correct": round(mi_acc, 4), "english_heldout_correct": round(en_acc, 4)},
        "threshold": threshold,
        "trips": {
            "reo_utterances_no_macrons": round(rate(s_reo), 4),
            "english_plus_one_reo_sentence": round(rate(s_mixed), 4),
            "english_plus_two_everyday_words": round(rate(s_named), 4),
            "english_only": round(rate(s_en), 4),
        },
        "median_score": {
            "english_only": round(statistics.median(s_en), 4),
            "english_plus_two_everyday_words": round(statistics.median(s_named), 4),
            "english_plus_one_reo_sentence": round(statistics.median(s_mixed), 4),
        },
    }
    payload = {
        "version": 1,
        "about": (
            "Character-trigram log-probabilities, Maori (mi) and English (en), for "
            "detector.TrigramModel. Built by scripts/train_detector.py."
        ),
        "sources": {
            "mi": "TeHikuMedia/nga-tautohetohe-reo @ "
            "ffb3b704816d4d4d06f0350b71e6f8bcca6014f2, hansardreomāori.csv, "
            f"utterances >= {REO_MIN_PERCENT:.0f}% reo, as written and macron-stripped (public domain)",
            "en": "Katherine Mansfield, The Garden Party and Bliss, Project Gutenberg #1429 and "
            "#44385 (public domain)",
        },
        "k": K,
        "threshold": threshold,
        "threshold_note": (
            "Provisional: set from synthetic mixed passages at "
            f"{RECALL_TARGET:.0%} recall. Replace with the value from scripts/tune.py "
            "on real recordings."
        ),
        "report": report,
        "mi_unseen": round(mi_unseen, 4),
        "en_unseen": round(en_unseen, 4),
        "mi": {g: round(v, 3) for g, v in sorted(mi.items())},
        "en": {g: round(v, 3) for g, v in sorted(en.items())},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KiB, {len(vocab)} trigrams)")


if __name__ == "__main__":
    main()
