"""May this transcript contain te reo Māori?

Simple on purpose, and biased to say yes: a missed recording costs
transcript quality, a false alarm costs a Papa Reo call. Three signals,
any one is enough:

- whisper detected the language as Māori (``mi``);
- whisper wrote a macron anywhere;
- the share of words whose character trigrams look more Māori than
  English is over one threshold.

The trigram score exists for mixed speech, where whisper picks English and
drops the macrons. It is one number over the whole transcript (no sliding
windows), and its threshold is tuned on real recordings (scripts/tune.py).
Replace this whole module when something better becomes available.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model" / "trigram.json"
MACRONS = frozenset("āēīōūĀĒĪŌŪ")
# Runs of letters (macron vowels included), allowing an internal apostrophe.
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def words(text: str) -> list[str]:
    """NFC, lowercase, letters only. Macrons are kept: they are signal."""
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def trigrams(word: str) -> list[str]:
    padded = f"^{word}$"
    return [padded[i : i + 3] for i in range(len(padded) - 2)]


@dataclass(frozen=True)
class TrigramModel:
    mi: dict[str, float]
    en: dict[str, float]
    mi_unseen: float
    en_unseen: float
    threshold: float

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> TrigramModel:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mi=data["mi"],
            en=data["en"],
            mi_unseen=float(data["mi_unseen"]),
            en_unseen=float(data["en_unseen"]),
            threshold=float(data["threshold"]),
        )

    def word_llr(self, word: str) -> float:
        """Mean log-likelihood ratio per trigram, Māori over English."""
        grams = trigrams(word)
        total = sum(self.mi.get(g, self.mi_unseen) - self.en.get(g, self.en_unseen) for g in grams)
        return total / len(grams)

    def score(self, text: str) -> float:
        """Share of words (two letters or more) that look Māori."""
        ws = [w for w in words(text) if len(w) >= 2]
        if not ws:
            return 0.0
        return sum(1 for w in ws if self.word_llr(w) > 0) / len(ws)


@dataclass(frozen=True)
class Decision:
    escalate: bool
    reasons: tuple[str, ...]
    score: float


def decide(text: str, language: str | None, model: TrigramModel) -> Decision:
    reasons: list[str] = []
    if (language or "").lower() == "mi":
        reasons.append("language")
    if any(c in MACRONS for c in text):
        reasons.append("macron")
    score = model.score(text)
    if score > model.threshold:
        reasons.append("trigram")
    return Decision(escalate=bool(reasons), reasons=tuple(reasons), score=score)
