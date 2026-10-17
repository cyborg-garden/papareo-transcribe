"""Tune the detector's threshold on real recordings.

    scripts/tune-in-agent-image.sh ~/dev/digicryptids/eval-memos

or, with faster-whisper installed locally:

    python -I scripts/tune.py --memos ~/dev/digicryptids/eval-memos [--model large-v3-turbo] [--write]

The memos folder holds the recordings and a labels.csv with the columns
``file,should_escalate,kind,notes`` (should_escalate is yes or no; kind is
free text such as english, english+names, mixed, reo). It lives outside
every repo: it is speech data and never goes in git.

Whisper runs with exactly the tool's settings, so tuning sees whisper's
real output. Transcripts are cached in <memos>/.transcripts/ (delete the
folder to redo them). The report shows each memo's signals, then the
highest threshold at which every should-escalate memo still trips. With
--write, that threshold (less a small margin) goes into the model file.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "hermes-plugin"
MARGIN = 0.01


def load_plugin_module(name: str):
    spec = importlib.util.spec_from_file_location(
        "papareo_transcribe", PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["papareo_transcribe"] = pkg
    sub = importlib.util.spec_from_file_location(f"papareo_transcribe.{name}", PLUGIN / f"{name}.py")
    mod = importlib.util.module_from_spec(sub)
    sys.modules[f"papareo_transcribe.{name}"] = mod
    sub.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--memos", required=True, type=Path)
    ap.add_argument("--model", default=None, help="whisper model (default: the tool's)")
    ap.add_argument("--write", action="store_true", help="write the tuned threshold into the model file")
    args = ap.parse_args()

    detector = load_plugin_module("detector")
    local = load_plugin_module("local")
    model = detector.TrigramModel.load()
    whisper = local.LocalWhisper(args.model or local.DEFAULT_MODEL, idle_unload_s=3600)

    labels_path = args.memos / "labels.csv"
    if not labels_path.exists():
        print(f"no {labels_path}; expected columns file,should_escalate,kind,notes", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    cache = args.memos / ".transcripts"
    cache.mkdir(exist_ok=True)

    results = []
    for row in rows:
        audio = args.memos / row["file"]
        if not audio.exists():
            print(f"missing: {audio}", file=sys.stderr)
            continue
        cached = cache / (row["file"] + ".json")
        if cached.exists():
            r = json.loads(cached.read_text(encoding="utf-8"))
        else:
            res = whisper.transcribe(str(audio))
            r = {"text": res.text, "language": res.language, "p": res.language_probability, "duration": res.duration}
            cached.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        d = detector.decide(r["text"], r["language"], model)
        results.append({
            "file": row["file"],
            "want": row["should_escalate"].strip().lower() in {"yes", "y", "true", "1"},
            "kind": row.get("kind", ""),
            "language": r["language"],
            "macron": any(c in detector.MACRONS for c in r["text"]),
            "score": d.score,
            "decision": d.escalate,
            "reasons": ",".join(d.reasons),
        })

    print(f"{'file':32} {'want':5} {'got':5} {'lang':4} {'macron':6} {'score':>6}  kind / reasons")
    for r in results:
        flag = "" if r["want"] == r["decision"] else ("  <- MISSED" if r["want"] else "  (false positive)")
        print(f"{r['file'][:32]:32} {str(r['want']):5} {str(r['decision']):5} {r['language']:4} "
              f"{str(r['macron']):6} {r['score']:6.3f}  {r['kind']} / {r['reasons']}{flag}")

    wanted = [r for r in results if r["want"]]
    # Memos that only the trigram can catch decide the threshold.
    trigram_only = [r for r in wanted if r["language"] != "mi" and not r["macron"]]
    current = model.threshold
    missed = [r["file"] for r in wanted if not r["decision"]]
    fp = sum(1 for r in results if not r["want"] and r["decision"])
    print(f"\nthreshold now {current:.3f}: missed {len(missed)} of {len(wanted)} should-escalate; "
          f"{fp} false positive(s) of {len(results) - len(wanted)}")
    if not trigram_only:
        print("every should-escalate memo trips on language or macrons; the trigram threshold is not tested here")
        return 0 if not missed else 1
    floor = min(r["score"] for r in trigram_only)
    suggested = round(max(0.0, floor - MARGIN), 3)
    print(f"lowest trigram score among memos only the trigram can catch: {floor:.3f}; suggested threshold {suggested:.3f}")
    if args.write:
        path = detector.MODEL_PATH
        data = json.loads(path.read_text(encoding="utf-8"))
        data["threshold"] = suggested
        data["threshold_note"] = (
            f"Tuned by scripts/tune.py on {len(results)} real recordings "
            f"({len(wanted)} should escalate): lowest trigram-only score {floor:.3f} less {MARGIN}."
        )
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote threshold {suggested:.3f} to {path.relative_to(ROOT)}")
    return 0 if not missed else 1


if __name__ == "__main__":
    sys.exit(main())
