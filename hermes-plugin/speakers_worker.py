"""Separate speakers in one recording. Runs alone, under install_speakers.sh's
environment, never imported by the plugin. Reads a request on stdin, writes turns
on stdout, and writes nothing to disk: voice samples exist only as numbers in this
process, for this one call."""

from __future__ import annotations

import json
import sys
import wave

import numpy as np
import sherpa_onnx as so

RATE = 16000
CLUSTER_THRESHOLD = 0.5
# Cosine similarity a sample needs to name a speaker. Below it the speaker stays
# "Speaker N": a missing name is better than the wrong one.
MATCH_THRESHOLD = 0.45
MAX_SPEAKER_AUDIO_S = 60


def load(path: str) -> np.ndarray:
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def embed(extractor, audio: np.ndarray) -> np.ndarray | None:
    stream = extractor.create_stream()
    stream.accept_waveform(RATE, audio)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    vector = np.array(extractor.compute(stream))
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def name_speakers(request: dict, audio: np.ndarray, turns: list[dict]) -> tuple[dict[str, str], list[str]]:
    extractor = so.SpeakerEmbeddingExtractor(
        so.SpeakerEmbeddingExtractorConfig(model=request["models"] + "/emb.onnx", num_threads=2)
    )
    voices: dict[int, np.ndarray] = {}
    for speaker in {t["speaker"] for t in turns}:
        pieces, left = [], MAX_SPEAKER_AUDIO_S * RATE
        for t in turns:
            if t["speaker"] == speaker and left > 0:
                piece = audio[int(t["start"] * RATE) : int(t["end"] * RATE)][:left]
                pieces.append(piece)
                left -= len(piece)
        vector = embed(extractor, np.concatenate(pieces)) if pieces else None
        if vector is not None:
            voices[speaker] = vector
    scores = []
    samples = {}
    for name, path in request["samples"].items():
        vector = embed(extractor, load(path))
        samples[name] = vector
        if vector is not None:
            scores += [(float(vector @ v), name, s) for s, v in voices.items()]
    names: dict[str, str] = {}
    used = set()
    for score, name, speaker in sorted(scores, reverse=True):
        if score >= MATCH_THRESHOLD and name not in used and str(speaker) not in names:
            names[str(speaker)] = name
            used.add(name)
    return names, [n for n in samples if n not in used]


def separate(models: str, audio: np.ndarray, clusters: int):
    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=models + "/seg.onnx"), num_threads=2
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=models + "/emb.onnx", num_threads=2),
        clustering=so.FastClusteringConfig(num_clusters=clusters, threshold=CLUSTER_THRESHOLD),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        sys.exit(2)
    result = so.OfflineSpeakerDiarization(config).process(audio).sort_by_start_time()
    return [{"start": float(r.start), "end": float(r.end), "speaker": int(r.speaker)} for r in result]


def main() -> None:
    request = json.load(sys.stdin)
    models = request["models"]
    audio = load(request["wav"])
    wanted = request.get("num_speakers") or -1
    turns = separate(models, audio, wanted)
    # Asking for more people than the audio can support collapses everyone into
    # one: when that happens, let the audio decide.
    if wanted > 1 and len({t["speaker"] for t in turns}) == 1:
        turns = separate(models, audio, -1)
    names, unmatched = name_speakers(request, audio, turns) if request.get("samples") else ({}, [])
    json.dump({"turns": turns, "names": names, "unmatched": unmatched}, sys.stdout)


if __name__ == "__main__":
    main()
