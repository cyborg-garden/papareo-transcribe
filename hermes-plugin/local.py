"""The local pass: faster-whisper, language auto-detected.

The model named in ``stt.local.model`` (Swarm Map sets ``large-v3-turbo``)
loads on first use and unloads after a quiet spell, so an idle agent gives
the memory back. ``language=None`` matters: whisper's language guess is one
of the detector's signals, and forcing English would also mangle te reo.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3-turbo"
IDLE_UNLOAD_S = 600.0


@dataclass(frozen=True)
class LocalResult:
    text: str
    language: str
    language_probability: float
    duration: float


def configured_model() -> str:
    try:
        from hermes_cli.config import load_config  # type: ignore[import-not-found]

        cfg = load_config() or {}
        name = ((cfg.get("stt") or {}).get("local") or {}).get("model")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:  # noqa: BLE001 - outside the harness, or no config
        pass
    return DEFAULT_MODEL


def decode(path: str) -> Any:
    """16 kHz mono float32 samples, decoded by ffmpeg.

    Whisper gets an array rather than a path: faster-whisper's own decoder
    goes through PyAV, and a PyAV newer than it expects fails on every file.
    ffmpeg also handles anything a chat platform sends (Opus, AAC, AMR, …).
    """
    import subprocess

    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
        check=False,
        timeout=600,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not decode the recording (exit {proc.returncode})")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _load(name: str) -> Any:
    try:
        # The harness's loader falls back from a broken CUDA runtime to CPU.
        from tools.transcription_tools import _load_local_whisper_model  # type: ignore[import-not-found]

        return _load_local_whisper_model(name)
    except ImportError:
        from faster_whisper import WhisperModel

        return WhisperModel(name, device="cpu", compute_type="int8")


class LocalWhisper:
    """One cached model, loaded lazily, unloaded when idle."""

    def __init__(self, model_name: str | None = None, idle_unload_s: float = IDLE_UNLOAD_S) -> None:
        self._name = model_name
        self._idle = idle_unload_s
        self._model: Any = None
        self._loaded_name: str | None = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._timer: threading.Timer | None = None

    def transcribe(self, path: str) -> LocalResult:
        name = self._name or configured_model()
        audio = decode(path)
        with self._lock:
            if self._model is None or self._loaded_name != name:
                logger.info("papareo_transcribe: loading whisper model %s", name)
                self._model = _load(name)
                self._loaded_name = name
            segments, info = self._model.transcribe(audio, language=None, vad_filter=True, beam_size=5)
            text = " ".join(s.text.strip() for s in segments).strip()
            self._last_used = time.monotonic()
            self._arm_unload()
        return LocalResult(
            text=text,
            language=str(getattr(info, "language", "") or ""),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
        )

    def _arm_unload(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._idle, self._unload_if_idle)
        self._timer.daemon = True
        self._timer.start()

    def _unload_if_idle(self) -> None:
        with self._lock:
            if self._model is not None and time.monotonic() - self._last_used >= self._idle:
                logger.info("papareo_transcribe: unloading idle whisper model %s", self._loaded_name)
                self._model = None
                self._loaded_name = None
