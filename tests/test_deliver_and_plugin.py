"""Filenames, and the plugin's registration."""

from __future__ import annotations

import os
from datetime import datetime

import pytest

WHEN = datetime(2026, 10, 16, 15, 30)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/opt/data/cache/audio/audio_0123456789ab.ogg", "transcript-20261016-1530.txt"),
        ("/opt/data/cache/documents/doc_0123456789ab_Hui notes 3.m4a", "transcript-20261016-1530-Hui-notes-3.txt"),
        ("/x/whaikōrero.wav", "transcript-20261016-1530-whaikorero.txt"),
        ("/x/" + "a" * 80 + ".mp3", "transcript-20261016-1530-" + "a" * 40 + ".txt"),
        ("/x/papareo-output.mp3", "transcript-20261016-1530-papareo-output.txt"),
    ],
)
def test_filename(plugin, path, expected):
    assert plugin.deliver.filename_for(path, WHEN) == expected


def test_register_pops_the_key_and_registers_one_tool(plugin, monkeypatch):
    monkeypatch.setenv("PAPAREO_API_KEY", "k-secret-123")
    registered = []

    class Ctx:
        def register_tool(self, **kw):
            registered.append(kw)

        def dispatch_tool(self, name, args):  # pragma: no cover
            raise AssertionError

    plugin.register(Ctx())
    assert "PAPAREO_API_KEY" not in os.environ
    [tool] = registered
    assert tool["name"] == "transcribe_recording"
    assert set(tool["schema"]["parameters"]["properties"]) == {"path", "engine"}
    assert "k-secret-123" not in repr(tool)


def test_local_whisper_hands_the_model_decoded_samples(plugin, monkeypatch, tmp_path):
    """faster-whisper never sees a path: its PyAV decoder breaks on newer PyAV."""
    seen = {}

    class FakeModel:
        def transcribe(self, audio, **kw):
            seen["audio"], seen["kw"] = audio, kw
            seg = type("S", (), {"text": " kia ora "})()
            info = type("I", (), {"language": "mi", "language_probability": 0.8, "duration": 1.5})()
            return iter([seg]), info

    monkeypatch.setattr(plugin.local, "decode", lambda path: [0.0] * 24000)
    monkeypatch.setattr(plugin.local, "_load", lambda name: FakeModel())
    w = plugin.local.LocalWhisper("large-v3-turbo", idle_unload_s=60)
    r = w.transcribe(str(tmp_path / "x.ogg"))
    assert r == plugin.local.LocalResult("kia ora", "mi", 0.8, 1.5, model="large-v3-turbo")
    assert seen["audio"] == [0.0] * 24000
    assert seen["kw"]["language"] is None, "language must be auto-detected"


def test_has_audio_reads_content_not_the_extension(plugin, tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("needs ffmpeg")
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", str(wav)], check=True)
    disguised = tmp_path / "attachment.bin"
    disguised.write_bytes(wav.read_bytes())
    junk = tmp_path / "notes.bin"
    junk.write_text("hello")
    assert plugin.local.has_audio(str(disguised)) is True
    assert plugin.local.has_audio(str(junk)) is False
    assert 0.9 < plugin.local.probe_duration(str(disguised)) < 1.1
