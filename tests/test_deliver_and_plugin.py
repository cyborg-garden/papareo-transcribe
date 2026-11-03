"""Filenames, and the plugin's registration."""

from __future__ import annotations

import json
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
    assert set(tool["schema"]["parameters"]["properties"]) == {
        "path", "url", "engine", "chosen_by_person", "speakers", "num_speakers", "voice_samples",
        "long_job_ok", "resume_job", "discard_job",
    }
    # The handler refuses when the agent picked the engine itself.
    r = json.loads(tool["handler"]({"path": "/nonexistent.ogg", "engine": "local"}))
    assert r["error"].startswith("Nothing was transcribed. Ask the person first")
    assert "k-secret-123" not in repr(tool)


def test_local_whisper_hands_the_model_decoded_samples(plugin, monkeypatch, tmp_path):
    """faster-whisper never sees a path: its PyAV decoder breaks on newer PyAV."""
    seen = {}

    class FakeModel:
        def transcribe(self, audio, **kw):
            seen["audio"], seen["kw"] = audio, kw
            seg = type("S", (), {"text": " kia ora ", "start": 0.0, "end": 1.0})()
            info = type("I", (), {"language": "mi", "language_probability": 0.8, "duration": 1.5})()
            return iter([seg]), info

    monkeypatch.setattr(plugin.local, "decode", lambda path: [0.0] * 24000)
    monkeypatch.setattr(plugin.local, "_load", lambda name: FakeModel())
    w = plugin.local.LocalWhisper("large-v3-turbo", idle_unload_s=60)
    r = w.transcribe(str(tmp_path / "x.ogg"))
    cue = plugin.speakers.Cue(0.0, 1.0, "kia ora")
    assert r == plugin.local.LocalResult("kia ora", "mi", 0.8, 1.5, model="large-v3-turbo", cues=(cue,))
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


def _registered(plugin, monkeypatch, tmp_path, cache=None):
    tools = {}

    class Ctx:
        def register_tool(self, name, toolset, schema, handler, description, emoji):
            tools[name] = handler

        def dispatch_tool(self, *a, **k):
            return "{}"

    monkeypatch.setattr(plugin, "_voice_caches", lambda: [cache] if cache else [])
    plugin.register(Ctx())
    return tools["transcribe_recording"]


@pytest.mark.parametrize("args,match", [
    ({"voice_samples": "not json"}, "voice_samples"),
    ({"voice_samples": ["a"]}, "voice_samples"),
    ({"num_speakers": "many"}, "num_speakers"),
    ({"num_speakers": float("inf")}, "num_speakers"),
    ({"num_speakers": 500}, "num_speakers"),
])
def test_bad_options_are_refused_not_dropped(plugin, monkeypatch, tmp_path, args, match):
    handle = _registered(plugin, monkeypatch, tmp_path)
    r = json.loads(handle({"path": "/x", "engine": "papareo", "chosen_by_person": True, **args}))
    assert r["sent"] is False and match in r["error"]


def test_nothing_downloads_when_the_call_would_be_refused(plugin, monkeypatch, tmp_path):
    handle = _registered(plugin, monkeypatch, tmp_path)
    monkeypatch.setattr(plugin.fetch, "fetch", lambda *a, **k: pytest.fail("must not download"))
    for args in ({"chosen_by_person": False, "engine": "papareo"}, {"chosen_by_person": True, "engine": "bogus"},
                 {"chosen_by_person": True, "engine": "papareo", "voice_samples": {"a": str(tmp_path / "no")}}):
        json.loads(handle({"url": "https://example.org/a.mp3", **args}))


def test_a_page_saved_as_mp3_is_never_sent_to_papa_reo(plugin, monkeypatch, tmp_path):
    handle = _registered(plugin, monkeypatch, tmp_path)
    fake = tmp_path / "recording.mp3"

    def fetched(url, dest):
        fake.write_bytes(b"not audio")
        return str(fake)

    monkeypatch.setattr(plugin.fetch, "fetch", fetched)
    monkeypatch.setattr(plugin, "has_audio", lambda p: False)
    monkeypatch.setattr(plugin.transcribe, "run", lambda *a, **k: pytest.fail("must not run"))
    r = json.loads(handle({"url": "https://example.org/a.mp3", "engine": "papareo", "chosen_by_person": True}))
    assert r["sent"] is False and "anyone with the link" in r["error"]


def test_used_clips_leave_the_cache_on_every_outcome(plugin, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    clip = cache / "audio_abc.ogg"
    recording = cache / "audio_rec.ogg"
    recording.write_bytes(b"x")
    outside = tmp_path / "mine.ogg"
    outside.write_bytes(b"x")
    handle = _registered(plugin, monkeypatch, tmp_path, cache=cache.resolve())

    def used_then_boom(*a, samples_used, **k):
        samples_used()
        raise RuntimeError("boom")

    def used_and_failed(*a, samples_used, **k):
        samples_used()
        return {"sent": False}

    def used_and_sent(*a, samples_used, **k):
        samples_used()
        return {"sent": True}

    for run in (used_then_boom, used_and_failed, used_and_sent):
        clip.write_bytes(b"x")
        monkeypatch.setattr(plugin.transcribe, "run", run)
        samples = {"a": str(clip), "b": str(outside), "c": str(recording)}
        try:
            handle({"path": str(recording), "engine": "local", "chosen_by_person": True, "voice_samples": samples})
        except RuntimeError:
            pass
        assert not clip.exists() and outside.exists() and recording.exists()


def test_a_clip_is_kept_when_separation_never_ran(plugin, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    clip = cache / "audio_abc.ogg"
    clip.write_bytes(b"x")
    handle = _registered(plugin, monkeypatch, tmp_path, cache=cache.resolve())
    monkeypatch.setattr(plugin.transcribe, "run", lambda *a, **k: {"sent": False, "error": "no recording at that path"})
    handle({"path": "/typo.ogg", "engine": "papareo", "chosen_by_person": True, "voice_samples": {"a": str(clip)}})
    assert clip.exists()


def test_names_that_collide_after_cleaning_are_refused(plugin):
    assert plugin._read_samples({"a\n": "/1.ogg", "a": "/2.ogg"})[1]


def test_a_refused_call_keeps_its_clips_for_the_retry(plugin, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    clip = cache / "audio_abc.ogg"
    clip.write_bytes(b"OggS")
    handle = _registered(plugin, monkeypatch, tmp_path, cache=cache.resolve())
    good = {"path": str(clip), "engine": "papareo", "voice_samples": {"a": str(clip)}}
    assert "Ask the person" in json.loads(handle({**good, "chosen_by_person": False}))["error"]
    assert clip.exists()
    json.loads(handle({**good, "chosen_by_person": True, "voice_samples": {"a": str(clip), "b": "/nope.ogg"}}))
    assert clip.exists()
    json.loads(handle({**good, "chosen_by_person": True, "num_speakers": "many"}))
    assert clip.exists()


def test_a_name_cannot_forge_a_line(plugin):
    samples, bad = plugin._read_samples({"Aroha\n[00:00] Someone else": "/a.ogg"})
    assert bad is None and list(samples) == ["Aroha [00:00] Someone else"]
    assert plugin._read_samples({"\n": "/a.ogg"})[1]


def test_probe_failure_is_not_paid_for(plugin, monkeypatch, tmp_path):
    handle = _registered(plugin, monkeypatch, tmp_path)
    fake = tmp_path / "recording.mp3"
    fake.write_bytes(b"x")
    monkeypatch.setattr(plugin.fetch, "fetch", lambda url, dest: str(fake))

    def missing(path):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(plugin, "has_audio", missing)
    monkeypatch.setattr(plugin.transcribe, "run", lambda *a, **k: pytest.fail("must not run"))
    r = json.loads(handle({"url": "https://example.org/a.mp3", "engine": "papareo", "chosen_by_person": True}))
    assert r["sent"] is False
