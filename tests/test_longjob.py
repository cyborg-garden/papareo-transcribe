"""Long recordings: parts, speakers across parts, what is paid for, and resuming."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

SECRET = "Tēnā koutou"


@pytest.fixture
def lj(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv(plugin.longjob.HOME_ENV, str(tmp_path / "jobs"))
    return plugin.longjob


def fake_ffmpeg(cmd, **kw):
    if "null" in cmd:
        return subprocess.CompletedProcess(cmd, 0, b"", b"silence_end: 1810 | silence_duration: 2\n")
    Path(cmd[-1]).write_bytes(b"flac")
    return subprocess.CompletedProcess(cmd, 0, b"", b"")


class FakeClient:
    def __init__(self, plugin, fail=None):
        self.plugin = plugin
        self.started = []
        self.finished = []
        self.fail = fail or {}

    def start_task(self, path):
        self.started.append(path)
        return f"task-{len(self.started) - 1}"

    def finish_task(self, task):
        self.finished.append(task)
        error = self.fail.pop(task, None)
        if error:
            raise error
        from papareo_transcribe.papareo import Timed
        from papareo_transcribe.speakers import Cue

        if task == "task-silent":
            return Timed("")
        word = f"{SECRET} {task}."
        return Timed(word, (Cue(1, 5, word),))


def world(plugin, lj, tmp_path, duration=4000.0, speakers=True, count=2, diarize=None, papareo=None, samples=None):
    sent, said, used = [], [], []
    (tmp_path / "r.m4a").write_bytes(b"audio")
    job = lj.new_job(str(tmp_path / "r.m4a"), duration, "signal", "chat", "t.txt", speakers, count,
                     lambda a, b: Path(b).write_bytes(Path(a).read_bytes()), fake_ffmpeg)
    env = lj.Env(
        papareo=papareo or FakeClient(plugin), diarize=diarize,
        attach=lambda name, text, caption: sent.append(text), say=said.append,
        samples=samples or {}, samples_used=lambda: used.append(1), run=fake_ffmpeg,
    )
    return job, env, sent, said, used


def run(plugin, lj, job, env):
    lj.run_job(job, env, plugin.transcribe.lay_out, "Transcript")


def two_voices(plugin):
    """Local numbering swaps between parts, as it does on real audio."""
    sp = plugin.speakers
    calls = []

    def diarize(path, count, samples, seconds, want_voices):
        assert want_voices is True
        flipped = len(calls) % 2 == 1
        calls.append(path)
        a, b = (1, 0) if flipped else (0, 1)
        return sp.Diarized(
            (sp.Turn(0, 3, a), sp.Turn(3, 6, b)), voices={a: (1.0, 0.0), b: (0.0, 1.0)}
        )

    return diarize


def test_parts_cover_everything_with_no_gaps(lj):
    parts = lj.plan_parts(15 * 3600.0, ())
    assert parts[0][0] == 0 and parts[-1][1] == 15 * 3600.0
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
    assert len(parts) == 30


def test_parts_cut_at_the_nearest_quiet_moment(lj):
    parts = lj.plan_parts(4000.0, (1000.0, 1750.0, 1810.0))
    assert parts[0] == (0.0, 1810.0)


def test_a_quiet_moment_too_far_away_is_not_used(lj):
    assert lj.plan_parts(4000.0, (100.0,))[0] == (0.0, lj.CHUNK_S)


def test_no_tiny_part_at_the_end(lj):
    assert len(lj.plan_parts(lj.CHUNK_S + 60, ())) == 1


def test_silences_read_from_ffmpeg(lj):
    assert lj.silences("x", fake_ffmpeg) == [1809.0]


def test_silences_failing_means_fixed_cuts(lj):
    def broken(*a, **k):
        raise OSError

    assert lj.silences("x", broken) == []


def test_voices_are_linked_across_parts(lj):
    linker = lj.VoiceLinker(None)
    assert linker.person((1.0, 0.0)) == 0
    assert linker.person((0.0, 1.0)) == 1
    assert linker.person((0.99, 0.05)) == 0


def test_the_same_person_split_in_a_part_stays_one_person(lj):
    linker = lj.VoiceLinker(2)
    assert {linker.person((1.0, 0.0)), linker.person((0.9, 0.1))} == {0}


def test_never_more_people_than_they_said(lj):
    linker = lj.VoiceLinker(2)
    linker.person((1.0, 0.0))
    linker.person((0.0, 1.0))
    assert linker.person((0.7, -0.7)) in (0, 1)


def test_a_new_person_is_kept_when_no_limit(lj):
    linker = lj.VoiceLinker(None)
    linker.person((1.0, 0.0))
    assert linker.person((0.0, 1.0)) == 1


def test_one_file_with_steady_labels_and_real_times(plugin, lj, tmp_path, caplog):
    job, env, sent, said, _ = world(plugin, lj, tmp_path, diarize=two_voices(plugin))
    with caplog.at_level(logging.DEBUG):
        run(plugin, lj, job, env)
    assert len(sent) == 1 and not said
    text = sent[0]
    assert "[00:01] Speaker 1" in text and "Speaker 3" not in text
    assert SECRET not in caplog.text and text.rstrip().endswith(plugin.papareo.NOTICE)


def test_times_move_along_and_people_stay_the_same_across_parts(plugin, lj, tmp_path):
    job, env, *_ = world(plugin, lj, tmp_path, diarize=two_voices(plugin))
    seen = {}

    def spy(cues, text, diarized, hints):
        seen.update(cues=cues, turns=diarized.turns)
        return text, cues, False

    lj.run_job(job, env, spy, "Transcript")
    starts = [c.start for c in seen["cues"]]
    assert starts == [job_start + 1 for job_start, _ in job.parts]
    turns = seen["turns"]
    assert turns[2].start == job.parts[1][0]
    # The fake swaps its local numbers in every second part: the people must not swap.
    assert [t.speaker for t in turns[:4]] == [0, 1, 0, 1]


def test_the_job_folder_is_gone_when_done(plugin, lj, tmp_path):
    job, env, *_ = world(plugin, lj, tmp_path, diarize=two_voices(plugin))
    run(plugin, lj, job, env)
    assert not (lj.jobs_home() / job.id).exists()


def test_nothing_is_paid_for_if_separation_fails(plugin, lj, tmp_path):
    def broken(*a):
        raise plugin.speakers.DiarizeError("speaker separation failed", True)

    client = FakeClient(plugin)
    job, env, sent, said, used = world(plugin, lj, tmp_path, diarize=broken, papareo=client)
    run(plugin, lj, job, env)
    assert client.started == [] and not sent and "nothing was spent" in said[0] and used == [1]
    assert not (lj.jobs_home() / job.id).exists()


def test_a_silent_part_does_not_stop_the_job(plugin, lj, tmp_path):
    sp = plugin.speakers
    diarize = two_voices(plugin)
    seen = []

    def some_silent(path, count, samples, seconds, want):
        seen.append(path)
        if len(seen) == 1:
            raise sp.NoSpeech("no speech", True)
        return diarize(path, count, samples, seconds, want)

    job, env, sent, said, _ = world(plugin, lj, tmp_path, diarize=some_silent)
    run(plugin, lj, job, env)
    assert len(sent) == 1 and not said


def test_voice_clips_go_before_any_money_is_spent(plugin, lj, tmp_path):
    order = []
    client = FakeClient(plugin)
    original = client.start_task
    client.start_task = lambda p: (order.append("paid"), original(p))[1]
    job, env, *_ = world(plugin, lj, tmp_path, diarize=two_voices(plugin), papareo=client, samples={"Aroha": "x"})
    env.samples_used = lambda: order.append("clips gone")
    run(plugin, lj, job, env)
    assert order[0] == "clips gone" and order[1] == "paid"


def test_resuming_never_pays_twice(plugin, lj, tmp_path):
    lost = plugin.papareo.PaidButLost("lost")
    client = FakeClient(plugin, fail={"task-1": lost})
    job, env, sent, said, _ = world(plugin, lj, tmp_path, diarize=two_voices(plugin), papareo=client)
    run(plugin, lj, job, env)
    assert not sent and "resume" in said[0]
    assert len(client.started) == 2

    again = lj.latest_for("signal", "chat")
    assert again.tasks == ["task-0", "task-1", None] or again.tasks == ["task-0", "task-1"]
    env.say = said.append
    diarize_must_not_run = lj.Env(**{**env.__dict__, "diarize": None})
    run(plugin, lj, again, diarize_must_not_run)
    assert len(client.started) == len(again.parts) and len(sent) == 1


def test_a_task_papa_reo_failed_is_uploaded_again(plugin, lj, tmp_path):
    client = FakeClient(plugin, fail={"task-0": plugin.papareo.TaskFailed("failed")})
    job, env, sent, said, _ = world(plugin, lj, tmp_path, speakers=False, count=None, papareo=client)
    run(plugin, lj, job, env)
    assert lj.latest_for("signal", "chat").tasks[0] is None


def test_a_failed_send_keeps_the_job_and_resending_is_free(plugin, lj, tmp_path):
    client = FakeClient(plugin)
    job, env, sent, said, _ = world(plugin, lj, tmp_path, diarize=two_voices(plugin), papareo=client)

    def broken(*a):
        raise RuntimeError(SECRET)

    good = env.attach
    env.attach = broken
    run(plugin, lj, job, env)
    assert "check if it arrived" in said[0] and SECRET not in said[0]
    uploads = len(client.started)
    env.attach = good
    run(plugin, lj, lj.latest_for("signal", "chat"), env)
    assert len(sent) == 1 and len(client.started) == uploads


def test_no_text_or_voice_is_ever_on_disk(plugin, lj, tmp_path):
    client = FakeClient(plugin, fail={"task-1": plugin.papareo.PaidButLost("x")})
    job, env, *_ = world(plugin, lj, tmp_path, diarize=two_voices(plugin), papareo=client)
    run(plugin, lj, job, env)
    saved = (lj.jobs_home() / job.id / lj.JOB_FILE).read_text()
    assert SECRET not in saved and "voices" not in saved


def test_the_folder_is_private(plugin, lj, tmp_path):
    job, *_ = world(plugin, lj, tmp_path)
    assert (lj.jobs_home() / job.id).stat().st_mode & 0o077 == 0


def test_old_jobs_are_swept(plugin, lj, tmp_path):
    job, *_ = world(plugin, lj, tmp_path)
    lj.sweep(now=lambda: 10**12)
    assert not (lj.jobs_home() / job.id).exists()


def test_another_chats_job_is_not_resumed(plugin, lj, tmp_path):
    world(plugin, lj, tmp_path)
    assert lj.latest_for("signal", "someone-else") is None


def test_only_one_job_at_a_time(lj):
    held = []
    assert lj.start(lambda: None, spawn=held.append)
    assert lj.is_busy() and not lj.start(lambda: None, spawn=held.append)
    held[0]()
    assert not lj.is_busy()


def test_refusals_come_before_any_spend(lj, tmp_path):
    f = tmp_path / "a.m4a"
    f.write_bytes(b"x")
    ask = lj.refusal(str(f), 54000.0, True, None, object(), True)
    assert "how many people" in ask
    assert "attached directly" in lj.refusal(str(f), 54000.0, False, None, None, False)
    assert "hours" in lj.refusal(str(f), lj.MAX_JOB_S + 1, False, None, None, True)
    assert "not installed" in lj.refusal(str(f), 54000.0, True, 4, None, True)
    assert lj.refusal(str(f), 54000.0, True, 4, object(), True) is None


def test_it_asks_before_spending(lj):
    answer = lj.confirmation(15 * 3600.0)
    assert answer["needs_agreement"] and answer["hours"] == 15.0 and answer["parts"] == 30
    assert "long_job_ok" in answer["note"] and answer["sent"] is False


@pytest.fixture
def tool(plugin, lj, monkeypatch):
    registered = []

    class Ctx:
        def register_tool(self, **kw):
            registered.append(kw)

    monkeypatch.setenv("PAPAREO_API_KEY", "k")
    monkeypatch.setenv("PAPAREO_DIRECT_PLATFORMS", "signal")
    monkeypatch.setattr(plugin, "_bridge_problem", lambda: "no bridge")
    monkeypatch.setattr(plugin.deliver, "session_platform", lambda: "signal")
    monkeypatch.setattr(plugin, "probe_duration", lambda path: 15 * 3600.0)
    monkeypatch.setattr(plugin.Diarizer, "why_not", lambda self: None)
    plugin.register(Ctx())
    return registered[0]["handler"]


def call(tool, path, **kw):
    import json

    return json.loads(tool({"path": str(path), "engine": "papareo", "chosen_by_person": True, **kw}))


def test_tool_asks_before_starting_and_starts_nothing(tool, lj, tmp_path):
    f = tmp_path / "r.m4a"
    f.write_bytes(b"x")
    answer = call(tool, f)
    assert answer["needs_agreement"] and answer["sent"] is False and not lj.jobs_home().exists()


def test_tool_asks_how_many_people_first(tool, tmp_path):
    f = tmp_path / "r.m4a"
    f.write_bytes(b"x")
    assert "how many people" in call(tool, f, speakers=True, long_job_ok=True)["error"]


def test_tool_starts_once_agreed_and_keeps_the_original(plugin, tool, lj, tmp_path, monkeypatch):
    f = tmp_path / "r.m4a"
    f.write_bytes(b"x")
    monkeypatch.setattr(plugin.deliver, "bind", lambda: ("signal", "chat", lambda *a: None, lambda m: None))
    monkeypatch.setattr(lj, "silences", lambda *a: [])
    monkeypatch.setattr(lj, "start", lambda work, spawn=None: True)
    answer = call(tool, f, long_job_ok=True)
    assert answer["started"] is True and answer["parts"] == 30 and f.exists()
    assert lj.latest_for("signal", "chat") is not None


def test_tool_has_nothing_to_resume(plugin, tool, monkeypatch):
    monkeypatch.setattr(plugin.deliver, "bind", lambda: ("signal", "chat", lambda *a: None, lambda m: None))
    assert "no stopped long job" in call(tool, "", resume_job=True)["error"]


def test_a_short_recording_is_not_a_job(plugin, tool, lj, tmp_path, monkeypatch):
    f = tmp_path / "r.m4a"
    f.write_bytes(b"x")
    monkeypatch.setattr(plugin, "probe_duration", lambda path: 60.0)
    assert "needs_agreement" not in call(tool, f)
