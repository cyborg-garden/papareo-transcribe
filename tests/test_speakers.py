"""Paragraphs and speaker labels: arithmetic on timings, and nothing kept."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def sp(plugin):
    return plugin.speakers


def test_srt_milliseconds_are_not_padded(sp):
    # Papa Reo writes 35 ms as ",35". Read as a fraction it would be 350 ms.
    cues = sp.parse_srt("1\n00:00:00,35 --> 00:00:01,385\nKia ora\n\n2\n00:00:13,15 --> 00:00:13,905\nāe\n")
    assert [(c.start, c.end, c.text) for c in cues] == [(0.035, 1.385, "Kia ora"), (13.015, 13.905, "āe")]


def test_srt_garbage_is_no_cues(sp):
    assert sp.parse_srt("not subtitles at all") == []
    assert sp.parse_srt("") == []


def test_paragraphs_without_speakers(sp):
    C = sp.Cue
    text = sp.render([C(0, 1, "one"), C(1, 2, "two"), C(5, 6, "three")])
    assert text == "one two\n\nthree"
    assert "Speaker" not in text and sp.LABEL_NOTE not in text


def test_long_runs_are_broken_up(sp):
    C = sp.Cue
    cues = [C(i, i + 1, "word " * 40 + "end.") for i in range(6)]
    assert sp.render(cues).count("\n\n") >= 1


def test_speakers_are_labelled_in_order_heard(sp):
    C, T = sp.Cue, sp.Turn
    d = sp.Diarized((T(0, 2, 7), T(2, 4, 3), T(4, 6, 7)))
    text = sp.render([C(0, 2, "a"), C(2, 4, "b"), C(4, 6, "c")], d)
    assert text == f"{sp.LABEL_NOTE}\n\n[00:00] Speaker 1\na\n\n[00:02] Speaker 2\nb\n\n[00:04] Speaker 1\nc"
    assert sp.speaker_count([C(0, 2, "a"), C(2, 4, "b")], d) == 2


def test_names_replace_numbers(sp):
    C, T = sp.Cue, sp.Turn
    d = sp.Diarized((T(0, 2, 0), T(2, 4, 1)), {1: "Aroha"})
    text = sp.render([C(0, 2, "a"), C(2, 4, "b")], d)
    assert "[00:00] Speaker 1\na" in text and "[00:02] Aroha\nb" in text


def test_cue_goes_to_the_speaker_with_most_overlap(sp):
    C, T = sp.Cue, sp.Turn
    turns = (T(0, 1, 0), T(1, 9, 1))
    assert sp._speaker_for(C(0.5, 5, "x"), turns) == 1
    assert sp._speaker_for(C(9.5, 10, "x"), turns) == 1  # just outside: the nearest
    assert sp._speaker_for(C(50, 51, "x"), turns) is None


def test_hour_clock(sp):
    assert sp._clock(3725) == "1:02:05"


def _home(tmp_path: Path) -> Path:
    (tmp_path / "venv/bin").mkdir(parents=True)
    (tmp_path / "venv/bin/python").write_text("")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/seg.onnx").write_text("")
    (tmp_path / "models/emb.onnx").write_text("")
    return tmp_path


def _fake_run(seen, answer, code=0):
    def run(cmd, **kw):
        seen.append((cmd, kw))
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        return subprocess.CompletedProcess(cmd, code, json.dumps(answer).encode(), b"")

    return run


def test_worker_gets_a_clean_environment_and_leaves_nothing(sp, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-travel")
    seen = []
    answer = {"turns": [{"start": 0, "end": 1, "speaker": 0}], "names": {"0": "Aroha"}, "unmatched": ["Tama"]}
    d = sp.Diarizer(str(_home(tmp_path)), run=_fake_run(seen, answer))
    got = d("/x.wav", 2, {"Aroha": "/m.ogg", "Tama": "/r.ogg"}, 10.0)
    assert got.names == {0: "Aroha"} and got.unmatched == ("Tama",)
    tmpdirs = set()
    for cmd, kw in seen:
        assert "OPENROUTER_API_KEY" not in kw["env"]
        tmpdirs.add(kw["env"]["HOME"])
    assert len(tmpdirs) == 1 and not Path(tmpdirs.pop()).exists()
    request = json.loads(seen[-1][1]["input"])
    assert request["num_speakers"] == 2 and set(request["samples"]) == {"Aroha", "Tama"}


def test_worker_output_never_reaches_the_error(sp, tmp_path):
    d = sp.Diarizer(str(_home(tmp_path)), run=_fake_run([], {}, code=1))
    with pytest.raises(sp.DiarizeError, match="speaker separation failed"):
        d("/x.wav")


def test_not_installed_says_so(sp, tmp_path):
    with pytest.raises(sp.DiarizeError, match="not installed"):
        sp.Diarizer(str(tmp_path))("/x.wav")


def test_no_speech_is_an_error(sp, tmp_path):
    d = sp.Diarizer(str(_home(tmp_path)), run=_fake_run([], {"turns": []}))
    with pytest.raises(sp.DiarizeError, match="no speech"):
        d("/x.wav")


def test_srt_keeps_every_word_whatever_the_blank_lines(sp):
    body = "1\n00:00:01,0 --> 00:00:02,0\none\n\ntwo\n3\n00:00:02,0 --> 00:00:03,0\nthree\n"
    assert [c.text for c in sp.parse_srt(body)] == ["one two", "three"]


def test_srt_reversed_cue_keeps_its_words(sp):
    assert sp.parse_srt("00:00:05,0 --> 00:00:04,0\nwords\n")[0].text == "words"


def test_same_words_is_exact(sp):
    cues = [sp.Cue(0, 1, "a b"), sp.Cue(1, 2, "c")]
    assert sp.same_words(cues, "a  b\nc\n")
    assert not sp.same_words(cues, "a b c d") and not sp.same_words(cues, "a c b")


def test_split_gives_each_speaker_their_own_words(sp):
    C, T = sp.Cue, sp.Turn
    cue = C(0, 10, "one two three four five. six seven eight nine ten")
    got = sp.split_at_turns([cue], [T(0, 5, 0), T(5, 10, 1)])
    assert [c.text for c in got] == ["one two three four five.", "six seven eight nine ten"]


def test_a_sentence_is_not_cut_for_a_voice_change_inside_it(sp):
    C, T = sp.Cue, sp.Turn
    cue = C(0, 10, "and it would be good for all of our people here")
    got = sp.split_at_turns([cue], [T(0, 5, 0), T(5, 10, 1)])
    assert [c.text for c in got] == [cue.text]


def test_a_split_moves_to_the_nearest_sentence_end(sp):
    C, T = sp.Cue, sp.Turn
    cue = C(0, 12, "a b c d. e f g h i j k l")
    got = sp.split_at_turns([cue], [T(0, 6, 0), T(6, 12, 1)])
    assert [c.text for c in got] == ["a b c d.", "e f g h i j k l"]


def test_a_very_long_cue_with_no_sentence_end_can_still_split(sp):
    C, T = sp.Cue, sp.Turn
    words = [f"w{i}" for i in range(sp.LONG_CUE_WORDS)]
    got = sp.split_at_turns([C(0, 30, " ".join(words))], [T(0, 15, 0), T(15, 30, 1)])
    assert len(got) == 2 and " ".join(c.text for c in got) == " ".join(words)


def test_a_snap_cannot_jump_over_the_next_voice_change(sp):
    C, T = sp.Cue, sp.Turn
    words = [f"w{i}" for i in range(40)]
    words[17] += "."
    got = sp.split_at_turns([C(0, 40, " ".join(words))], [T(0, 11, 0), T(11, 14, 1), T(14, 40, 2)])
    assert [len(c.text.split()) for c in got] == [11, 7, 22] and got[1].text.endswith("w17.")


def test_a_brief_voice_change_goes_to_its_neighbour(sp):
    T = sp.Turn
    assert sp.steady([T(0, 10, 0), T(10, 11, 1), T(11, 20, 0)]) == (T(0, 20, 0),)


def test_flapping_settles_into_turns(sp):
    C, T = sp.Cue, sp.Turn
    flap = [T(i, i + 1, i % 2) for i in range(6)] + [T(6, 12, 1)]
    d = sp.Diarized(tuple(flap))
    got = sp.split_at_turns([C(0, 12, "a b c d e f g h")], d.turns)
    assert sp.render(got, d).count("Speaker") == 2  # the note's mention plus one label
    assert " ".join(c.text for c in got) == "a b c d e f g h"


def test_split_follows_the_hints_over_a_steady_pace(sp):
    C, T = sp.Cue, sp.Turn
    cue = C(0, 10, "a b c. d e f g h. i j")
    hints = [C(0, 5, "a b c. d e f g h."), C(5, 10, "i j")]
    got = sp.split_at_turns([cue], [T(0, 5, 0), T(5, 10, 1)], hints)
    assert [c.text for c in got] == ["a b c. d e f g h.", "i j"]


def test_split_ignores_a_tiny_overlap(sp):
    C, T = sp.Cue, sp.Turn
    got = sp.split_at_turns([C(0, 10, "a b c")], [T(0, 9.8, 0), T(9.8, 10, 1)])
    assert [c.text for c in got] == ["a b c"]


def test_a_lopsided_hint_cannot_erase_a_speaker(sp):
    C, T = sp.Cue, sp.Turn
    cue = C(0, 10, "a b c d e. f g h i j")
    for hints in ([C(0, 10, "a b c d e f g h i j")], [C(0, 2, "a"), C(2, 4, "b")]):
        got = sp.split_at_turns([cue], [T(0, 5, 0), T(5, 10, 1)], hints)
        assert len(got) == 2 and all(q.text for q in got)


def test_a_short_turn_in_a_sparse_cue_keeps_a_word(sp):
    C, T = sp.Cue, sp.Turn
    got = sp.split_at_turns([C(0, 20, "a b c d e")], [T(0, 1, 0), T(1, 20, 1)])
    assert [q.text for q in got] == ["a b c d e"]


def test_more_people_than_words_keeps_every_word_once(sp):
    C, T = sp.Cue, sp.Turn
    got = sp.split_at_turns([C(0, 9, "a b")], [T(0, 3, 0), T(3, 5, 1), T(5, 9, 2)])
    assert " ".join(q.text for q in got) == "a b"


def test_an_interruption_inside_a_long_turn_is_heard(sp):
    C, T = sp.Cue, sp.Turn
    d = sp.Diarized((T(0, 10, 0), T(2, 8, 1)))
    assert sp.flatten(d.turns) == (T(0, 2, 0), T(2, 8, 1), T(8, 10, 0))
    got = sp.split_at_turns([C(0, 10, "a b. c d e f g h. i j")], d.turns)
    text = sp.render(got, d)
    assert sp.speaker_count(got, d) == 2 and text.count("Speaker 1\n") == 2 and text.count("Speaker 2\n") == 1


def test_flatten_handles_touching_and_identical_turns(sp):
    T = sp.Turn
    assert sp.flatten([T(0, 1, 0), T(1, 2, 0)]) == (T(0, 2, 0),)
    assert sp.flatten([T(0, 1, 0), T(0, 1, 1)]) == (T(0, 1, 1),)
    assert sp.flatten([]) == ()


@pytest.mark.parametrize("seed", range(40))
def test_splitting_never_loses_or_reorders_a_word(sp, seed):
    import random

    rng = random.Random(seed)
    C, T = sp.Cue, sp.Turn
    t, turns = 0.0, []
    for _ in range(rng.randint(1, 8)):
        d = rng.uniform(0.1, 6)
        turns.append(T(t, t + d, rng.randint(0, 3)))
        t += d + rng.uniform(0, 1)
    cues, n, c = [], 0, 0.0
    for _ in range(rng.randint(1, 6)):
        d = rng.uniform(0.5, 9)
        k = rng.randint(1, 12)
        cues.append(C(c, c + d, " ".join(str(n + i) for i in range(k))))
        n += k
        c += d
    hints = [C(rng.uniform(0, t), rng.uniform(0, t), "x y") for _ in range(rng.randint(0, 5))]
    got = sp.split_at_turns(cues, turns, hints)
    assert " ".join(q.text for q in got).split() == [str(i) for i in range(n)]


def test_a_person_named_like_a_number_is_not_doubled(sp):
    C, T = sp.Cue, sp.Turn
    d = sp.Diarized((T(0, 2, 0), T(2, 4, 1), T(4, 6, 2)), {1: "Speaker 2"})
    text = sp.render([C(0, 2, "a"), C(2, 4, "b"), C(4, 6, "c")], d)
    assert text.count("Speaker 2\n") == 1 and "Speaker 1\n" in text and "Speaker 3\n" in text


def test_only_audio_containers_may_be_opened(sp):
    allowed = sp.AUDIO_DEMUXERS.split(",")
    assert "hls" not in allowed and "concat" not in allowed and "dash" not in allowed


def test_bracketed_text_keeps_its_label(sp):
    C, T = sp.Cue, sp.Turn
    d = sp.Diarized((T(0, 2, 0), T(5, 9, 1)))
    text = sp.render([C(0, 2, "[Music]"), C(5, 9, "speaker two")], d)
    assert "[00:00] Speaker 1\n[Music]" in text and "[00:05] Speaker 2\nspeaker two" in text


def test_the_worker_stage_decides_whether_clips_were_used(sp, tmp_path):
    def decode_fails(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, b"", b"")

    with pytest.raises(sp.DiarizeError) as e:
        sp.Diarizer(str(_home(tmp_path)), run=decode_fails)("/x.wav")
    assert e.value.clips_used is False
    with pytest.raises(sp.DiarizeError) as e:
        sp.Diarizer(str(tmp_path), run=_fake_run([], {}, code=1))("/x.wav")
    assert e.value.clips_used is True


def test_very_deep_json_is_no_segments(sp):
    assert sp.parse_segments("[" * 200000) == ()
