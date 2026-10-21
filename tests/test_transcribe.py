"""transcribe_recording's flow and its failure rules (PRD B-5)."""

from __future__ import annotations

import json
import logging
import secrets
import types
from datetime import datetime

import pytest

FILENAME = "transcript-20261016-1530.txt"


class FakeVault:
    def __init__(self, destinations=("signal",), up=True, put_fails=False, folder=False, export_fails=False):
        self._dest = destinations
        self._up = up
        self._put_fails = put_fails
        self._folder = folder
        self._export_fails = export_fails
        self.puts: list[dict] = []
        self.exports: list[tuple] = []

    def health(self):
        if not self._up:
            raise OSError("no socket")
        return {"status": "ok"}

    def destinations(self, vault):
        out = [types.SimpleNamespace(name=n, kind="send") for n in self._dest]
        return out + ([types.SimpleNamespace(name="folder", kind="fs")] if self._folder else [])

    def export(self, handle, destination, filename):
        if self._export_fails:
            raise RuntimeError("EXISTS")
        self.exports.append((handle, destination, filename))

    def put(self, body, vault, **kw):
        if self._put_fails:
            raise RuntimeError("put failed")
        self.puts.append({"body": body, "vault": vault, **kw})
        return types.SimpleNamespace(handle="v_0123456789ab")


class FakePapareo:
    def __init__(self, configured=True, text=None, error=None):
        self.configured = configured
        self.text = text or ("Tēnā koutou. " + secrets.token_hex(16))
        self.error = error
        self.calls = 0

    def why_not(self):
        return "" if self.configured else "no Papa Reo key is configured on this agent"

    def transcribe(self, path, duration):
        self.calls += 1
        self.duration = duration
        if self.error:
            raise self.error
        return self.text


class World:
    """Fakes for every dependency, with a record of what happened."""

    def __init__(self, plugin, *, text="Just a quick English note about Thursday.", language="en",
                 escalate=None, papareo=None, vault=None, bridge=True, platform="signal",
                 send_result=None, local_fails=False, detector_fails=False, is_audio=None,
                 compliant=False, direct=(), local_send_fails=False):
        self.plugin = plugin
        self.direct = frozenset(direct)
        self.local_send_fails = local_send_fails
        self.local_sent: list[tuple] = []
        self.handle_sent: list[tuple] = []
        self.papareo = papareo or FakePapareo()
        self.vault_ = vault or FakeVault()
        self.send_result = send_result or {"sent": True, "handle": "v_0123456789ab"}
        self.escalate = escalate
        self.local_result = plugin.local.LocalResult(text=text, language=language, language_probability=0.9, duration=42.0, model="large-v3-turbo")
        self.local_fails = local_fails
        self.local_calls = 0
        self.is_audio = is_audio
        self.compliant = compliant
        self.detector_fails = detector_fails
        self.bridge = bridge
        self.platform_ = platform

    def deps(self):
        p = self.plugin

        def local(path):
            self.local_calls += 1
            if self.local_fails:
                raise RuntimeError("whisper fell over")
            return self.local_result

        # escalate=None and no detector_fails: no detector plugged in (today's
        # plugin). Otherwise a stand-in for may_contain_mi through Deps.detect.
        def detect(text, language):
            if self.detector_fails:
                raise ValueError("bug")
            return p.transcribe.Decision(self.escalate, ("forced",) if self.escalate else (), 0.0)

        no_detector = self.escalate is None and not self.detector_fails
        return p.transcribe.Deps(
            local=local,
            duration=lambda path: 400.0,
            is_audio=self.is_audio,
            share_papareo_text=self.compliant,
            detect=None if no_detector else detect,
            papareo=self.papareo,
            vault=lambda: self.vault_,
            send_local=self._send_local,
            direct_platforms=self.direct,
            send_handle=lambda h, f, c: (self.handle_sent.append((h, f, c)), self.send_result)[1],
            platform=lambda: self.platform_,
            bridge_problem=lambda: (
                None if self.bridge is True
                else "protected outputs (ovenmitt) are not set up on this agent" if self.bridge is False
                else self.bridge
            ),
            now=lambda: datetime(2026, 10, 16, 15, 30),
            filename_for=lambda path, when: FILENAME,
        )

    def _send_local(self, f, t, c):
        if self.local_send_fails:
            raise RuntimeError("adapter refused")
        self.local_sent.append((f, t, c))

    def run(self, path, engine="auto"):
        return self.plugin.transcribe.run(str(path), engine, self.deps())


@pytest.fixture
def memo(tmp_path):
    p = tmp_path / "audio_0123456789ab.ogg"
    p.write_bytes(b"OggS" + secrets.token_bytes(64))
    return p


def assert_no_papareo_text(result, world):
    assert world.papareo.text not in json.dumps(result, ensure_ascii=False)


def test_english_goes_local_and_the_agent_sees_the_text(plugin, memo):
    w = World(plugin)
    r = w.run(memo, engine="local")
    src = "whisper large-v3-turbo, run locally"
    assert r == {"engine": "local", "transcribed_by": src, "filename": FILENAME, "text": w.local_result.text,
                 "next": plugin.transcribe.NEXT_LOCAL, "sent": True}
    assert "Don't summarise" in r["next"]
    assert w.local_sent == [
        (FILENAME, f"{w.local_result.text}\n\nTranscribed by {src}.\n", f"Transcript ({src})")
    ], "the file and its caption say what made it"
    assert w.papareo.calls == 0 and w.vault_.puts == [] and w.handle_sent == []


def test_without_a_detector_the_agent_must_ask_first(plugin, memo):
    w = World(plugin, text="Kia ora, tēnā koutou katoa.", language="mi")
    r = w.run(memo)  # engine="auto"
    assert r["sent"] is False and r["error"].startswith("Nothing was transcribed. Ask the person first")
    assert w.local_calls == 0 and w.papareo.calls == 0 and w.local_sent == []


def test_papareo_never_runs_whisper(plugin, memo):
    w = World(plugin)
    r = w.run(memo, engine="papareo")
    assert r["sent"] is True and r["engine"] == "papareo"
    assert w.local_calls == 0 and w.local_sent == []
    assert w.papareo.duration == 400.0, "length comes from the probe, for the long endpoint"


def test_papareo_failure_without_whisper_offers_local_instead(plugin, memo):
    w = World(plugin, papareo=FakePapareo(error=plugin.papareo.PapareoError("Papa Reo timed out")))
    r = w.run(memo, engine="papareo")
    assert r["sent"] is False and r["engine"] == "none"
    assert "engine='local'" in r["note"] and "Ask again to retry Papa Reo" in r["note"]
    assert w.local_calls == 0 and w.local_sent == [] and w.papareo.calls == 1


def test_papareo_preflight_failure_spends_nothing_and_runs_nothing(plugin, memo):
    w = World(plugin, platform="discord")
    r = w.run(memo, engine="papareo")
    assert r["engine"] == "none" and "nothing was spent" in r["note"]
    assert w.papareo.calls == 0 and w.local_calls == 0


def test_flagged_recording_goes_to_papareo_vault_and_bridge(plugin, memo):
    w = World(plugin, text="Kia ora, tēnā koutou katoa.", language="en", escalate=True)
    r = w.run(memo)
    assert r == {"sent": True, "engine": "papareo", "transcribed_by": "Papa Reo, Te Hiku Media",
                 "handle": "v_0123456789ab", "filename": FILENAME, "next": plugin.transcribe.NEXT_PAPAREO}
    # The footer is the licence notice; the agent names Papa Reo every time and
    # explains the licence only when it's asked about or about to be broken.
    assert "Papa Reo transcript" in r["next"]
    assert "don't need an explicit reminder" in r["next"]
    assert w.papareo.calls == 1
    [put] = w.vault_.puts
    assert put["vault"] == "papareo"
    assert put["name"] == FILENAME and put["labels"] == [FILENAME]
    assert put["mime"].startswith("text/plain")
    body = put["body"].decode()
    assert body.startswith(w.papareo.text)
    assert body.endswith(
        "This text transcribed by Papa Reo API. Please be sure to comply with the terms of the "
        "Kaitiakitanga license in use of this material.\n" + plugin.papareo.NOTICE + "\n"
    )
    assert w.handle_sent == [("v_0123456789ab", FILENAME, "Transcript (Papa Reo, Te Hiku Media)")]
    assert w.local_sent == []
    assert_no_papareo_text(r, w)


def test_engine_local_overrides_the_detector(plugin, memo):
    w = World(plugin, text="Tēnā koutou katoa", language="mi", escalate=True)
    r = w.run(memo, engine="local")
    assert r["engine"] == "local" and w.papareo.calls == 0


def test_engine_papareo_overrides_the_detector(plugin, memo):
    w = World(plugin, escalate=False)
    r = w.run(memo, engine="papareo")
    assert r["engine"] == "papareo" and w.papareo.calls == 1


@pytest.mark.parametrize(
    ("world_kwargs", "why"),
    [
        ({"papareo": FakePapareo(configured=False)}, "no Papa Reo key"),
        ({"bridge": False}, "not set up"),
        ({"vault": FakeVault(destinations=("discord",))}, "not allowed protected files on signal"),
        ({"vault": FakeVault(up=False)}, "not reachable"),
        ({"platform": None}, "no chat"),
    ],
)
def test_preflight_failure_spends_nothing(plugin, memo, world_kwargs, why):
    w = World(plugin, escalate=True, **world_kwargs)
    r = w.run(memo)
    assert w.papareo.calls == 0
    assert r["engine"] == "local" and r["sent"] is True
    assert why in r["note"]
    assert w.local_sent and why in w.local_sent[0][2], "the user sees the note on the file"


def test_papareo_failure_falls_back_once_without_retry(plugin, memo):
    w = World(plugin, escalate=True, papareo=FakePapareo(error=plugin.papareo.PapareoError("Papa Reo timed out")))
    r = w.run(memo)
    assert w.papareo.calls == 1
    assert r["engine"] == "local" and "Ask again" in r["note"]
    assert w.vault_.puts == []


def test_put_failure_says_it_was_paid_for(plugin, memo):
    w = World(plugin, escalate=True, vault=FakeVault(put_fails=True))
    r = w.run(memo)
    assert r["engine"] == "local" and r["papareo_charged"] is True
    assert "nothing was kept" in r["note"]
    assert w.handle_sent == []
    assert_no_papareo_text(r, w)


def test_refused_send_keeps_the_object_for_a_resend(plugin, memo):
    refused = {"error": {"code": "DESTINATION_NOT_FOUND", "message": "x"}}
    w = World(plugin, escalate=True, send_result=refused)
    r = w.run(memo)
    assert r["sent"] is False and r["handle"] == "v_0123456789ab"
    assert "DESTINATION_NOT_FOUND" in r["note"] and "ovenmitt_vault_send" in r["note"]
    assert_no_papareo_text(r, w)


@pytest.mark.parametrize(
    "send",
    [
        "raises",
        {"error": "Tool execution failed: AttributeError: boom"},  # the harness's shape
        {"error": {"code": "NOT_CONFIGURED", "message": "x"}},
        None,
    ],
)
def test_after_paying_the_agent_always_leaves_with_the_handle(plugin, memo, send, caplog):
    w = World(plugin, escalate=True)
    if send == "raises":
        def boom(h, f, c):
            raise RuntimeError("send tool missing")
        deps = w.deps()
        deps.send_handle = boom
        with caplog.at_level(logging.DEBUG):
            r = plugin.transcribe.run(str(memo), "auto", deps)
    else:
        w.send_result = send if send is not None else {}
        with caplog.at_level(logging.DEBUG):
            r = w.run(memo)
    assert r["sent"] is False and r["handle"] == "v_0123456789ab" and r["papareo_charged"] is True
    assert "ovenmitt_vault_send" in r["note"]
    assert_no_papareo_text(r, w)
    assert w.papareo.text not in caplog.text


def test_bridge_that_cannot_send_stops_the_spend(plugin, memo):
    w = World(plugin, escalate=True, bridge="sending is not configured on this agent")
    r = w.run(memo)
    assert w.papareo.calls == 0
    assert "sending is not configured" in r["note"]


def test_only_send_destinations_count(plugin, memo):
    class ExportOnly(FakeVault):
        def destinations(self, vault):
            return [types.SimpleNamespace(name="signal", kind="fs")]

    w = World(plugin, escalate=True, vault=ExportOnly())
    r = w.run(memo)
    assert w.papareo.calls == 0 and "not allowed protected files on signal" in r["note"]


def test_any_papareo_error_still_gets_the_local_transcript(plugin, memo, caplog):
    w = World(plugin, escalate=True, papareo=FakePapareo(error=RuntimeError("odd failure with secret-ish detail")))
    with caplog.at_level(logging.DEBUG):
        r = w.run(memo)
    assert r["engine"] == "local" and r["sent"] is True and "Ask again" in r["note"]
    assert "secret-ish" not in json.dumps(r) and "secret-ish" not in caplog.text


def test_detector_error_escalates(plugin, memo, caplog):
    w = World(plugin, detector_fails=True)
    with caplog.at_level(logging.ERROR):
        r = w.run(memo)
    assert r["engine"] == "papareo"
    assert "DETECTOR ERROR" in caplog.text


def test_local_failure(plugin, memo):
    assert World(plugin, local_fails=True).run(memo, engine="local")["sent"] is False
    w = World(plugin, local_fails=True, escalate=False)  # with a detector plugged in
    assert w.run(memo)["engine"] == "papareo", "biased to fire when there is nothing to judge"
    w2 = World(plugin, local_fails=True, papareo=FakePapareo(configured=False))
    r = w2.run(memo, engine="papareo")
    assert r["sent"] is False and "engine='local'" in r["note"]


@pytest.mark.parametrize("name", ["missing.ogg", "notes.txt"])
def test_bad_paths_do_nothing(plugin, tmp_path, name):
    p = tmp_path / name
    if name.endswith(".txt"):
        p.write_text("x")
    w = World(plugin)
    r = w.run(p)
    assert r["sent"] is False and "error" in r
    assert w.local_sent == [] and w.papareo.calls == 0


def test_bad_engine(plugin, memo):
    assert "engine must be" in World(plugin).run(memo, engine="cloud")["error"]


def test_papareo_text_never_logged(plugin, memo, caplog):
    w = World(plugin, escalate=True)
    with caplog.at_level(logging.DEBUG):
        w.run(memo)
    assert w.papareo.text not in caplog.text


def test_papareo_transcript_is_also_exported_to_the_folder(plugin, memo):
    w = World(plugin, escalate=True, vault=FakeVault(folder=True))
    r = w.run(memo)
    assert r["sent"] is True
    assert w.vault_.exports == [("v_0123456789ab", "folder", FILENAME)]


def test_no_folder_no_export_and_a_failed_export_still_sends(plugin, memo):
    w = World(plugin, escalate=True)
    assert w.run(memo)["sent"] is True and w.vault_.exports == []
    w = World(plugin, escalate=True, vault=FakeVault(folder=True, export_fails=True))
    assert w.run(memo)["sent"] is True and len(w.handle_sent) == 1


def test_unknown_extension_is_accepted_when_it_holds_audio(plugin, tmp_path):
    p = tmp_path / "attachment-0123.bin"  # how chat platforms often save audio
    p.write_bytes(b"RIFF" + secrets.token_bytes(64))
    w = World(plugin, is_audio=lambda path: True)
    assert w.run(p, engine="local")["sent"] is True


def test_unknown_extension_without_audio_is_refused(plugin, tmp_path):
    p = tmp_path / "attachment-0123.bin"
    p.write_bytes(b"not audio")
    for probe in (None, lambda path: False, lambda path: 1 / 0):
        w = World(plugin, is_audio=probe)
        r = w.run(p, engine="local")
        assert r["sent"] is False and "no audio found" in r["error"] and w.local_calls == 0


def test_declared_compliant_agent_gets_the_papareo_text(plugin, memo, caplog):
    w = World(plugin, compliant=True)
    with caplog.at_level(logging.DEBUG):
        r = w.run(memo, engine="papareo")
    assert r["sent"] is True and r["text"] == w.papareo.text and r["kaitiakitanga_compliant"] is True
    assert w.papareo.text not in caplog.text, "never logged, even when shared"
    [put] = w.vault_.puts
    assert put["body"].decode().startswith(w.papareo.text), "still stored and sent as before"


def test_not_declared_means_no_papareo_text(plugin, memo):
    w = World(plugin)
    r = w.run(memo, engine="papareo")
    assert "text" not in r and "kaitiakitanga_compliant" not in r
    assert_no_papareo_text(r, w)


def test_nothing_runs_unless_the_person_chose(plugin, memo):
    for engine in ("local", "papareo"):
        w = World(plugin)
        r = plugin.transcribe.run(str(memo), engine, w.deps(), chosen_by_person=False)
        assert r["sent"] is False and r["error"].startswith("Nothing was transcribed. Ask the person first")
        assert w.local_calls == 0 and w.papareo.calls == 0 and w.local_sent == []


# --- direct delivery: no ovenmitt, the plugin attaches the file itself -------


def test_direct_platform_sends_papareo_file_without_ovenmitt(plugin, memo):
    w = World(plugin, bridge=False, vault=FakeVault(up=False), direct=("signal",))
    r = w.run(memo, engine="papareo")
    assert r == {"sent": True, "engine": "papareo", "delivery": "direct",
                 "transcribed_by": "Papa Reo, Te Hiku Media", "filename": FILENAME,
                 "next": plugin.transcribe.NEXT_PAPAREO_DIRECT}
    assert_no_papareo_text(r, w)
    assert w.vault_.puts == [] and w.handle_sent == [] and w.local_calls == 0
    [(fname, body, caption)] = w.local_sent
    assert fname == FILENAME and caption == "Transcript (Papa Reo, Te Hiku Media)"
    assert body.startswith(w.papareo.text)
    assert "Kaitiakitanga license" in body and body.rstrip().endswith("Kaitiakitanga License.")


def test_direct_only_on_listed_platforms(plugin, memo):
    w = World(plugin, bridge=False, platform="discord", direct=("signal",))
    r = w.run(memo, engine="papareo")
    assert r["engine"] == "none" and "nothing was spent" in r["note"]
    assert w.papareo.calls == 0 and w.local_sent == []


def test_direct_still_needs_the_key(plugin, memo):
    w = World(plugin, bridge=False, papareo=FakePapareo(configured=False), direct=("signal",))
    r = w.run(memo, engine="papareo")
    assert "no Papa Reo key" in r["note"] and w.papareo.calls == 0


def test_direct_shares_text_only_when_compliant(plugin, memo):
    w = World(plugin, bridge=False, direct=("signal",), compliant=True)
    r = w.run(memo, engine="papareo")
    assert r["text"] == w.papareo.text.strip() and r["kaitiakitanga_compliant"] is True


def test_direct_send_failure_says_it_was_paid_and_keeps_no_copy(plugin, memo, caplog):
    w = World(plugin, bridge=False, direct=("signal",), local_send_fails=True)
    with caplog.at_level(logging.DEBUG):
        r = w.run(memo, engine="papareo")
    assert r["sent"] is False and r["papareo_charged"] is True and r["delivery"] == "direct"
    assert "couldn't be attached" in r["note"]
    assert_no_papareo_text(r, w)
    assert w.papareo.text not in caplog.text


def test_ovenmitt_wins_when_both_are_available(plugin, memo):
    w = World(plugin, direct=("signal",))
    r = w.run(memo, engine="papareo")
    assert r["handle"] == "v_0123456789ab" and "delivery" not in r
    assert w.local_sent == []
