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
    def __init__(self, destinations=("signal",), up=True, put_fails=False):
        self._dest = destinations
        self._up = up
        self._put_fails = put_fails
        self.puts: list[dict] = []

    def health(self):
        if not self._up:
            raise OSError("no socket")
        return {"status": "ok"}

    def destinations(self, vault):
        return [types.SimpleNamespace(name=n, kind="send") for n in self._dest]

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
        if self.error:
            raise self.error
        return self.text


class World:
    """Fakes for every dependency, with a record of what happened."""

    def __init__(self, plugin, *, text="Just a quick English note about Thursday.", language="en",
                 escalate=None, papareo=None, vault=None, bridge=True, platform="signal",
                 send_result=None, local_fails=False, detector_fails=False):
        self.plugin = plugin
        self.local_sent: list[tuple] = []
        self.handle_sent: list[tuple] = []
        self.papareo = papareo or FakePapareo()
        self.vault_ = vault or FakeVault()
        self.send_result = send_result or {"sent": True, "handle": "v_0123456789ab"}
        self.escalate = escalate
        self.local_result = plugin.local.LocalResult(text=text, language=language, language_probability=0.9, duration=42.0)
        self.local_fails = local_fails
        self.detector_fails = detector_fails
        self.bridge = bridge
        self.platform_ = platform

    def deps(self):
        p = self.plugin

        def local(path):
            if self.local_fails:
                raise RuntimeError("whisper fell over")
            return self.local_result

        def detect(text, language):
            if self.detector_fails:
                raise ValueError("bug")
            if self.escalate is None:
                return p.detector.decide(text, language, p.detector.TrigramModel.load())
            return p.detector.Decision(self.escalate, ("forced",) if self.escalate else (), 0.0)

        return p.transcribe.Deps(
            local=local,
            detect=detect,
            papareo=self.papareo,
            vault=lambda: self.vault_,
            send_local=lambda f, t, c: self.local_sent.append((f, t, c)),
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
    r = w.run(memo)
    assert r == {"engine": "local", "filename": FILENAME, "text": w.local_result.text, "sent": True}
    assert w.local_sent == [(FILENAME, w.local_result.text, "Transcript")]
    assert w.papareo.calls == 0 and w.vault_.puts == [] and w.handle_sent == []


def test_maori_goes_to_papareo_vault_and_bridge(plugin, memo):
    w = World(plugin, text="Kia ora, tēnā koutou katoa.", language="en")
    r = w.run(memo)
    assert r == {"sent": True, "engine": "papareo", "handle": "v_0123456789ab", "filename": FILENAME}
    assert w.papareo.calls == 1
    [put] = w.vault_.puts
    assert put["vault"] == "private"
    assert put["name"] == FILENAME and put["labels"] == [FILENAME]
    assert put["mime"].startswith("text/plain")
    body = put["body"].decode()
    assert body.startswith(w.papareo.text)
    assert body.endswith(
        "This text transcribed by Papa Reo API. Please be sure to comply with the terms of the "
        "Kaitiakitanga license in use of this material.\n"
    )
    assert w.handle_sent == [("v_0123456789ab", FILENAME, "Transcript")]
    assert w.local_sent == []
    assert_no_papareo_text(r, w)


def test_engine_local_overrides_the_detector(plugin, memo):
    w = World(plugin, text="Tēnā koutou katoa", language="mi")
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
    w = World(plugin, local_fails=True)
    assert w.run(memo)["engine"] == "papareo", "biased to fire when there is nothing to judge"
    w2 = World(plugin, local_fails=True, papareo=FakePapareo(configured=False))
    r = w2.run(memo)
    assert r["sent"] is False and "local transcription failed too" in r["note"]


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
