"""The Papa Reo client, against a mock of the published API."""

from __future__ import annotations

import secrets

import httpx
import pytest


@pytest.fixture
def audio(tmp_path):
    p = tmp_path / "memo.m4a"
    p.write_bytes(secrets.token_bytes(256))
    return p


def client(plugin, handler, *, key="k-123"):
    return plugin.papareo.PapareoClient(key, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_no_key_makes_no_request(plugin, audio):
    def handler(request):  # pragma: no cover
        raise AssertionError("no request may be made")

    with pytest.raises(plugin.papareo.NotConfigured, match="no Papa Reo key"):
        client(plugin, handler, key=None).transcribe(str(audio), 10.0)


def test_short_recording(plugin, audio):
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json={"success": True, "transcription": "kia ora", "model_version": "x"})

    assert client(plugin, handler).transcribe(str(audio), 42.0) == "kia ora"
    assert seen["url"] == "https://api.papareo.io/tuhi/transcribe"
    assert seen["auth"] == "Token k-123"
    assert b'name="audio_file"; filename="memo.m4a"' in seen["body"]
    assert audio.read_bytes() in seen["body"], "the original file, not a re-encode"


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [
        (401, {"detail": "Unauthorized or token expired"}, "refused the key"),
        (500, {"detail": "boom"}, "HTTP 500"),
        (200, {"success": False, "transcription": "secret words"}, "did not return"),
        (200, {"success": True}, "did not return"),
    ],
)
def test_short_failures_never_echo_the_body(plugin, audio, status, body, match):
    c = client(plugin, lambda r: httpx.Response(status, json=body))
    with pytest.raises(plugin.papareo.PapareoError, match=match) as ei:
        c.transcribe(str(audio), 42.0)
    assert "secret words" not in str(ei.value)


def test_timeout_is_a_papareo_error(plugin, audio):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(plugin.papareo.PapareoError, match="timed out"):
        client(plugin, handler).transcribe(str(audio), 42.0)


def test_long_recording_uses_the_large_flow(plugin, audio):
    calls = []
    statuses = iter(["PENDING", "STARTED", "SUCCESS"])

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/tuhi/transcribe/large":
            return httpx.Response(202, json={"task_id": "t1", "status": "PENDING", "links": {}})
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"task_id": "t1", "status": next(statuses), "links": {}})
        if request.url.path.endswith("/download"):
            return httpx.Response(200, text="he kōrero roa")
        raise AssertionError(request.url)

    assert client(plugin, handler).transcribe(str(audio), 600.0) == "he kōrero roa"
    assert calls[0][:2] == ("POST", "/tuhi/transcribe/large")
    assert calls[-1] == ("GET", "/tuhi/transcribe/large/t1/download", {"download_type": "txt"})
    assert sum(1 for c in calls if c[1].endswith("/status")) == 3


def test_long_recording_failure(plugin, audio):
    def handler(request):
        if request.url.path == "/tuhi/transcribe/large":
            return httpx.Response(202, json={"task_id": "t1", "status": "PENDING", "links": {}})
        return httpx.Response(200, json={"task_id": "t1", "status": "FAILURE", "links": {}})

    with pytest.raises(plugin.papareo.PapareoError, match="FAILURE"):
        client(plugin, handler).transcribe(str(audio), 600.0)


def test_footer_is_verbatim(plugin):
    out = plugin.papareo.with_footer("  kia ora  \n")
    assert out == (
        b"kia ora\n\nThis text transcribed by Papa Reo API. Please be sure to comply with the "
        b"terms of the Kaitiakitanga license in use of this material.\n"
    )
