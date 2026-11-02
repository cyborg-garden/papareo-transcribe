"""Share links: the right place to download from, and refusing what isn't audio."""

from __future__ import annotations

import base64
import socket
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def f(plugin):
    return plugin.fetch


def public(host, port, **kw):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def private(host, port, **kw):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", port))]


def test_google_drive_link(f):
    got = f.candidates("https://drive.google.com/file/d/AbC_1-x/view?usp=sharing")
    assert got == ["https://drive.usercontent.google.com/download?id=AbC_1-x&export=download&confirm=t"]


def test_sharepoint_link_asks_for_the_download(f):
    assert "download=1" in f.candidates("https://uni-my.sharepoint.com/:u:/g/personal/x/EaBc?e=zz")[0]


@pytest.mark.parametrize("url", ["https://1drv.ms/u/s!AbCdE", "https://onedrive.live.com/?id=1&cid=2"])
def test_onedrive_link_uses_the_shares_api(f, url):
    first, second = f.candidates(url)
    token = first.split("shares/u!")[1].split("/root")[0]
    assert base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode() == url
    assert "download=1" in second


def test_onedrive_falls_through_to_the_second_way(f, tmp_path):
    def handler(request):
        if request.url.host == "api.onedrive.com":
            return httpx.Response(401)
        return httpx.Response(200, headers={"content-type": "audio/mp4"}, content=b"audio")

    path = f.fetch("https://1drv.ms/u/s!AbC", str(tmp_path), transport=httpx.MockTransport(handler), resolve=public)
    assert Path(path).read_bytes() == b"audio"


def test_redirects_are_followed_and_named(f, tmp_path):
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://cdn.example.org/a"})
        return httpx.Response(
            200, headers={"content-type": "application/octet-stream", "content-disposition": 'attachment; filename="hui.m4a"'},
            content=b"audio",
        )

    path = f.fetch("https://example.org/start", str(tmp_path), transport=httpx.MockTransport(handler), resolve=public)
    assert Path(path).name == "recording.m4a"


@pytest.mark.parametrize("response", [
    httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>sign in</html>"),
    httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"  <!DOCTYPE html><html>"),
    httpx.Response(403),
])
def test_login_pages_are_refused_and_never_kept(f, tmp_path, response):
    with pytest.raises(f.FetchError, match="anyone with the link"):
        f.fetch("https://example.org/a.mp3", str(tmp_path), transport=httpx.MockTransport(lambda r: response), resolve=public)
    assert list(tmp_path.iterdir()) == []


def test_private_addresses_are_refused(f, tmp_path):
    transport = httpx.MockTransport(lambda r: pytest.fail("must not connect"))
    with pytest.raises(f.FetchError, match="public address"):
        f.fetch("https://intranet.example/a.mp3", str(tmp_path), transport=transport, resolve=private)


def test_redirect_to_a_private_address_is_refused(f, tmp_path):
    def handler(request):
        return httpx.Response(302, headers={"location": "https://inner.example/a"})

    def resolve(host, port, **kw):
        return private(host, port) if host == "inner.example" else public(host, port)

    with pytest.raises(f.FetchError, match="public address"):
        f.fetch("https://example.org/a", str(tmp_path), transport=httpx.MockTransport(handler), resolve=resolve)


@pytest.mark.parametrize("url", ["http://example.org/a.mp3", "file:///etc/passwd", "ftp://example.org/a"])
def test_only_https(f, tmp_path, url):
    with pytest.raises(f.FetchError, match="only https"):
        f.fetch(url, str(tmp_path), resolve=public)


def test_too_large(f, tmp_path):
    handler = lambda r: httpx.Response(200, headers={"content-type": "audio/mp4"}, content=b"x" * 100)  # noqa: E731
    with pytest.raises(f.FetchError, match="too large"):
        f.fetch("https://example.org/a.mp3", str(tmp_path), max_bytes=10, transport=httpx.MockTransport(handler), resolve=public)


def test_errors_never_repeat_the_link(f, tmp_path):
    secret = "https://example.org/a.mp3?token=SEKRET"

    def handler(request):
        raise httpx.ConnectError("boom " + secret)

    with pytest.raises(f.FetchError) as e:
        f.fetch(secret, str(tmp_path), transport=httpx.MockTransport(handler), resolve=public)
    assert "SEKRET" not in str(e.value)


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "64:ff9b::7f00:1", "::127.0.0.1",
    "fec0::1", "ff02::1", "224.0.0.1", "2002:7f00:1::", "fe80::1%en0",
])
def test_not_public(f, address):
    assert not f._is_public(address)


@pytest.mark.parametrize("address", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
def test_public(f, address):
    assert f._is_public(address)


@pytest.mark.parametrize("url", ["https://[::1/x", "https://example.org:abc/x", "https://user:pw@example.org/x"])
def test_odd_links_are_a_clean_error(f, tmp_path, url):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"x"))
    with pytest.raises(f.FetchError) as e:
        f.fetch(url, str(tmp_path), transport=transport, resolve=public)
    assert "a" * 70 not in str(e.value) and "pw" not in str(e.value)


def test_redirect_to_a_broken_link_is_a_clean_error(f, tmp_path):
    transport = httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "https://[bad"}))
    with pytest.raises(f.FetchError):
        f.fetch("https://example.org/a", str(tmp_path), transport=transport, resolve=public)


def test_slow_headers_stop_at_the_deadline(f, tmp_path):
    import threading
    import time

    release = threading.Event()

    def handler(request):
        release.wait(5)
        return httpx.Response(200, headers={"content-type": "audio/mp4"}, content=b"x")

    started = time.monotonic()
    with pytest.raises(f.FetchError, match="too long"):
        f.fetch("https://example.org/a.mp3", str(tmp_path), max_seconds=0.2, transport=httpx.MockTransport(handler), resolve=public)
    release.set()
    assert time.monotonic() - started < 2


def test_a_drip_stops_writing_after_the_deadline(f, tmp_path):
    import time

    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(200):
                time.sleep(0.05)
                yield b"x"

    transport = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/mp4"}, stream=Drip()))
    with pytest.raises(f.FetchError, match="too long"):
        f.fetch("https://example.org/a.mp3", str(tmp_path), max_seconds=0.2, transport=transport, resolve=public)


def test_a_served_playlist_name_is_not_kept(f, tmp_path):
    headers = {"content-type": "application/octet-stream", "content-disposition": 'attachment; filename="a.m3u8"'}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, headers=headers, content=b"#EXTM3U"))
    path = f.fetch("https://example.org/x", str(tmp_path), transport=transport, resolve=public)
    assert Path(path).name == "recording"


def test_drive_keeps_the_resource_key_and_account_paths(f):
    got = f.candidates("https://drive.google.com/file/u/1/d/AbC/view?resourcekey=0-xy")[0]
    assert "id=AbC" in got and "resourcekey=0-xy" in got


def test_a_label_too_long_to_resolve_is_a_clean_error(f, tmp_path):
    host = "a" * 70 + ".org"
    with pytest.raises(f.FetchError) as e:
        f.fetch(f"https://{host}/x", str(tmp_path))  # the real resolver
    assert host not in str(e.value)


def test_only_a_few_downloads_at_once_and_slots_come_back(f, tmp_path):
    import threading

    release = threading.Event()

    def handler(request):
        release.wait(5)
        return httpx.Response(200, headers={"content-type": "audio/mp4"}, content=b"x")

    transport = httpx.MockTransport(handler)
    for _ in range(3):
        with pytest.raises(f.FetchError, match="too long"):
            f.fetch("https://example.org/a.mp3", str(tmp_path), max_seconds=0.01, transport=transport, resolve=public)
    with pytest.raises(f.FetchError, match="too many"):
        f.fetch("https://example.org/a.mp3", str(tmp_path), max_seconds=0.01, transport=transport, resolve=public)
    release.set()
    for _ in range(200):
        if f._SLOTS.acquire(timeout=0.05):
            f._SLOTS.release()
            break
    assert f.fetch("https://example.org/a.mp3", str(tmp_path), transport=httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "audio/mp4"}, content=b"x")), resolve=public)
