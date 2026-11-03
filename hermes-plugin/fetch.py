"""Fetch a recording from a share link, into a temp dir the caller removes.

The download runs in a process of its own, with no keys in its environment. A
server that drips bytes can hold a read open for as long as it likes, and only a
process can be stopped for certain when the time is up. This file is that process
as well, so it imports nothing from the plugin.

Handles direct links, Google Drive, Dropbox and OneDrive/SharePoint file links.
A link that needs a login, or a folder link, ends up at a web page, which is
refused rather than transcribed. Messages never repeat the link: share links
often carry access tokens.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

# httpx logs every URL at INFO, and share links carry access tokens.
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 20 * 60.0
MAX_REDIRECTS = 8
TIMEOUT_S = 120.0
TOO_LONG = "that link took too long to download"
UNREACHABLE = "that link couldn't be reached"
NEEDS_LINK = (
    "that link didn't give me an audio file. It may need a login, point at a folder, or have been "
    "copied from the address bar after opening the file, which only works in that browser. "
    "Ask for the original share link to one file, shared as 'anyone with the link can view', or for the file itself"
)
_DRIVE_ID = re.compile(r"/file/(?:u/\d+/)?d/([\w-]+)")
_NOT_PUBLIC = tuple(
    ipaddress.ip_network(n)
    for n in ("64:ff9b::/96", "64:ff9b:1::/48", "::/96", "2002::/16", "fec0::/10", "2001::/32")
)
# Only these names survive. Anything else is judged by its content alone.
_AUDIO_SUFFIXES = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".opus",
    ".aac", ".flac", ".amr", ".3gp", ".caf", ".mov",
}


class FetchError(Exception):
    """The recording could not be fetched. The message is safe to pass on."""


def _with_query(url: str, **params: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True)) | params
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def candidates(url: str) -> list[str]:
    """Where to try downloading a share link from, best first."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host in {"drive.google.com", "docs.google.com"}:
        match = _DRIVE_ID.search(parts.path)
        query = dict(parse_qsl(parts.query))
        file_id = match.group(1) if match else query.get("id")
        if file_id:
            params = {"id": file_id, "export": "download", "confirm": "t"}
            if query.get("resourcekey"):
                params["resourcekey"] = query["resourcekey"]
            return ["https://drive.usercontent.google.com/download?" + urlencode(params)]
    if host in {"dropbox.com", "www.dropbox.com"}:
        return [_with_query(url, dl="1")]
    if host.endswith(".sharepoint.com"):
        return [_with_query(url, download="1")]
    if host in {"1drv.ms", "onedrive.live.com"}:
        # Microsoft's documented way to turn a sharing link into its content.
        encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        return [f"https://api.onedrive.com/v1.0/shares/u!{encoded}/root/content", _with_query(url, download="1")]
    return [url]


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if not ip.is_global or ip.is_multicast:
        return False
    return not (ip.version == 6 and any(ip in net for net in _NOT_PUBLIC))


def _public_host(host: str, resolve: Callable[..., list]) -> bool:
    try:
        infos = resolve(host, 443, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return False
    return bool(infos) and all(_is_public(info[4][0]) for info in infos)


def _check(url: str, resolve: Callable[..., list]) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise FetchError("only https links can be fetched")
    if parts.username or parts.password:
        raise FetchError("that link doesn't look right")
    if not _public_host(parts.hostname, resolve):
        raise FetchError("that link doesn't point at a public address")


def _extension(response: httpx.Response, url: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition)
    name = match.group(1) if match else urlsplit(url).path.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    return suffix if suffix in _AUDIO_SUFFIXES else ""


def _download(
    client: httpx.Client, url: str, dest_dir: Path, max_bytes: int, resolve: Callable[..., list]
) -> Path | None:
    for _ in range(MAX_REDIRECTS):
        _check(url, resolve)
        # Audio barely compresses, and a compressed reply can be a bomb that only
        # grows once decoded, so ask for it as is and take it as it comes.
        with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as r:
            if r.is_redirect and r.headers.get("location"):
                url = urljoin(url, r.headers["location"])
                continue
            if not 200 <= r.status_code < 300:
                return None
            if "text/html" in r.headers.get("content-type", "").lower():
                return None
            if r.headers.get("content-encoding", "identity").strip().lower() != "identity":
                return None
            target = dest_dir / ("recording" + _extension(r, url))
            total = 0
            first = b""
            with target.open("wb") as f:
                for chunk in r.iter_raw():
                    if not first:
                        first = chunk[:64].lstrip().lower()
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError("that recording is too large")
                    f.write(chunk)
            if not total or first.startswith((b"<!doctype", b"<html")):
                target.unlink(missing_ok=True)
                return None
            return target
    raise FetchError("that link redirected too many times")


def download(
    url: str,
    dest_dir: str,
    *,
    max_bytes: int = MAX_BYTES,
    transport: httpx.BaseTransport | None = None,
    resolve: Callable[..., list] = socket.getaddrinfo,
) -> str:
    """The download itself. Call fetch(), which puts a time limit on this."""
    try:
        # trust_env off: a proxy in the environment must not see the link.
        with httpx.Client(timeout=TIMEOUT_S, follow_redirects=False, transport=transport, trust_env=False) as client:
            for candidate in candidates(url.strip()):
                got = _download(client, candidate, Path(dest_dir), max_bytes, resolve)
                if got:
                    return str(got)
    except httpx.HTTPError as e:
        raise FetchError(UNREACHABLE) from e
    except (httpx.InvalidURL, ValueError) as e:  # UnicodeError is a ValueError
        raise FetchError("that link doesn't look right") from e
    raise FetchError(NEEDS_LINK)


def fetch(
    url: str,
    dest_dir: str,
    *,
    max_seconds: float = MAX_SECONDS,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Download in a separate process, and stop it for certain at the time limit.
    The link goes in on stdin: on a command line, anyone could read it."""
    try:
        done = run(
            [sys.executable, "-I", os.path.abspath(__file__)],
            input=json.dumps({"url": url, "dest": dest_dir}).encode(),
            capture_output=True,
            timeout=max_seconds,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": dest_dir},
        )
    except subprocess.TimeoutExpired as e:
        raise FetchError(TOO_LONG) from e
    except OSError as e:
        raise FetchError(UNREACHABLE) from e
    try:
        answer = json.loads(done.stdout)
        path, error = answer.get("path"), answer.get("error")
    except (ValueError, AttributeError) as e:
        raise FetchError(UNREACHABLE) from e
    if isinstance(error, str) and error:
        raise FetchError(error)
    if isinstance(path, str) and Path(path).resolve().parent == Path(dest_dir).resolve():
        return path
    raise FetchError(UNREACHABLE)


def main() -> None:
    try:
        request = json.load(sys.stdin)
        answer = {"path": download(request["url"], request["dest"])}
    except FetchError as e:
        answer = {"error": str(e)}
    except Exception:  # noqa: BLE001 - its message could repeat the link
        answer = {"error": UNREACHABLE}
    json.dump(answer, sys.stdout)


if __name__ == "__main__":
    main()
