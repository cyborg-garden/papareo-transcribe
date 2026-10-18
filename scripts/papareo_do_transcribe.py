"""One live call to Papa Reo through the plugin's client, to check it still
matches the live API (verified 2026-10-18, both routes).

    PAPAREO_API_KEY=... python -I scripts/papareo_do_transcribe.py <recording> [--large] [--show]

Spends money: one recording, one call. Prints the shape of each response
(status, keys, types, lengths) and, only with --show, the transcript: run
it at a terminal on a recording you may read the transcript of, never
inside an agent.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hermes-plugin"))

import papareo  # noqa: E402


def _shape(value: object, depth: int = 0) -> str:
    """A value's structure with every leaf replaced by its type (and a
    length for strings and lists). Never a value."""
    if depth > 3:
        return "…"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {_shape(v, depth + 1)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        inner = _shape(value[0], depth + 1) if value else ""
        return f"list[{len(value)}]" + (f" of {inner}" if inner else "")
    if isinstance(value, str):
        return f"str[{len(value)}]"
    return type(value).__name__


# Only these values are ever printed, and only when they look like what they
# are: a status word and a percentage can't carry transcript text.
_STATUS = re.compile(r"^[A-Za-z_]{1,20}$")
_PERCENT = re.compile(r"^[0-9]{1,3}(\.[0-9]+)?$")


class Recording(httpx.HTTPTransport):
    """Passes requests through and prints each response's shape as it
    arrives. Status polls print their shape once, then one line that updates
    in place with the task's status and percent complete."""

    def __init__(self) -> None:
        super().__init__()
        self._polls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)
        response.read()
        try:
            body = response.json()
        except ValueError:
            body = None
        if request.url.path.endswith("/status") and response.status_code == 200 and isinstance(body, dict):
            self._polls += 1
            if self._polls == 1:
                self._print(request, response, body)
            self._progress(body)
            return response
        self._end_progress()
        self._print(request, response, body)
        return response

    def _print(self, request: httpx.Request, response: httpx.Response, body: object) -> None:
        shape = f"{request.method} {request.url.path} -> HTTP {response.status_code} {response.headers.get('content-type')}"
        # Key names and value types only, never values: until the live
        # response is known, any key could be the one holding transcript
        # text, and this output may land in a terminal an LLM reads.
        shape += " " + _shape(body) if body is not None else f" ({len(response.content)} bytes, not JSON)"
        print(" ", shape, flush=True)

    def _progress(self, body: dict) -> None:
        status = str(body.get("status", ""))
        percent = str(body.get("progress_percent", ""))
        status = status if _STATUS.match(status) else "?"
        percent = f"{percent}%" if _PERCENT.match(percent) else "?%"
        line = f"  waiting: {status} {percent} (check {self._polls})"
        if sys.stdout.isatty():
            print("\r" + line.ljust(60), end="", flush=True)
        else:
            print(line, flush=True)

    def _end_progress(self) -> None:
        if self._polls and sys.stdout.isatty():
            print(flush=True)
        self._polls = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--large", action="store_true", help="exercise the long-recording flow")
    ap.add_argument("--show", action="store_true", help="print the transcript (your own recording only)")
    args = ap.parse_args()
    key = os.environ.get("PAPAREO_API_KEY")
    if not key:
        print("set PAPAREO_API_KEY", file=sys.stderr)
        return 2

    transport = Recording()
    client = papareo.PapareoClient(key, transport=transport)
    try:
        text = client.transcribe(str(args.recording), 600.0 if args.large else 30.0)
    except papareo.PapareoError as e:
        transport._end_progress()
        print("FAILED:", e)
        return 1
    print(f"client returned {len(text)} chars: OK")
    if args.show:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
