"""One live call to Papa Reo, to verify the client before it is switched on.

    PAPAREO_API_KEY=... python -I scripts/papareo_smoke.py <recording> [--large] [--show]

Spends money: one recording, one call. Prints the shape of each response
(status, keys, types, lengths) and, only with --show, the transcript: run
it at a terminal on a recording you may read the transcript of, never
inside an agent. When the shapes match the client, set LIVE_VERIFIED = True
in hermes-plugin/papareo.py and record the date in the README.
"""

from __future__ import annotations

import argparse
import os
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


class Recording(httpx.HTTPTransport):
    """Passes requests through and keeps each response's shape."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)
        response.read()
        shape = f"{request.method} {request.url.path} -> HTTP {response.status_code} {response.headers.get('content-type')}"
        try:
            # Key names and value types only, never values: until the live
            # response is known, any key could be the one holding transcript
            # text, and this output may land in a terminal an LLM reads.
            shape += " " + _shape(response.json())
        except ValueError:
            shape += f" ({len(response.content)} bytes, not JSON)"
        self.seen.append(shape)
        return response


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
    client = papareo.PapareoClient(key, transport=transport, verified=True)
    try:
        text = client.transcribe(str(args.recording), 600.0 if args.large else 30.0)
    except papareo.PapareoError as e:
        print("FAILED:", e)
        for s in transport.seen:
            print(" ", s)
        return 1
    for s in transport.seen:
        print(" ", s)
    print(f"client returned {len(text)} chars: OK")
    if args.show:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
