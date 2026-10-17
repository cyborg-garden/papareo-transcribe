#!/bin/sh
# Run scripts/tune.py inside the agent image, so whisper is the exact
# faster-whisper build the tool uses. The memos folder is mounted read-write
# only so transcripts can be cached in <memos>/.transcripts/; whisper models
# are cached in a named volume between runs.
#
#   scripts/tune-in-agent-image.sh <memos-dir> [extra tune.py args, e.g. --write]
set -eu

MEMOS="$(cd "$1" && pwd)"; shift
IMAGE="${AGENT_IMAGE:-hermes-agent-mt:arm64-local}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm --entrypoint "" \
  -v "$REPO:/src" \
  -v "$MEMOS:/memos" \
  -v papareo-transcribe-whisper-cache:/root/.cache/huggingface \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$IMAGE" \
  /opt/hermes/.venv/bin/python -I /src/scripts/tune.py --memos /memos "$@"
