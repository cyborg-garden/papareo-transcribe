#!/bin/sh
# Install local speaker separation for papareo_transcribe, inside the agent's
# container. Everything lives under one persistent directory, outside the image.
# Every download is pinned and checked by hash before use. Safe to run twice, and
# after being interrupted.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
HOME_DIR="${PAPAREO_SPEAKERS_HOME:-/opt/data/papareo-speakers}"
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download
SEG_TAR_SHA=24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488
SEG_SHA=220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079
EMB_SHA=aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2

UV=$(command -v uv || ls /opt/hermes/tools/uv-*/uv 2>/dev/null | head -n 1 || true)
[ -n "$UV" ] || { echo "install_speakers.sh needs uv, and couldn't find it" >&2; exit 1; }

has() { [ -f "$2" ] && echo "$1  $2" | sha256sum -c - >/dev/null 2>&1; }

mkdir -p "$HOME_DIR/models"
HOME_DIR=$(cd "$HOME_DIR" && pwd)
# Run as root, uv must not leave its cache where the agent's own user needs to write.
UV_CACHE_DIR="$HOME_DIR/.uv-cache"
export UV_CACHE_DIR
cd "$HOME_DIR"
venv/bin/python -c "pass" 2>/dev/null || { rm -rf venv; "$UV" venv --python 3.13 -q venv; }
"$UV" pip install -q --python venv/bin/python --require-hashes -r "$HERE/speakers-requirements.txt"

cd models
if ! has "$SEG_SHA" seg.onnx; then
  rm -f seg.onnx
  curl -fsSL -o seg.tar.bz2 "$BASE/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  has "$SEG_TAR_SHA" seg.tar.bz2 || { echo "segmentation model failed its check" >&2; rm -f seg.tar.bz2; exit 1; }
  ../venv/bin/python -I -c "import tarfile; tarfile.open('seg.tar.bz2').extractall('.', filter='data')"
  cp sherpa-onnx-pyannote-segmentation-3-0/LICENSE seg.LICENSE
  cp sherpa-onnx-pyannote-segmentation-3-0/model.onnx seg.tmp
  has "$SEG_SHA" seg.tmp || { echo "segmentation model failed its check" >&2; rm -f seg.tmp; exit 1; }
  mv seg.tmp seg.onnx
  rm -rf seg.tar.bz2 sherpa-onnx-pyannote-segmentation-3-0
fi
if ! has "$EMB_SHA" emb.onnx; then
  rm -f emb.onnx
  curl -fsSL -o emb.tmp "$BASE/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
  has "$EMB_SHA" emb.tmp || { echo "voice model failed its check" >&2; rm -f emb.tmp; exit 1; }
  mv emb.tmp emb.onnx
fi
rm -rf "$HOME_DIR/.uv-cache"
"$HOME_DIR/venv/bin/python" -I -c "import sherpa_onnx; print('speaker separation ready:', sherpa_onnx.__version__)"
