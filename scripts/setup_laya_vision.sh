#!/usr/bin/env bash
# Install laya-vision into its own venv under third_party/ (gitignored), pinned.
#
#   bash scripts/setup_laya_vision.sh
#   third_party/laya-vision/.venv/bin/python scripts/laya_server.py
#
# Its own venv because it needs transformers >= 5.3 and lerobot 0.3.2 (the
# project venv) pins transformers < 4.52. Needs uv. Downloads ~770 MB of
# weights into the Hugging Face cache. Weights: CC BY-NC-SA 4.0, non-commercial.
set -euo pipefail

REPO=https://github.com/r33drichards/laya-vision
COMMIT=568feeeada793f70f736756b0f3a7643d1e75910      # 2026-09-24
MODEL=thaitea/laya-vision
REVISION=8b318c99d7ad3ce19c24369263463882eada9d1e    # keep in sync with laya_server.py

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/laya-vision"

if [ ! -d "$DEST/.git" ]; then
  git clone --quiet "$REPO" "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"

[ -x "$DEST/.venv/bin/python" ] || uv venv --quiet --python 3.12 "$DEST/.venv"
uv pip install --quiet --python "$DEST/.venv/bin/python" -e "$DEST" torchvision pillow

"$DEST/.venv/bin/python" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", revision="$REVISION")
import laya, torch
print("laya-vision ready:", "$MODEL@${REVISION:0:8}", "| mps:", torch.backends.mps.is_available())
EOF
