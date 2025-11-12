#!/usr/bin/env bash
set -euo pipefail

# Apply local patches (in ./patches) into the installed MetaAF package path.
# Usage: from repo root that contains ./patches/aec_infer.py and ./patches/learners.py
# Example:
#   ./scripts/apply_patches.sh
#
# This script finds where `metaaf` is installed, then copies the patched files in.

PYTHON_BIN="${PYTHON_BIN:-python}"
PKG_DIR="$($PYTHON_BIN - <<'PY'
import os, metaaf
print(os.path.dirname(metaaf.__file__))
PY
)"

echo "[apply_patches] metaaf package dir: $PKG_DIR"

# Where our patches live (relative to repo root where this script is)
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches"

if [[ ! -d "$PATCH_DIR" ]]; then
  echo "[apply_patches] ERROR: patches/ directory not found next to scripts/"
  exit 1
fi

for f in aec_infer.py learners.py; do
  if [[ -f "$PATCH_DIR/$f" ]]; then
    echo "[apply_patches] Copying $PATCH_DIR/$f -> $PKG_DIR/$f"
    cp -f "$PATCH_DIR/$f" "$PKG_DIR/$f"
  else:
    echo "[apply_patches] WARNING: $PATCH_DIR/$f not found; skipping"
  fi
done

echo "[apply_patches] Done."
