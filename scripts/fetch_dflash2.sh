#!/usr/bin/env bash
# Download the DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2, a pinned revision, ~1 GB) into DFLASH2_DIR.
# Resumable: each file downloads to NAME.part and takes its name once its SHA-256 matches. Files already there are kept
# (start.sh checks the weights' SHA-256 at every start). ./start.sh runs this by itself when DRAFTER=dflash2.
# Usage: scripts/fetch_dflash2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
mkdir -p "$DFLASH2_DIR"
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
get() {
  local f=$1 want=$2 to="$DFLASH2_DIR/$1" part="$DFLASH2_DIR/$1.part"
  [[ -s "$to" ]] && return
  if [[ "$(sha "$part")" != "$want" ]]; then   # a finished .part has nothing left to resume
    log "fetching the DFlash2 drafter's $f"
    curl -fL --retry 5 --retry-delay 5 -C - -o "$part" "https://huggingface.co/$DFLASH2_REPO/resolve/$DFLASH2_REVISION/$f" ||
      die "could not fetch $f from $DFLASH2_REPO@${DFLASH2_REVISION:0:12}; run it again to resume"
    [[ "$(sha "$part")" == "$want" ]] || { rm -f -- "$part"; die "$f from $DFLASH2_REPO does not match its SHA-256 ($want); removed, run it again"; }
  fi
  mv -- "$part" "$to"
}
get config.json "$DFLASH2_CONFIG_SHA256"
get model.safetensors "$DFLASH2_SHA256"
