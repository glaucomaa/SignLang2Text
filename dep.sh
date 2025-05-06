#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <local_dir> <remote_target>" >&2
  exit 1
fi

LOCAL_DIR="$1"
REMOTE="$2"

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Error: Local directory '$LOCAL_DIR' does not exist." >&2
  exit 1
fi

rsync -avhP \
  --compress \
  --human-readable \
  "$LOCAL_DIR/" \
  "$REMOTE"

