#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zzf/myeye/CAMS-MBRCNet
exec "$ROOT/scripts/train_proposed.sh" ROSSA "${1:?Usage: train_rossa.sh VARIANT [SEED]}" "${2:-0}"
