#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zzf/myeye/CAMS-MBRCNet
exec "$ROOT/scripts/train_proposed.sh" OCTA500_3M "${1:?Usage: train_3m.sh VARIANT [SEED]}" "${2:-0}"
