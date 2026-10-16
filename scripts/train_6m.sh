#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zzf/myeye/CAMS-MBRCNet
exec "$ROOT/scripts/train_proposed.sh" OCTA500_6M "${1:?Usage: train_6m.sh VARIANT [SEED]}" "${2:-0}"
