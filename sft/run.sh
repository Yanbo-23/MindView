#!/usr/bin/env bash
set -euo pipefail

SFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" != --help && "${1:-}" != -h && -n "${CONFIG_FILE:-}" ]]; then
  set -a
  source "${CONFIG_FILE}"
  set +a
fi
exec "${BAGEL_PYTHON:-${PYTHON_BIN:-python}}" "${SFT_ROOT}/train/launch.py" "$@"
