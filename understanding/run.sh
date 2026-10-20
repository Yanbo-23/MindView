#!/usr/bin/env bash
set -euo pipefail

UNDERSTANDING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
SHOW_HELP=0
for argument in "$@"; do
  case "$argument" in --help|-h) SHOW_HELP=1;; esac
done
if [[ "${SKIP_CONFIG:-0}" != 1 && "$SHOW_HELP" != 1 && -n "${CONFIG_FILE:-}" ]]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi
exec "${EVAL_PYTHON:-${BAGEL_PYTHON:-${PYTHON_BIN:-python3}}}" \
  "$UNDERSTANDING_ROOT/launch.py" "$@"
