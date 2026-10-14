#!/usr/bin/env bash
# Open the chartgen desktop app.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ ! -x .venv/bin/python ]]; then
    echo "chartgen is not set up yet. Run scripts/install.sh first." >&2
    exit 1
fi
exec .venv/bin/python -m chartgen.app
