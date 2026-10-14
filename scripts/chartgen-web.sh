#!/usr/bin/env bash
# Start the chartgen web UI on http://127.0.0.1:8471. Arguments go to
# `python -m chartgen.web`, for example `--host 0.0.0.0` to reach it over the
# LAN or Tailscale. See docs/remote-access.md for CHARTGEN_WEB_OWNERS and
# Cloudflare Tunnel.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ ! -x .venv/bin/python ]]; then
    echo "chartgen is not set up yet. Run scripts/install.sh first." >&2
    exit 1
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec .venv/bin/python -m chartgen.web "$@"
