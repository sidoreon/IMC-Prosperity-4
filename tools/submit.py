#!/usr/bin/env python3
"""
IMC Prosperity portal submit helper (Playwright + Arc over CDP).

Run from the repository root, paths relative to that root:

  python tools/submit.py traders/round3/r3v01hvoptvol.py

Inspect mode (screenshots in tools/inspect_captures/):

  python tools/submit.py traders/latest_trader.py --inspect

Override CDP URL (default http://localhost:9222):

  SUBMIT_CDP_URL=http://127.0.0.1:9222 python tools/submit.py ...
"""

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from submit_common import BrowserProfile, run_main

ARC_PROFILE = BrowserProfile(
    label="Arc",
    connect_fail_message=(
        "\n  ERROR: Could not connect to Arc on {cdp_url}\n"
        "  Arc must be running with remote debugging enabled for that URL.\n"
        "\n  Start Arc from a terminal with a flag such as:\n"
        "    --remote-debugging-port=9222\n"
        "  (exact command depends on OS and where Arc is installed.)\n"
        "\n  Keep that window open, log in to the portal there, then re-run this script.\n"
        "  If CDP is on another host/port, set SUBMIT_CDP_URL.\n"
    ),
)

if __name__ == "__main__":
    run_main(ARC_PROFILE)
