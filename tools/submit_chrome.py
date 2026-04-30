#!/usr/bin/env python3
"""
IMC Prosperity portal submit helper (Playwright + Google Chrome over CDP).

Run from the repository root:

  python tools/submit_chrome.py traders/round3/r3v01hvoptvol.py

Inspect mode:

  python tools/submit_chrome.py traders/latest_trader.py --inspect

Start Chrome or Chromium with remote debugging (quit other browser windows first if the profile is in use):

  <chrome-or-chromium> --remote-debugging-port=9222

  Use whatever executable name or full path applies on the machine (e.g. chrome, google-chrome, chromium).

Override port / URL:

  SUBMIT_CDP_URL=http://127.0.0.1:9223 python tools/submit_chrome.py traders/latest_trader.py
"""

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from submit_common import BrowserProfile, run_main

CHROME_PROFILE = BrowserProfile(
    label="Google Chrome",
    connect_fail_message=(
        "\n  ERROR: Could not connect to Google Chrome on {cdp_url}\n"
        "  Chrome (or Chromium) must be running with remote debugging enabled for that URL.\n"
        "\n  Start it from a terminal with a flag such as:\n"
        "    --remote-debugging-port=9222\n"
        "  (exact command depends on OS and install; try chrome, google-chrome, or chromium.)\n"
        "\n  Log in to prosperity.imc.com in that window, then re-run this script.\n"
        "  If CDP is elsewhere, set SUBMIT_CDP_URL.\n"
    ),
)

if __name__ == "__main__":
    run_main(CHROME_PROFILE)
