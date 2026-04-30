#!/usr/bin/env bash
# Run from repository root: ./scripts/verify.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "== rustfmt (backtester) =="
cd "$ROOT/backtester"
./scripts/cargo_local.sh fmt --all -- --check

echo ""
echo "== rust_backtester (library tests) =="
./scripts/cargo_local.sh test --lib

echo ""
echo "== tools/submit*.py (syntax) =="
python3 -m py_compile "$ROOT/tools/submit_common.py" "$ROOT/tools/submit.py" "$ROOT/tools/submit_chrome.py"

echo ""
echo "verify: ok"
