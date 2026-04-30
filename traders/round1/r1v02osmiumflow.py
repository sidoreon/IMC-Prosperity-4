# r1v02osmiumflow.py: vNN counts only inside the `osmiumflow` family (same round can have other r1v01… files with different tags).
# Osmium book; flow / microprice.
#
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cur in p.parents:
        if cur.name == "traders":
            return cur.parent
    return p.parent.parent

REPO        = _repo_root()
TRADER_SRC  = REPO / "traders" / "round1/r1v01osmiumflow.py"
TEMP_TRADER = REPO / "traders" / "_opt_opus_temp.py"
RUNS_DIR    = REPO / "runs"
CARGO       = str(REPO / "scripts" / "cargo_local.sh")

PARAM_SPACE = {
    "HISTORY_WINDOW":      [50, 100, 150, 200, 250, 300],
    "MIN_HISTORY":         [10, 15, 20, 25, 30],
    "EMA_ALPHA":           [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
    "TAKE_EDGE":           [1, 2, 3, 4],
    "CLEAR_WIDTH":         [0, 1, 2],
    "BASE_PASSIVE_SIZE":   [4, 6, 8, 10, 12, 15],
    "INVENTORY_SKEW":      [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18],
    "SOFT_POSITION_LIMIT": [20, 25, 30, 40, 50],
    "SIZE_ADJUST_RATE":    [5, 8, 10, 15, 20],
    "AR_WEIGHT":           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "AR_MIN_HISTORY":      [10, 15, 20, 25, 30],
    "MOMENTUM_WINDOW":     [4, 6, 8, 10, 15, 20],
    "MOMENTUM_WEIGHT":     [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    "MOMENTUM_LEAN":       [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
}

def sample_params() -> dict:
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}

def enforce_constraints(params: dict) -> dict:

    params["SOFT_POSITION_LIMIT"] = min(params["SOFT_POSITION_LIMIT"], 60)
    return params

def inject_params(source: str, params: dict) -> str:
    lines = source.splitlines()
    out = []
    for line in lines:
        replaced = False
        for key, val in params.items():
            pattern = rf"^(\s+{re.escape(key)}\s*=\s*)[\d.]+(.*)$"
            m = re.match(pattern, line)
            if m:
                out.append(f"{m.group(1)}{val}")
                replaced = True
                break
        if not replaced:
            out.append(line)
    return "\n".join(out)

def run_backtest(trader_path: str) -> float:
    cmd = [
        CARGO, "run", "--",
        "--trader", trader_path,
        "--dataset", "round1",
        "--products", "summary",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        return float("-inf")

    run_dirs = re.findall(r"(runs/backtest-\S+)", result.stdout)
    if not run_dirs:
        return float("-inf")

    total_pnl = 0.0
    found = 0
    for rel_dir in run_dirs:
        metrics_file = REPO / rel_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        try:
            with open(metrics_file) as f:
                m = json.load(f)
            pnl = m.get("final_pnl_by_product", {}).get("ASH_COATED_OSMIUM", 0.0)
            total_pnl += pnl
            found += 1
        except Exception:
            pass

    return total_pnl if found > 0 else float("-inf")

def format_params(params: dict) -> str:
    return "  " + "\n  ".join(f"{k}={v}" for k, v in params.items())

def main():
    parser = argparse.ArgumentParser(description="Random-search optimiser for opus.py")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed",  type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not TRADER_SRC.exists():
        print(f"ERROR: {TRADER_SRC} not found", file=sys.stderr)
        sys.exit(1)

    source = TRADER_SRC.read_text()

    best_pnl    = float("-inf")
    best_params = None
    results     = []

    print(f"Starting random search: {args.iters} iterations over {len(PARAM_SPACE)} parameters")
    print(f"Trader: {TRADER_SRC}")
    print(f"Dataset: round1 (all 3 days), scoring ASH_COATED_OSMIUM PnL\n")

    for i in range(1, args.iters + 1):
        params = enforce_constraints(sample_params())

        patched = inject_params(source, params)
        TEMP_TRADER.write_text(patched)

        t0      = time.time()
        pnl     = run_backtest(str(TEMP_TRADER.relative_to(REPO)))
        elapsed = time.time() - t0

        results.append((pnl, params))

        status  = "BEST" if pnl > best_pnl else "    "
        if pnl > best_pnl:
            best_pnl    = pnl
            best_params = params

        pnl_str = f"{pnl:>12,.1f}" if pnl != float("-inf") else "       FAILED"
        print(f"[{i:>3}/{args.iters}] {status}  PnL={pnl_str}  ({elapsed:.1f}s)")

    if TEMP_TRADER.exists():
        TEMP_TRADER.unlink()

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 60)
    print("TOP 5 RESULTS")
    print("=" * 60)
    for rank, (pnl, params) in enumerate(results[:5], 1):
        print(f"\n#{rank}  PnL = {pnl:,.1f}")
        print(format_params(params))

    print("\n" + "=" * 60)
    print("BEST CONSTANTS TO PASTE INTO opus.py")
    print("=" * 60)
    if best_params:
        for k, v in best_params.items():
            print(f"    {k} = {v}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
