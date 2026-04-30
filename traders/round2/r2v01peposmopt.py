# r2v01peposmopt.py: vNN counts only inside the `peposmopt` family (same round can have other r2v01… files with different tags).
# Pepper (+ osmium where dual-class); Optuna shell.
#
import argparse
import json
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

import optuna
from optuna.samplers import TPESampler

REPO        = _repo_root()
TRADER_SRC  = REPO / "traders" / "round2/r2v01peposminv.py"
CARGO       = str(REPO / "scripts" / "cargo_local.sh")
DB_PATH     = REPO / "traders" / "optuna_joymummy.db"

source_cache: str = ""

def define_params(trial: optuna.Trial) -> dict:
    return {

        "TAKE_BUY_PX":    trial.suggest_categorical("TAKE_BUY_PX",    [9999, 10000, 10001]),
        "TAKE_SELL_PX":   trial.suggest_categorical("TAKE_SELL_PX",   [10000, 10001, 10002]),

        "STRONG_BUY_PX":  trial.suggest_categorical("STRONG_BUY_PX",  [9997, 9998, 9999, 10000]),
        "STRONG_SELL_PX": trial.suggest_categorical("STRONG_SELL_PX", [10001, 10002, 10003, 10004]),

        "MAX_TAKE_SIZE":  trial.suggest_categorical("MAX_TAKE_SIZE",  [15, 20, 25, 30, 40]),
        "PASSIVE_SIZE":   trial.suggest_categorical("PASSIVE_SIZE",   [8, 10, 12, 15, 18, 20]),
        "STRONG_PASSIVE": trial.suggest_categorical("STRONG_PASSIVE", [15, 18, 20, 25, 30]),

        "SOFT_POS":       trial.suggest_categorical("SOFT_POS",       [25, 30, 35, 40, 50]),
        "DANGER_POS":     trial.suggest_categorical("DANGER_POS",     [50, 55, 60, 65, 70]),

        "STOP_DRAWDOWN":  trial.suggest_categorical("STOP_DRAWDOWN",  [400, 500, 600, 650, 750, 900, 1200]),

        "EARLY_TICKS":    trial.suggest_categorical("EARLY_TICKS",    [100_000, 150_000, 200_000, 250_000, 300_000, 400_000]),
    }

def inject_params(source: str, params: dict) -> str:
    lines = source.splitlines()
    out = []
    for line in lines:
        replaced = False
        for key, val in params.items():
            pattern = rf"^(\s+{re.escape(key)}\s*=\s*)[\d_]+(.*)$"
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
            by_product = m.get("final_pnl_by_product", {})
            pnl = (
                by_product.get("ASH_COATED_OSMIUM", 0.0)
                + by_product.get("INTARIAN_PEPPER_ROOT", 0.0)
            )
            total_pnl += pnl
            found += 1
        except Exception:
            pass

    return total_pnl if found > 0 else float("-inf")

def objective(trial: optuna.Trial) -> float:
    params = define_params(trial)

    for k, v in params.items():
        trial.set_user_attr(k, v)

    patched = inject_params(source_cache, params)
    temp_path = REPO / "traders" / f"_opt_joymummy_temp_{trial.number}.py"
    temp_path.write_text(patched)

    t0 = time.time()
    pnl = run_backtest(str(temp_path.relative_to(REPO)))
    elapsed = time.time() - t0

    temp_path.unlink(missing_ok=True)

    if pnl == float("-inf"):
        raise optuna.exceptions.TrialPruned()

    print(
        f"  Trial {trial.number:>4}  PnL={pnl:>12,.1f}  ({elapsed:.1f}s)  "
        + "  ".join(f"{k}={v}" for k, v in params.items())
    )

    return pnl

def main():
    global source_cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--jobs",  type=int, default=1)
    parser.add_argument("--study", type=str, default="joymummy")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if not TRADER_SRC.exists():
        print(f"ERROR: {TRADER_SRC} not found", file=sys.stderr)
        sys.exit(1)

    source_cache = TRADER_SRC.read_text()

    storage = f"sqlite:///{DB_PATH}"

    if args.fresh and DB_PATH.exists():
        try:
            optuna.delete_study(study_name=args.study, storage=storage)
            print(f"Deleted existing study '{args.study}'")
        except Exception:
            pass

    sampler = TPESampler(n_startup_trials=20, seed=42)

    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Optuna study: '{args.study}'")
    print(f"Database:     {DB_PATH}")
    print(f"Dashboard:    optuna-dashboard {storage}  →  http://localhost:8080")
    print(f"Trials:       {args.iters}  (jobs={args.jobs})")
    print(f"Sampler:      TPE (20 random warmup)\n")
    print("─" * 70)

    study.optimize(
        objective,
        n_trials=args.iters,
        n_jobs=args.jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"BEST  PnL = {best.value:,.1f}  (trial #{best.number})")
    print("=" * 60)
    for k, v in best.user_attrs.items():
        print(f"  {k} = {v}")

    print("\n── paste into joymummy.py ──────────────────────────────────")
    for k, v in best.user_attrs.items():
        print(f"    {k} = {v}")

    for f in REPO.glob("traders/_opt_joymummy_temp_*.py"):
        f.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
