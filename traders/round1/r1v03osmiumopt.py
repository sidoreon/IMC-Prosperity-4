# r1v03osmiumopt.py: vNN counts only inside the `osmiumopt` family (same round can have other r1v01… files with different tags).
# Osmium book; Optuna study wrapper.
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
TRADER_SRC  = REPO / "traders" / "round1/r1v03osmiuminv.py"
CARGO       = str(REPO / "scripts" / "cargo_local.sh")
DB_PATH     = REPO / "traders" / "optuna_osmiumv2.db"

source_cache: str = ""

def define_params(trial: optuna.Trial) -> dict:
    return {
        "HISTORY_WINDOW":      trial.suggest_categorical("HISTORY_WINDOW",      [50, 100, 150, 200, 220, 250, 300]),
        "MIN_HISTORY":         trial.suggest_categorical("MIN_HISTORY",          [10, 15, 20, 25, 30]),
        "EMA_ALPHA":           trial.suggest_categorical("EMA_ALPHA",            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
        "FAIR_BLEND":          trial.suggest_categorical("FAIR_BLEND",           [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
        "TAKE_EDGE":           trial.suggest_int        ("TAKE_EDGE",            1, 4),
        "STRONG_TAKE_EDGE":    trial.suggest_categorical("STRONG_TAKE_EDGE",     [2.5, 3.0, 3.5, 4.0, 5.0]),
        "MIN_PENNY_SPREAD":    trial.suggest_categorical("MIN_PENNY_SPREAD",     [2, 3, 4, 5, 6]),
        "PASSIVE_OFFSET":      trial.suggest_categorical("PASSIVE_OFFSET",       [1, 2]),
        "BASE_PASSIVE_SIZE":   trial.suggest_categorical("BASE_PASSIVE_SIZE",    [6, 8, 10, 12, 15]),
        "BIG_PASSIVE_SIZE":    trial.suggest_categorical("BIG_PASSIVE_SIZE",     [10, 12, 15, 18, 20]),
        "REDUCE_TRIGGER":      trial.suggest_categorical("REDUCE_TRIGGER",       [20, 25, 30, 40, 50]),
        "REDUCE_SIZE":         trial.suggest_categorical("REDUCE_SIZE",          [2, 3, 4, 6, 8]),
        "INVENTORY_SKEW":      trial.suggest_categorical("INVENTORY_SKEW",       [0.02, 0.04, 0.06, 0.08, 0.10]),
        "LOW_SIGNAL_Z":        trial.suggest_categorical("LOW_SIGNAL_Z",         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "IMBALANCE_THRESHOLD": trial.suggest_categorical("IMBALANCE_THRESHOLD",  [0.1, 0.15, 0.2, 0.25, 0.3]),
        "TIGHT_SPREAD":        trial.suggest_categorical("TIGHT_SPREAD",         [2, 3, 4, 5, 6]),
    }

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
    cmd = [CARGO, "run", "--", "--trader", trader_path, "--dataset", "round1", "--products", "summary"]
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=120)

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

def objective(trial: optuna.Trial) -> float:
    params = define_params(trial)
    for k, v in params.items():
        trial.set_user_attr(k, v)

    patched = inject_params(source_cache, params)
    temp_path = REPO / "traders" / f"_opt_osmv2_temp_{trial.number}.py"
    temp_path.write_text(patched)

    t0 = time.time()
    pnl = run_backtest(str(temp_path.relative_to(REPO)))
    elapsed = time.time() - t0

    temp_path.unlink(missing_ok=True)

    if pnl == float("-inf"):
        raise optuna.exceptions.TrialPruned()

    print(f"  Trial {trial.number:>4}  PnL={pnl:>12,.1f}  ({elapsed:.1f}s)")
    return pnl

def main():
    global source_cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",  type=int, default=100)
    parser.add_argument("--study",  type=str, default="osmiumv2")
    parser.add_argument("--fresh",  action="store_true")
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

    print(f"Study:     {args.study}")
    print(f"Database:  {DB_PATH}")
    print(f"Dashboard: optuna-dashboard sqlite:///{DB_PATH}  →  http://localhost:8080")
    print(f"Trials:    {args.iters}  (20 random warmup, then TPE)\n")
    print("─" * 60)

    study.optimize(objective, n_trials=args.iters, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST  PnL = {best.value:,.1f}  (trial #{best.number})")
    print(f"{'='*60}")

    print("\n── paste into osmiumv2.py ──")
    for k, v in best.user_attrs.items():
        print(f"    {k} = {v}")

    for f in REPO.glob("traders/_opt_osmv2_temp_*.py"):
        f.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
