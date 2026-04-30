# r2v02osmiumopt.py: vNN counts only inside the `osmiumopt` family (same round can have other r2v01… files with different tags).
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

REPO       = _repo_root()
TRADER_SRC = REPO / "traders" / "round2/r2v01peposmpair.py"
CARGO      = str(REPO / "scripts" / "cargo_local.sh")
DB_PATH    = REPO / "traders" / "optuna_nigar.db"

source_cache: str = ""

def define_params(trial: optuna.Trial) -> dict:
    n = trial.suggest_int("ema_n", 50, 500)
    return {
        "ema_n":             n,
        "OSMIUM_INV_SKEW":   trial.suggest_categorical("OSMIUM_INV_SKEW",  [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]),
        "OSMIUM_TAKE_WIDTH": trial.suggest_categorical("OSMIUM_TAKE_WIDTH", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    }

def inject_params(source: str, params: dict) -> str:
    # Injects OSMIUM_INV_SKEW and OSMIUM_TAKE_WIDTH as plain floats. Injects OSMIUM_EMA_ALPHA as 2.0 / {n}.0 to keep the ...
    n = params["ema_n"]
    lines = source.splitlines()
    out = []
    for line in lines:
        replaced = False

        if "OSMIUM_EMA_ALPHA" in line:
            new_line = re.sub(r"2\.0\s*/\s*\d+\.0", f"2.0 / {n}.0", line)
            out.append(new_line)
            replaced = True

        if not replaced:
            for key in ("OSMIUM_INV_SKEW", "OSMIUM_TAKE_WIDTH"):
                if key not in params:
                    continue
                pattern = rf"^(\s*{re.escape(key)}\s*=\s*)[\d.]+(.*)$"
                m = re.match(pattern, line)
                if m:
                    out.append(f"{m.group(1)}{params[key]}{m.group(2)}")
                    replaced = True
                    break

        if not replaced:
            out.append(line)

    return "\n".join(out)

def run_backtest(trader_path: str) -> float:
    cmd = [
        CARGO, "run", "--",
        "--trader", trader_path,
        "--dataset", "round2",
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

def objective(trial: optuna.Trial) -> float:
    params = define_params(trial)

    for k, v in params.items():
        trial.set_user_attr(k, v)

    patched = inject_params(source_cache, params)
    temp_path = REPO / "traders" / f"_opt_nigar_temp_{trial.number}.py"
    temp_path.write_text(patched)

    t0 = time.time()
    pnl = run_backtest(str(temp_path.relative_to(REPO)))
    elapsed = time.time() - t0

    temp_path.unlink(missing_ok=True)

    if pnl == float("-inf"):
        raise optuna.exceptions.TrialPruned()

    print(
        f"  Trial {trial.number:>4}  OsmPnL={pnl:>10,.1f}  ({elapsed:.1f}s)"
        f"  n={params['ema_n']}  skew={params['OSMIUM_INV_SKEW']}  width={params['OSMIUM_TAKE_WIDTH']}"
    )

    return pnl

def main():
    global source_cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--jobs",  type=int, default=1)
    parser.add_argument("--study", type=str, default="nigar_osmium")
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

    print(f"Optuna study: 'nigar_osmium'")
    print(f"Database:     {DB_PATH}")
    print(f"Dashboard:    optuna-dashboard {storage}  ->  http://localhost:8080")
    print(f"Trials:       {args.iters}  (jobs={args.jobs})")
    print(f"Sampler:      TPE (20 random warmup)\n")
    print(f"Objective:    ASH_COATED_OSMIUM PnL (3-day sum, round2)\n")
    print("─" * 70)

    study.optimize(
        objective,
        n_trials=args.iters,
        n_jobs=args.jobs,
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"BEST  OsmPnL = {best.value:,.1f}  (trial #{best.number})")
    print("=" * 60)
    for k, v in best.user_attrs.items():
        print(f"  {k} = {v}")

    n = best.user_attrs["ema_n"]
    print("\n── paste into nigar.py ─────────────────────────────────────")
    print(f"  OSMIUM_EMA_ALPHA = 2.0 / {n}.0")
    print(f"  OSMIUM_INV_SKEW  = {best.user_attrs['OSMIUM_INV_SKEW']}")
    print(f"  OSMIUM_TAKE_WIDTH = {best.user_attrs['OSMIUM_TAKE_WIDTH']}")

    for f in REPO.glob("traders/_opt_nigar_temp_*.py"):
        f.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
