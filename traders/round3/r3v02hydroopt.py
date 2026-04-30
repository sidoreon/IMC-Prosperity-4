# r3v02hydroopt.py: vNN counts only inside the `hydroopt` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; Optuna-tuned parameter shell around base.
#
import argparse
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
from optuna.trial import TrialState

REPO = _repo_root()
TRADER_SRC = REPO / "traders" / "round3/r3v02hydroema.py"
DB_PATH = REPO / "traders" / "optuna_esnu.db"

source_cache = ""

def define_params(trial: optuna.Trial) -> dict:
    static_w = trial.suggest_categorical("FAIR_STATIC_W", [0.45, 0.55, 0.65, 0.75, 0.85])
    return {
        "HYDROGEL_HS": trial.suggest_int("HYDROGEL_HS", 5, 9),
        "HYDROGEL_SKEW": trial.suggest_categorical("HYDROGEL_SKEW", [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]),
        "HYDROGEL_SIZE": trial.suggest_categorical("HYDROGEL_SIZE", [15, 20, 25, 30, 35]),
        "HYDROGEL_TAKE_SIZE": trial.suggest_categorical("HYDROGEL_TAKE_SIZE", [4, 6, 8, 10, 12, 15]),
        "MIN_MAKE_SPREAD": trial.suggest_int("MIN_MAKE_SPREAD", 12, 16),
        "SLOW_ALPHA": trial.suggest_categorical("SLOW_ALPHA", [0.003, 0.005, 0.008, 0.01, 0.015, 0.02]),
        "FAST_ALPHA": trial.suggest_categorical("FAST_ALPHA", [0.04, 0.06, 0.08, 0.10, 0.12, 0.16]),
        "FAIR_STATIC_W": static_w,
        "FAIR_SLOW_W": round(1.0 - static_w, 10),
        "FAIR_TAKE_EDGE": trial.suggest_int("FAIR_TAKE_EDGE", 0, 8),
        "FAST_TAKE_EDGE": trial.suggest_int("FAST_TAKE_EDGE", 5, 12),
        "GAP_TRIGGER": trial.suggest_int("GAP_TRIGGER", 4, 9),
        "GAP_BIAS": trial.suggest_int("GAP_BIAS", 0, 3),
    }

def inject_params(source: str, params: dict) -> str:
    out = []
    for line in source.splitlines():
        replaced = False
        for key, val in params.items():
            pattern = rf"^(\s*{re.escape(key)}\s*=\s*)[-\d.]+(.*)$"
            match = re.match(pattern, line)
            if match:
                out.append(f"{match.group(1)}{val}{match.group(2)}")
                replaced = True
                break
        if not replaced:
            out.append(line)
    return "\n".join(out) + "\n"

def parse_pnl(stdout: str) -> float:
    product_match = re.search(r"^HYDROGEL_PACK\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)", stdout, re.M)
    if product_match:
        return float(product_match.group(1))
    total_match = re.search(r"^TOTAL\s+-\s+\d+\s+\d+\s+([-\d.]+)", stdout, re.M)
    if total_match:
        return float(total_match.group(1))
    return float("-inf")

def run_backtest(trader_path: str) -> float:
    result = subprocess.run(
        ["rust_backtester", "--trader", trader_path],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        return float("-inf")
    return parse_pnl(result.stdout)

def objective(trial: optuna.Trial) -> float:
    params = define_params(trial)
    for k, v in params.items():
        trial.set_user_attr(k, v)

    temp_path = REPO / "traders" / f"_opt_esnu_temp_{trial.number}.py"
    temp_path.write_text(inject_params(source_cache, params))

    t0 = time.time()
    pnl = run_backtest(str(temp_path.relative_to(REPO)))
    elapsed = time.time() - t0
    temp_path.unlink(missing_ok=True)

    if pnl == float("-inf"):
        raise optuna.exceptions.TrialPruned()

    print(
        f"Trial {trial.number:>4}  PnL={pnl:>10,.0f}  ({elapsed:.1f}s)  "
        f"hs={params['HYDROGEL_HS']} skew={params['HYDROGEL_SKEW']} "
        f"size={params['HYDROGEL_SIZE']} take={params['HYDROGEL_TAKE_SIZE']} "
        f"spread={params['MIN_MAKE_SPREAD']} fair={params['FAIR_STATIC_W']:.2f}"
    )
    return pnl

def main():
    global source_cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--study", type=str, default="esnu_hydrogel")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if not TRADER_SRC.exists():
        print(f"ERROR: {TRADER_SRC} not found", file=sys.stderr)
        sys.exit(1)

    source_cache = TRADER_SRC.read_text()
    storage = f"sqlite:///{DB_PATH}"

    if args.fresh:
        try:
            optuna.delete_study(study_name=args.study, storage=storage)
            print(f"Deleted existing study '{args.study}'")
        except Exception:
            pass

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=30, seed=42),
        load_if_exists=True,
    )

    print(f"Study:     {args.study}")
    print(f"Database:  {DB_PATH}")
    print(f"Trials:    {args.iters}  jobs={args.jobs}")
    print(f"Baseline:  esnu.py was 41,471 on round3\n")

    study.optimize(objective, n_trials=args.iters, n_jobs=args.jobs, show_progress_bar=True)

    complete = study.get_trials(states=(TrialState.COMPLETE,))
    if not complete:
        print("\nNo completed trials. Check that `rust_backtester` runs from the repo root.")
        sys.exit(1)

    best = study.best_trial
    print("\n" + "=" * 64)
    print(f"BEST HYDROGEL PnL = {best.value:,.0f}  trial #{best.number}")
    print("=" * 64)
    for k, v in best.user_attrs.items():
        print(f"{k} = {v}")

    print("\nPaste into esnu.py:")
    for k, v in best.user_attrs.items():
        print(f"    {k} = {v}")

    for f in (REPO / "traders").glob("_opt_esnu_temp_*.py"):
        f.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
