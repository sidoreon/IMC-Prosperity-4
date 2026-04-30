# r3v03hydroopt.py: vNN counts only inside the `hydroopt` family (same round can have other r3v01… files with different tags).
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
TRADER_SRC = REPO / "traders" / "round3/r3v05hydroema.py"
DB_PATH = REPO / "traders" / "optuna_hydrojoy.db"

source_cache = ""

def define_params(trial: optuna.Trial) -> dict:
    return {
        "POSITION_LIMIT": trial.suggest_categorical("POSITION_LIMIT", [80, 100, 120, 150, 180, 200]),
        "HG_FAIR": trial.suggest_int("HG_FAIR", 9985, 10005),
        "HG_TAKE_EDGE": trial.suggest_int("HG_TAKE_EDGE", 12, 34),
        "HG_TAKE_SIZE": trial.suggest_categorical("HG_TAKE_SIZE", [4, 6, 8, 10, 12, 15, 20, 25]),
        "HG_POST_EDGE": trial.suggest_int("HG_POST_EDGE", 5, 10),
        "HG_POST_SIZE": trial.suggest_categorical("HG_POST_SIZE", [8, 10, 12, 15, 20, 25, 30]),
        "HG_SKEW_DIV": trial.suggest_categorical("HG_SKEW_DIV", [20, 25, 30, 35, 40, 50, 60, 80]),
        "HG_MOM_ALPHA": trial.suggest_categorical("HG_MOM_ALPHA", [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]),
        "HG_MOM_K": trial.suggest_categorical("HG_MOM_K", [0.0, 0.15, 0.25, 0.35, 0.45, 0.60, 0.80]),
        "HG_MOM_CAP": trial.suggest_int("HG_MOM_CAP", 0, 10),
        "HG_MOM_TAKE_BOOST": trial.suggest_categorical("HG_MOM_TAKE_BOOST", [0.0, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]),
    }

def inject_params(source: str, params: dict) -> str:
    out = []
    for line in source.splitlines():
        if '"HYDROGEL_PACK":' in line and "POSITION_LIMIT" in params:
            line = re.sub(r'("HYDROGEL_PACK":\s*)\d+', rf"\g<1>{params['POSITION_LIMIT']}", line)
            out.append(line)
            continue

        replaced = False
        for key, val in params.items():
            if key == "POSITION_LIMIT":
                continue
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
    m = re.search(r"^HYDROGEL_PACK\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)", stdout, re.M)
    if m:
        return float(m.group(1))
    m = re.search(r"^TOTAL\s+-\s+\d+\s+\d+\s+([-\d.]+)", stdout, re.M)
    return float(m.group(1)) if m else float("-inf")

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

    temp_path = REPO / "traders" / f"_opt_hydrojoy_temp_{trial.number}.py"
    temp_path.write_text(inject_params(source_cache, params))

    t0 = time.time()
    pnl = run_backtest(str(temp_path.relative_to(REPO)))
    elapsed = time.time() - t0
    temp_path.unlink(missing_ok=True)

    if pnl == float("-inf"):
        raise optuna.exceptions.TrialPruned()

    print(
        f"Trial {trial.number:>4}  PnL={pnl:>10,.0f}  ({elapsed:.1f}s)  "
        f"lim={params['POSITION_LIMIT']} fair={params['HG_FAIR']} "
        f"take={params['HG_TAKE_EDGE']}/{params['HG_TAKE_SIZE']} "
        f"post={params['HG_POST_EDGE']}/{params['HG_POST_SIZE']} "
        f"mom={params['HG_MOM_ALPHA']}/{params['HG_MOM_K']}"
    )
    return pnl

def main():
    global source_cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--study", type=str, default="hydrojoy_hydrogel")
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

    print(f"Study:    {args.study}")
    print(f"Database: {DB_PATH}")
    print(f"Trials:   {args.iters}  jobs={args.jobs}\n")

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

    print("\nPaste into hydrojoy.py:")
    print("POSITION_LIMITS = {")
    print(f'    "HYDROGEL_PACK": {best.user_attrs["POSITION_LIMIT"]},')
    print("}")
    for k, v in best.user_attrs.items():
        if k != "POSITION_LIMIT":
            print(f"{k} = {v}")

    for f in (REPO / "traders").glob("_opt_hydrojoy_temp_*.py"):
        f.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
