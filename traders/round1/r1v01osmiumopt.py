# r1v01osmiumopt.py: vNN counts only inside the `osmiumopt` family (same round can have other r1v01… files with different tags).
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

REPO  = _repo_root()
CARGO = str(REPO / "scripts" / "cargo_local.sh")

source_cache: str = ""
trader_src: Path  = None

def define_params(trial: optuna.Trial) -> dict:
    params = {
        "HISTORY_WINDOW":    trial.suggest_categorical("HISTORY_WINDOW",    [50, 100, 150, 200, 220, 250, 300]),
        "MIN_HISTORY":       trial.suggest_categorical("MIN_HISTORY",        [5, 10, 15, 20, 25, 30]),
        "EMA_ALPHA":         trial.suggest_categorical("EMA_ALPHA",          [0.20, 0.25, 0.302259, 0.35, 0.40]),
        "VOL_WINDOW_SIZE":   trial.suggest_categorical("VOL_WINDOW_SIZE",    [4, 6, 8, 10, 15, 20]),
        "FAIR_BLEND":        trial.suggest_categorical("FAIR_BLEND",         [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]),
        "TAKE_EDGE":         trial.suggest_int        ("TAKE_EDGE",          1, 4),
        "STRONG_TAKE_EDGE":  trial.suggest_categorical("STRONG_TAKE_EDGE",   [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]),
        "MIN_PENNY_SPREAD":  trial.suggest_categorical("MIN_PENNY_SPREAD",   [2, 3, 4, 5, 6]),
        "PASSIVE_OFFSET":    trial.suggest_categorical("PASSIVE_OFFSET",     [1, 2]),
        "BASE_PASSIVE_SIZE": trial.suggest_categorical("BASE_PASSIVE_SIZE",  [6, 8, 10, 12, 15, 18]),
        "BIG_PASSIVE_SIZE":  trial.suggest_categorical("BIG_PASSIVE_SIZE",   [10, 12, 15, 18, 20, 24]),
        "REDUCE_TRIGGER":    trial.suggest_categorical("REDUCE_TRIGGER",     [20, 25, 30, 40, 50, 60]),
        "REDUCE_SIZE":       trial.suggest_categorical("REDUCE_SIZE",        [2, 3, 4, 6, 8]),
        "INVENTORY_SKEW":    trial.suggest_categorical("INVENTORY_SKEW",     [0.02, 0.04, 0.06, 0.08, 0.10]),
        "LOW_SIGNAL_Z":      trial.suggest_categorical("LOW_SIGNAL_Z",       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "TIGHT_SPREAD":      trial.suggest_categorical("TIGHT_SPREAD",       [2, 3, 4, 5, 6]),
    }
    return params

def inject_params(source: str, params: dict) -> str:
    lines = source.splitlines()
    out = []
    for line in lines:
        replaced = False
        for key, val in params.items():
            m = re.match(rf"^(\s+{re.escape(key)}\s*=\s*)[\d.]+(.*)$", line)
            if m:
                out.append(f"{m.group(1)}{val}")
                replaced = True
                break
        if not replaced:
            out.append(line)
    return "\n".join(out)

def run_backtest(trader_path: str) -> float:
    cmd = [CARGO, "run", "--", "--trader", trader_path,
           "--dataset", "round1", "--products", "summary"]
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        return float("-inf")

    run_dirs = re.findall(r"(runs/backtest-\S+)", result.stdout)
    if not run_dirs:
        return float("-inf")

    total, found = 0.0, 0
    for rel in run_dirs:
        mf = REPO / rel / "metrics.json"
        if not mf.exists():
            continue
        try:
            pnl = json.load(open(mf)).get("final_pnl_by_product", {}).get("ASH_COATED_OSMIUM", 0.0)
            total += pnl
            found += 1
        except Exception:
            pass

    return total if found > 0 else float("-inf")

def objective(trial: optuna.Trial) -> float:
    params = define_params(trial)
    for k, v in params.items():
        trial.set_user_attr(k, v)

    patched = inject_params(source_cache, params)
    tmp = REPO / "traders" / f"_opt_vX_temp_{trial.number}.py"
    tmp.write_text(patched)

    t0  = time.time()
    pnl = run_backtest(str(tmp.relative_to(REPO)))
    tmp.unlink(missing_ok=True)

    if pnl == float("-inf"):
        raise optuna.exceptions.TrialPruned()

    print(f"  Trial {trial.number:>4}  PnL={pnl:>12,.1f}  ({time.time()-t0:.1f}s)")
    return pnl

def main():
    global source_cache, trader_src

    parser = argparse.ArgumentParser()
    parser.add_argument("--trader", required=True,
                        help="Trader file to tune, e.g. traders/osmiumv3.py")
    parser.add_argument("--iters", type=int, default=150)
    args = parser.parse_args()

    trader_src = REPO / args.trader
    if not trader_src.exists():
        print(f"ERROR: {trader_src} not found"); sys.exit(1)

    source_cache = trader_src.read_text()
    study_name = trader_src.stem

    sampler = TPESampler(
        n_startup_trials=30,
        multivariate=True,
        seed=42,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Study:  {study_name}")
    print(f"Trials: {args.iters}  (30 random warmup, multivariate TPE, in-memory)\n")
    print("─" * 60)

    study.optimize(objective, n_trials=args.iters, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST  PnL = {best.value:,.1f}  (trial #{best.number})")
    print(f"{'='*60}")
    print(f"\n── paste into {trader_src.name} ──")
    for k, v in best.user_attrs.items():
        print(f"    {k} = {v}")

    for f in REPO.glob("traders/_opt_vX_temp_*.py"):
        f.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
