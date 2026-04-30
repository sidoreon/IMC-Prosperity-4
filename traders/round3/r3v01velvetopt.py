# r3v01velvetopt.py: vNN counts only inside the `velvetopt` family (same round can have other r3v01… files with different tags).
# Velvetfruit-focused; Optuna shell.
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

REPO   = _repo_root()
CARGO  = str(REPO / "scripts" / "cargo_local.sh")
DB     = REPO / "traders" / "optuna_randwajoy_velvet.db"
SRC    = REPO / "traders" / "round3/r3v13hvoptvol.py"

TARGET_PRODUCTS = {
    "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
}

source_cache: str = ""

def define_params(trial: optuna.Trial) -> dict:
    return {
        "VE_ANCHOR_WEIGHT": trial.suggest_float("VE_ANCHOR_WEIGHT", 0.10, 0.90, step=0.02),
        "VE_EMA_ALPHA":     trial.suggest_float("VE_EMA_ALPHA",     0.02, 0.35, step=0.01),
        "VE_IMBALANCE_K":   trial.suggest_float("VE_IMBALANCE_K",   0.50, 6.00, step=0.25),
        "VE_IMBALANCE_CAP": trial.suggest_float("VE_IMBALANCE_CAP", 0.50, 6.00, step=0.25),
        "VE_EDGE":          trial.suggest_int  ("VE_EDGE",           2,    20),
        "VE_SIZE":          trial.suggest_int  ("VE_SIZE",           3,    30),
        "VE_POST_EDGE":     trial.suggest_int  ("VE_POST_EDGE",      1,     8),
        "VE_POST_SIZE":     trial.suggest_int  ("VE_POST_SIZE",      5,    50),
    }

def inject_params(source: str, params: dict) -> str:
    lines = source.splitlines()
    out = []
    for line in lines:
        replaced = False
        for key, val in params.items():

            m = re.match(rf"^(\s+{re.escape(key)}\s*=\s*)[\d.]+(.*)$", line)
            if m:
                out.append(f"{m.group(1)}{val}{m.group(2)}")
                replaced = True
                break
        if not replaced:
            out.append(line)
    return "\n".join(out)

def run_backtest(path: str) -> float:
    r = subprocess.run(
        [CARGO, "run", "--", "--trader", path, "--dataset", "round3", "--products", "summary"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return float("-inf")
    pnl = 0.0
    found = 0
    for rel in re.findall(r"(runs/backtest-\S+)", r.stdout):
        mf = REPO / rel / "metrics.json"
        if not mf.exists():
            continue
        try:
            by_product = json.load(open(mf)).get("final_pnl_by_product", {})
            for prod, val in by_product.items():
                if prod in TARGET_PRODUCTS:
                    pnl += float(val)
            found += 1
        except Exception:
            pass
    return pnl if found > 0 else float("-inf")

def objective(trial: optuna.Trial) -> float:
    params = define_params(trial)
    for k, v in params.items():
        trial.set_user_attr(k, v)
    patched = inject_params(source_cache, params)
    tmp = REPO / "traders" / f"_rv_tmp_{trial.number}.py"
    tmp.write_text(patched)
    t0  = time.time()
    pnl = run_backtest(str(tmp.relative_to(REPO)))
    tmp.unlink(missing_ok=True)
    if pnl == float("-inf"):
        raise optuna.exceptions.TrialPruned()
    print(f"  [{trial.number:>4}]  PnL={pnl:>10,.0f}  ({time.time()-t0:.1f}s)  "
          + "  ".join(f"{k}={v}" for k, v in params.items()))
    return pnl

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--jobs",  type=int, default=1)
parser.add_argument("--study", type=str, default="randwajoy_velvet")
parser.add_argument("--fresh", action="store_true")
args = parser.parse_args()

source_cache = SRC.read_text()
storage = f"sqlite:///{DB}"

if args.fresh and DB.exists():
    try:
        optuna.delete_study(study_name=args.study, storage=storage)
    except Exception:
        pass

study = optuna.create_study(
    study_name=args.study, storage=storage, direction="maximize",
    sampler=TPESampler(n_startup_trials=20, seed=42), load_if_exists=True,
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"Study: {args.study}  |  DB: {DB}")
print(f"Trials: {args.iters}  |  Dashboard: optuna-dashboard {storage}\n")
print("─" * 70)

study.optimize(objective, n_trials=args.iters, n_jobs=args.jobs, show_progress_bar=True)

best = study.best_trial
print(f"\nBEST  PnL={best.value:,.0f}  (trial #{best.number})")
print("── paste into randwajoy_velvet.py ─" + "─" * 30)
for k, v in best.user_attrs.items():
    print(f"    {k} = {v}")

for f in REPO.glob("traders/_rv_tmp_*.py"):
    f.unlink(missing_ok=True)
