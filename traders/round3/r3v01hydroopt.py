# r3v01hydroopt.py: vNN counts only inside the `hydroopt` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; Optuna-tuned parameter shell around base.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List
import numpy as np
from collections import deque

LIMIT             = 180
Z_ENTRY           = 3.0
Z_EXIT_PARTIAL    = 1.0
MM_BASE_SIZE      = 20
MM_REDUCED_SIZE   = 10
MM_SPREAD_MIN     = 4
MM_SPREAD_MAX     = 8
TREND_THRESHOLD   = 5.0
SLOPE_THRESHOLD   = 1.5

class Trader:

    def __init__(self):
        self.product = "HYDROGEL_PACK"
        self.limit = LIMIT
        self.mid_history = deque(maxlen=200)
        self.dev_history = deque(maxlen=500)
        self.price_change_history = deque(maxlen=50)

    def run_hydrogel(self, state: TradingState):
        product = self.product
        orders: List[Order] = []

        if product not in state.order_depths:
            return orders

        od = state.order_depths[product]
        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2
        position = state.position.get(product, 0)

        if len(self.mid_history) > 0:
            self.price_change_history.append(mid - self.mid_history[-1])
        self.mid_history.append(mid)

        if len(self.mid_history) < 100:
            return orders

        ma50  = np.mean(list(self.mid_history)[-50:])
        ma200 = np.mean(self.mid_history)
        fair  = 0.7 * ma50 + 0.3 * ma200

        short_ma     = np.mean(list(self.mid_history)[-20:])
        long_ma      = np.mean(list(self.mid_history)[-100:])
        trend        = short_ma - long_ma
        slope        = np.mean(np.diff(list(self.mid_history)[-20:]))
        strong_trend = abs(trend) > TREND_THRESHOLD or abs(slope) > SLOPE_THRESHOLD

        deviation = mid - fair
        self.dev_history.append(deviation)

        sigma    = np.std(self.dev_history) if len(self.dev_history) > 50 else 10
        z        = deviation / sigma if sigma > 0 else 0
        momentum = np.mean(self.price_change_history) if len(self.price_change_history) > 10 else 0

        long_cap  = self.limit - position
        short_cap = self.limit + position

        if strong_trend:
            if trend < 0 and position > 0:
                orders.append(Order(product, best_bid, -min(position, 50)))
            elif trend > 0 and position < 0:
                orders.append(Order(product, best_ask, min(-position, 50)))

        if position > 0 and z > -Z_EXIT_PARTIAL:
            orders.append(Order(product, best_bid, -min(position // 2, MM_BASE_SIZE)))
        elif position < 0 and z < Z_EXIT_PARTIAL:
            orders.append(Order(product, best_ask, min((-position) // 2, MM_BASE_SIZE)))

        allow_directional = not (strong_trend and abs(z) < 3.5)
        if allow_directional and abs(position) < 160:
            if z > Z_ENTRY and momentum < 0.5:
                size = min(25, short_cap)
                if size > 0:
                    orders.append(Order(product, best_bid, -size))
            elif z < -Z_ENTRY and momentum > -0.5:
                size = min(25, long_cap)
                if size > 0:
                    orders.append(Order(product, best_ask, size))

        if strong_trend:
            if trend > 0 and z < -2:
                size = min(MM_BASE_SIZE, long_cap)
                if size > 0:
                    orders.append(Order(product, best_ask, size))
            elif trend < 0 and z > 2:
                size = min(MM_BASE_SIZE, short_cap)
                if size > 0:
                    orders.append(Order(product, best_bid, -size))

        spread  = int(max(MM_SPREAD_MIN, min(MM_SPREAD_MAX, 0.5 * sigma)))
        size    = MM_BASE_SIZE if abs(position) <= 100 else MM_REDUCED_SIZE
        bid_off = -spread
        ask_off = +spread
        if strong_trend:
            if trend > 0:
                ask_off += 2
            else:
                bid_off -= 2

        if long_cap > 0:
            orders.append(Order(product, int(fair + bid_off), min(size, long_cap)))
        if short_cap > 0:
            orders.append(Order(product, int(fair + ask_off), -min(size, short_cap)))

        return orders

    def run(self, state: TradingState):
        result = {}
        orders = self.run_hydrogel(state)
        if orders:
            result[self.product] = orders
        return result, 0, ""

if __name__ == "__main__":
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
    DB     = REPO / "traders" / "optuna_loo.db"
    SRC    = Path(__file__)

    source_cache: str = ""

    def define_params(trial: optuna.Trial) -> dict:
        spread_min = trial.suggest_int("MM_SPREAD_MIN", 2, 8)
        spread_max = trial.suggest_int("MM_SPREAD_MAX", spread_min + 1, 20)
        return {
            "Z_ENTRY":         trial.suggest_float("Z_ENTRY",         1.5, 5.0,  step=0.25),
            "Z_EXIT_PARTIAL":  trial.suggest_float("Z_EXIT_PARTIAL",  0.0, 2.5,  step=0.25),
            "TREND_THRESHOLD": trial.suggest_float("TREND_THRESHOLD", 2.0, 12.0, step=0.5),
            "SLOPE_THRESHOLD": trial.suggest_float("SLOPE_THRESHOLD", 0.5, 4.0,  step=0.25),
            "MM_SPREAD_MIN":   spread_min,
            "MM_SPREAD_MAX":   spread_max,
            "MM_BASE_SIZE":    trial.suggest_int("MM_BASE_SIZE", 5, 40),
        }

    def inject_params(source: str, params: dict) -> str:
        lines = source.splitlines()
        out = []
        for line in lines:
            replaced = False
            for key, val in params.items():
                m = re.match(rf"^({re.escape(key)}\s*=\s*)[\d.]+(.*)$", line)
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
                pnl += json.load(open(mf)).get("final_pnl_by_product", {}).get("HYDROGEL_PACK", 0.0)
                found += 1
            except Exception:
                pass
        return pnl if found > 0 else float("-inf")

    def objective(trial: optuna.Trial) -> float:
        params = define_params(trial)
        for k, v in params.items():
            trial.set_user_attr(k, v)
        patched  = inject_params(source_cache, params)
        tmp      = REPO / "traders" / f"_loo_tmp_{trial.number}.py"
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
    parser.add_argument("--study", type=str, default="loo_hydrogel")
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
    print("── paste into loo.py " + "─" * 40)
    for k, v in best.user_attrs.items():
        print(f"  {k} = {v}")

    for f in REPO.glob("traders/_loo_tmp_*.py"):
        f.unlink(missing_ok=True)
