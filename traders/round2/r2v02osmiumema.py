# r2v02osmiumema.py: vNN counts only inside the `osmiumema` family (same round can have other r2v01… files with different tags).
# Osmium book; EMA fair.
#
import json
import re
import subprocess
import time
from pathlib import Path

def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cur in p.parents:
        if cur.name == "traders":
            return cur.parent
    return p.parent.parent

REPO       = _repo_root()
TRADER_SRC = REPO / "traders" / "round2/r1v01peposmema.py"
CARGO      = str(REPO / "scripts" / "cargo_local.sh")

SWEEP = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

source = TRADER_SRC.read_text()

def inject(n: int) -> Path:
    patched = re.sub(r"(WARMUP_TICKS\s*=\s*)\d+", rf"\g<1>{n}", source)
    p = REPO / "traders" / f"_fresh_sweep_{n}.py"
    p.write_text(patched)
    return p

def run(path: Path) -> float:
    result = subprocess.run(
        [CARGO, "run", "--", "--trader", str(path.relative_to(REPO)),
         "--dataset", "round2", "--products", "summary"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return float("-inf")

    total, found = 0.0, 0
    for rel_dir in re.findall(r"(runs/backtest-\S+)", result.stdout):
        mf = REPO / rel_dir / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text())
            total += m.get("final_pnl_by_product", {}).get("ASH_COATED_OSMIUM", 0.0)
            found += 1
        except Exception:
            pass
    return total if found > 0 else float("-inf")

print(f"{'WARMUP_TICKS':>14}  {'OSM PnL (3-day)':>18}")
print("─" * 36)

results = []
for n in SWEEP:
    p = inject(n)
    t0 = time.time()
    pnl = run(p)
    p.unlink(missing_ok=True)
    results.append((n, pnl))
    print(f"{n:>14}  {pnl:>18,.1f}  ({time.time()-t0:.1f}s)")

best_n, best_pnl = max(results, key=lambda x: x[1])
print("─" * 36)
print(f"BEST: WARMUP_TICKS = {best_n}  →  {best_pnl:,.1f}")

for p in REPO.glob("traders/_fresh_sweep_*.py"):
    p.unlink(missing_ok=True)
