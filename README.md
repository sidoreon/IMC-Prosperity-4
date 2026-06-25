# Prosperity 4 — research and submissions

This repository collects **research explorations**, **round-by-round trader experiments**, and **local backtests** for IMC Prosperity 4. The Rust backtester engine lives under `backtester/` and is based on [prosperity_rust_backtester](https://github.com/GeyzsoN/prosperity_rust_backtester). Upstream **Apache-2.0** and **MIT** license texts are in that directory (`backtester/LICENSE-APACHE`, `backtester/LICENSE-MIT`).

## Final Results

- __Prelims__ (Rounds 1 and 2) - India Rank 159
- __Finals__ (Rounds 3, 4 and 5) - India Rank 59 (Global Rank 390)


## Layout

| Path | Contents |
|------|----------|
| `datasets/` | Official round data (tutorial through round 8) and optional `submission.log` |
| `traders/` | Python `Trader` implementations under `traders/tutorial/`, `traders/round1/` … `traders/round5/`; `latest_trader.py` is the backtester default |
| `research/` | Analysis scripts and notes; inputs read from `datasets/` — see `research/README.md` |
| `runs/` | Backtest outputs (`metrics.json`, `submission.log`, …) |
| `backtester/` | Rust crate, `Makefile`, macOS-oriented `scripts/` (`cargo_local.sh`, `doctor_local.sh`), licenses |
| `scripts/` | `verify.sh` (fmt + Rust tests + submit helpers); symlinks to `backtester/scripts/*.sh` for Optuna wrappers |
| `tools/` | Optional portal paste via Playwright + CDP (`submit.py` / `submit_chrome.py`, shared `submit_common.py`) |
| `submissions/` | Submissions for the competition |

## Verification

`cargo test` runs several targets. All **47** backtester unit tests live in the **library** (`src/lib.rs`); the `rust_backtester` **binary** (`src/main.rs`) and **doc-tests** currently define none, so Cargo prints `running 0 tests` for those lines even when everything is fine.

From the repository root:

```bash
./scripts/verify.sh
```

That runs **`cargo fmt --check`** for `backtester/`, then the library tests and Python syntax checks.

Or only Rust (same tests `make test` runs):

```bash
cd backtester && make test
```

The same checks run in **GitHub Actions** (`.github/workflows/verify.yml`).

### Repository health

- **`./scripts/verify.sh`** — `rustfmt --check` for `backtester/`, Rust library tests (**47**), and `py_compile` on `tools/submit*.py`.
- **`scripts/cargo_local.sh`** — Symlinked from `backtester/scripts/` so studies that reference `REPO / "scripts" / "cargo_local.sh"` keep working from the repo root.

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

`requirements.txt` is only for the Playwright submit helpers (`tools/submit.py`, `tools/submit_chrome.py`). Strategy code depends on `datamodel` at runtime via the competition environment or the local backtester. Optuna-based traders need `pip install optuna` separately.

## Backtests

```bash
cd backtester
make tutorial
```

Other targets (`backtest`, `test`, `round3`, …) are defined in `backtester/Makefile`. Full CLI and Docker notes: `backtester/README.md`.

### `rust_backtester` and `datasets/` path

A `rust_backtester` binary installed from **crates.io** (or an older `cargo install`) resolves data next to the crate only, so from `backtester/` it may look for `backtester/datasets/` and fail in this monorepo.

Use any of these:

```bash
# A — run the copy built from this repo (recommended)
cd backtester
./scripts/cargo_local.sh run -- --trader "../traders/round3/r3v01hvoptvol.py" --dataset round3

# B — reinstall the CLI from this tree (monorepo layout + macOS `libpython` rpath)
cd backtester && PYO3_PYTHON="$(which python3)" cargo install --path . --force --locked

# C — stay on crates.io binary: run from repository root (directory that contains `datasets/`)
rust_backtester --trader "traders/round3/r3v01hvoptvol.py" --dataset round3

# D — force workspace root (when using a build that supports PROSPERITY_WORKSPACE_ROOT)
export PROSPERITY_WORKSPACE_ROOT="$PWD"
rust_backtester --trader "traders/round3/r3v01hvoptvol.py" --dataset round3
```

On **macOS**, if `rust_backtester` aborts with `Library not loaded: @rpath/libpython...`, rebuild with the command in **B** above so `build.rs` can embed an rpath to that Python’s `lib/` (Conda `base` is fine as long as `which python3` is the intended interpreter).

## Strategies explored (by round)

Files follow **`r{round}v{variant}{family}.py`**: the numeric **variant** increments within a “family” tag (see the header comment in each file). Suffixes are shorthand for the idea under test:

| Tag | Typical meaning |
|-----|-----------------|
| `core` | Baseline logic for that family (fair + spread / simple structure) |
| `inv` | Inventory-skewed quoting / position-aware market making |
| `vol` | Volatility- or ladder-aware sizing (esp. hydrogel + VEV books) |
| `flow` | Order-flow, microprice, or toxicity tilts on top of a base |
| `ema` | EMA-smoothed fair or signal layer |
| `opt` | **Optuna** study wrapper that injects hyperparameters into another trader source |
| `pair` | Paired or relative-value logic between products |
| `mark` | Extra “mark” or overlay logic on an existing HV / VEV stack |

### Tutorial (`traders/tutorial/`)

Tutorial traders stress-test **take-then-make**, **fair-value models**, and **inventory** on the small tomato + emerald book before the competition rounds scale up.

- Uses the same **`r{round}v{variant}{family}.py`** pattern with **round = 0**; the numeric **variant** increments **within** each **family** (same rule as rounds 1–5; see the header comment in each file).
- **`emtomcore`** — `r0v01emtomcore` matches the spirit of `traders/latest_trader.py`: tiny **one-sided** quotes for quick backtester / portal smoke tests.
- **`imcprac`** — `r0v01imcprac` through `r0v04imcprac`: an EWMA / momentum / **inventory-penalty** progression in the “IMC prosperity trader” style (later variants add hybrid MM + alpha, stronger penalties, and product-specific quote logic).
- **`emerald`** — `r0v01emerald` through `r0v06emerald`: **take-then-make** and **tomato-fair** iterations (EMA, momentum, spread placement, position limits). **`r0v06emerald`** documents post-CSV findings (e.g. passive-only emeralds when the tape shows no profitable takes, **mean-reversion** on tomatoes, top-of-book quoting).
- **`allin`** — `r0v01allin` and `r0v02allin`: take-then-make with **class-level** constants vs a per-product **`PARAMS`** table; **`r0v02allin`** fixes tomato fair value (EMA) vs a wrong hardcoded placeholder.
- **`emtomparam`** (`r0v01emtomparam`) — two-pass take-then-make driven by one **`PARAMS` map per product** (momentum tilt on tomato EMA fair) for grid search / tuning.
- **`emtomrw`** (`r0v01emtomrw`) — treats **TOMATOES** as a random walk: earn the spread, strong inventory skew, no crossing; **EMERALDS** still take misprices vs the pegged fair.
- **`emtomimitate`** (`r0v01emtomimitate`) — shapes behavior toward **observed backtest** patterns (aggressive emerald takes + top-of-book tomato MM, sparse takes vs EMA).
- **`emtomanchor`** (`r0v01emtomanchor`) — take-then-make **anchored to known fair values** from the spec (same logical template for every listed product).
- **`emtomlinreg`** (`r0v01emtomlinreg`) — **rolling linear regression** on tomato mids for a dynamic fair; **EMERALDS** use the fixed fair from the spec.
- Exploratory notebook: **`research/r0_notebook_prosper0.ipynb`**.

### Round 1 — Osmium + Intarian pepper

Round 1 is a **two-name** regime: one metal and one agricultural root, each with its own microstructure but shared tooling (fair, ladder, flow).

- Two **parallel** books: **ASH_COATED_OSMIUM** (families tagged **`osmium*`**) and **INTARIAN_PEPPER_ROOT** (families tagged **`peposm*`**).
- **Fair construction** — **EMA / blend** mids with tunable **history windows** so quotes track slower or faster tape without chasing every tick.
- **Passive depth** — **ladder**-style sizes and price **offsets** away from touch to earn spread while controlling adverse selection.
- **Execution style** — **take** (cross) vs **penny** (improve) at the edge; families vary how aggressively each name leans into one side.
- **Risk and tilt** — **inventory skew** (`*inv`) pulls quotes when position builds; **volatility-gated** tightness avoids quoting too tight in noisy regimes; **order-flow / microprice** tilts (`*flow`) lean bids and asks with short-horizon pressure; **EMA-smoothed** layers (`*ema`) damp fair-value jitter before it hits the quote engine.
- **Search** — **`opt`** files wrap base traders and run **Optuna** sweeps over spread, skew, window, and ladder knobs instead of hand-tuning every scalar.
- **Relative value** — **`pair`** files express **spread / ratio** ideas between osmium and pepper (and related structures) so one leg can hedge or lean on the other.
- **Naming** — Within each book, **`core`** is the baseline MM stack; suffixes in the table above (`inv`, `vol`, `flow`, `ema`, `opt`, `pair`) flag which layer dominates that file.

### Round 2 — Same two books, round-2 tape

Round 2 reuses the **osmium / pepper** split on **new round-2 data**: same strategy families, different noise and regime, so **`inv`** and **`pair`** get more mileage.

- Continues the **osmium** and **peposm** families on the **round-2** tape with more **`inv`**, **`pair`**, and **`ema`** variants.
- **Inventory-first** — Many variants stress **position limits** and **skew** because round-2 fills can stack inventory faster than round-1 intuition suggests.
- **`opt`** shells tune **round-2** inventory-focused strategies (**TPESampler**, SQLite study DB paths under `traders/`, subprocess backtests to `rust_backtester`).
- **Workflow** — Studies typically **spawn** `rust_backtester` per trial so each Optuna draw gets a full deterministic replay on the official tape.

### Round 3 — Hydrogel, velvet, VEV ladder, cross-asset

Round 3 adds **hydrogel**, **velvet**, and a full **voucher ladder** (`VEV_*`), so strategies must **route risk** across spot-like names, an extract, and many strikes at once.

- **Hydrogel (`HYDROGEL_PACK`)** — **`hydrocore`**: mean-reversion-style fair and edge (lean back toward a central fair when the book dislocates); **`hydroema`**: smoothing on that stack so hydrogel quotes do not flicker with every mid update; **`hydroopt`** / **`hydroflow`**: Optuna or flow-style overlays on top of the hydro logic for sizing and tilt.
- **Velvet + vouchers (`VELVETFRUIT_EXTRACT`, `VEV_*`)** — **`velvetcore`**: **intrinsic** / **strike-based** quoting so extract and each voucher line has a coherent notion of cheap vs rich; **`velvetvol`** and **`velvetopt`**: volatility- and search-driven behavior when smile or wing risk matters; **`velvetmark`**: **mark** logic isolated in its own family so mark experiments do not entangle the core velvet fair.
- **Joint HV + ladder (`hvopt*`)** — Large family tying **hydrogel + velvet + vouchers** into one codebase path: **`hvoptcore`** combines **OFI-skewed** quoting on velvet with **passive hydrogel** so flow and mean-reversion roles stay separated by product; **`hvoptvol`** (many variants) adds **ladder-wide** position limits, **momentum** on hydrogel, and **vol- / smile-aware** voucher handling (including “robust” / profit-lock style notes in comments); **`hvoptinv`** stresses **inventory** across many **`VEV_*`** lines so no single strike silently dominates the book; **`hvoptpair`**, **`hvoptema`**, **`hvoptflow`** specialize **pair**, **EMA**, and **flow** layers without rewriting the whole HV stack each time.
- **Cross-asset (`asset*`)** — Quote **all visible products** from one trader: **`assetcore`-style** breadth (shared framework per name), plus **`assetema`**, **`assetinv`**, **`assetflow`**, and **`assetopt`** (Optuna over a deep **`hvoptinv`** / **`hvoptvol`** base) for universe-wide sweeps.
- **Research** — Correlation and option-structure notes for this round often live under `research/` (e.g. scripts comparing round-2 vs round-3 behavior) alongside the `hvopt*` traders.

### Round 4 — Richer HV / VEV + cross-asset

Round 4 keeps the **round-3 instrument set** but pushes **mark models**, **cross-strike** behavior, and **full-ladder** quoting further—especially where vouchers and velvet interact.

- **Same instrument set as round 3**, with more **mark-model** work and **breadth** experiments across hydrogel, velvet, and the full voucher ladder.
- **`assetcore`** — **Mid ± edge** quoting with an explicit **inventory penalty** so position naturally mean-reverts; **dislocation takes** fire when the visible book is far from fair and passive quotes would leave money on the table.
- **`hvoptmark`** / **`assetmark`** — Layer **mark-flow** ideas on hydrogel + velvet + vouchers: mark-driven overlays sit on top of the HV stack to react when internal marks disagree with touch.
- **`hvoptvol`** — Continues **vol-aware** market making on the joint book (wing vs ATM risk, clip sizes, and vol-regime gating vary by variant).
- **`vevcore`** — Focuses **ATM vs wing** voucher routing and **strike-specific** behavior so capital is not spread evenly across strikes that have different edge and toxicity.
- **`velvetmark`** — Tightens **velvet-side** mark overlays relative to the core velvet families—useful when extract fair is the bottleneck for the whole ladder.
- **Theme** — Round 4 files often read as “round 3 HV/VEV, plus more explicit **mark** and **routing** discipline.”

### Round 5 — Wide SKU universe

Round 5 explodes the **product count**: many unrelated SKUs, each with its own limits, liquidity, and failure mode—strategies emphasize **filters**, **caps**, and **which names to touch at all**.

- Many **named products** (e.g. galaxy sounds, microchips, panels, robots, sleep pods) with round-5-specific limits and books.
- **`round5vol`** — **Realized-vol**, **microprice**, and **markout-delay** style features drive sizing; **hard caps** and per-order **clips** prevent one toxic print from blowing the book; explicit **AVOID** sets skip names that backtests flag as fragile or one-sided.
- **`round5ema`** — Adds an **EMA / trend** signal layer on top of the same broad universe so trending names get directional tilt without a separate trader per product.
- **`round5pair`** — **Ratio-history** (cointegration / residual) trades between **configured pairs**: spreads mean-revert when the ratio leaves its band.
- **`emtomcore`** — Carries a **broad `LIMITS`** table (**legacy** tutorial names + round-5 SKUs + vouchers) as **scaffolding** for wide-universe runs so one file can legally quote every listed line.
- **`assetvol`** — Ties **cross-asset** quoting to **vol-style** sizing and risk knobs across many products at once—closer to a single risk budget across the whole SKU list than per-name hand tuning.
- **Operational picture** — Successful round-5 stacks usually combine **who to trade** (filters / avoids), **how big** (vol- and clip-aware sizing), and **how directional** (EMA / pair overlays).

## Portal submit helpers

Start a Chromium-based browser with `--remote-debugging-port=9222`, then run `python tools/submit.py <trader.py>` (Arc) or `python tools/submit_chrome.py <trader.py>`. Use `python tools/submit.py --help` for `--inspect` and `SUBMIT_CDP_URL` if the CDP port differs. Inspect captures go under `tools/inspect_captures/` (gitignored).

## Credits

Upstream backtester: **prosperity_rust_backtester** (GeyzsoN on GitHub). Everything else here is part of a personal Prosperity 4 workspace.

## License

The **rust_backtester** crate is under the upstream **Apache-2.0** and **MIT** terms; see `backtester/LICENSE-APACHE` and `backtester/LICENSE-MIT`. Traders, research notes, and other original material in this repository are not covered by those licenses unless stated otherwise in those files.
