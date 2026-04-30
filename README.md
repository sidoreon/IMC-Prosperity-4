# Prosperity 4 — research and submissions

This repository collects **research explorations**, **round-by-round trader experiments**, and **local backtests** for IMC Prosperity 4. The Rust backtester engine lives under `backtester/` and is based on [prosperity_rust_backtester](https://github.com/GeyzsoN/prosperity_rust_backtester). Upstream **Apache-2.0** and **MIT** license texts are in that directory (`backtester/LICENSE-APACHE`, `backtester/LICENSE-MIT`).

## Layout

| Path | Contents |
|------|----------|
| `datasets/` | Official round data (tutorial through round 8) and optional `submission.log` |
| `traders/` | Python `Trader` implementations under `traders/tutorial/`, `traders/round1/` … `traders/round5/`; `latest_trader.py` is the backtester default |
| `research/` | Analysis scripts and notes; inputs read from `datasets/` — see `research/README.md` |
| `runs/` | Backtest outputs (`metrics.json`, `submission.log`, …) |
| `backtester/` | Rust crate, `Makefile`, macOS-oriented `scripts/` (`cargo_local.sh`, `doctor_local.sh`), licenses |
| `scripts/` | `verify.sh` (fmt + Rust tests + submit helpers); symlinks to `backtester/scripts/*.sh` for Optuna wrappers |
| `tools/submit.py` | Playwright + **Arc** (CDP) — paste a trader into the portal |
| `tools/submit_chrome.py` | Same flow for **Google Chrome** (CDP) |
| `tools/submit_common.py` | Shared CDP / navigation logic for the two helpers |

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

**Tutorial (`traders/tutorial/`)** — Same **`r{round}v{variant}{family}.py`** pattern with **round = 0**; **variant** counts upward within each **family** (same rule as rounds 1–5). **`r0v01emtomcore`** matches the spirit of `traders/latest_trader.py` (tiny one-sided smoke-test quotes). **`imcprac`** — `r0v01imcprac` … `r0v04imcprac`: EWMA / momentum / inventory-penalty progression (“IMC prosperity trader”). **`emerald`** — `r0v01emerald` … `r0v06emerald`: take-then-make and tomato-fair iterations; **`r0v06emerald`** documents post-CSV tweaks (no emerald take pass, mean-reversion on tomatoes). **`allin`** — `r0v01allin`, `r0v02allin`: take-then-make with class-level vs `PARAMS` layout (`r0v02allin` fixes tomato fair vs a wrong hardcoded constant). **`emtomparam`** (`r0v01emtomparam`): one `PARAMS` map per product for tuning. **`emtomrw`**, **`emtomimitate`**, **`emtomanchor`**, **`emtomlinreg`** — `r0v01emtomrw` (random-walk tomato MM), `r0v01emtomimitate` (match observed backtest shapes), `r0v01emtomanchor` (known-fair anchor MM), `r0v01emtomlinreg` (rolling linear regression on tomato mids). Notebook: **`research/r0_notebook_prosper0.ipynb`**.

**Round 1 — Osmium + Intarian pepper** — Two parallel books: **ASH_COATED_OSMIUM** (`osmium*`) and **INTARIAN_PEPPER_ROOT** (`peposm*`). Explored **EMA / blend fair** with history windows, **passive ladder** sizes and offsets, **take** vs **penny** edges, **inventory skew**, and **volatility**-gated tight quotes (`*inv`, `*flow`, `*ema`). **`opt`** files wrap base traders and sweep Optuna over those knobs. **`pair`** files test relative-value style legs between instruments.

**Round 2 — Same two books, round-2 tape** — Continues **osmium** and **peposm** families with more **`inv`**, **`pair`**, and **`ema`** variants; **`opt`** shells tune **round-2** inventory strategies (TPESampler, SQLite DB paths under `traders/`).

**Round 3 — Hydrogel, velvet, VEV ladder, cross-asset** — Main product groups:

- **Hydrogel (`HYDROGEL_PACK`)** — `hydrocore` mean-reversion-style fair and edge; `hydroema` adds smoothing; `hydroopt` / `hydroflow` add search shells or flow-style overlays.
- **Velvet + vouchers (`VELVETFRUIT_EXTRACT`, `VEV_*`)** — `velvetcore` intrinsic / strike-based quoting; `velvetvol` and `velvetopt` push vol- and search-driven behavior; `velvetmark` where mark logic is isolated.
- **Joint HV + ladder (`hvopt*`)** — Large family: **`hvoptcore`** combines OFI-skewed quoting on velvet with passive hydrogel; **`hvoptvol`** (many variants) adds **ladder-wide** position limits, **momentum** on hydrogel, and **vol- / smile-aware** voucher handling (including “robust” / profit-lock style notes in comments); **`hvoptinv`** stresses **inventory** across many `VEV_*` lines; **`hvoptpair`**, **`hvoptema`**, **`hvoptflow`** specialize pair, EMA, and flow layers.
- **Cross-asset (`asset*`)** — Quote **all visible products**: `assetcore`-style breadth, plus **`assetema`**, **`assetinv`**, **`assetflow`**, and **`assetopt`** (Optuna over a deep `hvoptinv` / `hvoptvol` base).

**Round 4 — Richer HV / VEV + cross-asset** — Same instrument set as round 3, with more **mark-model** and **breadth** experiments: **`assetcore`** uses mid ± edge with **inventory penalty** and dislocation **takes**; **`hvoptmark`** / **`assetmark`** layer **mark-flow** ideas on hydrogel + velvet + vouchers; **`hvoptvol`** continues vol-aware MM; **`vevcore`** focuses **ATM vs wing** voucher routing; **`velvetmark`** tightens velvet-side overlays.

**Round 5 — Wide SKU universe** — Many named products (e.g. galaxy sounds, microchips, panels, robots, sleep pods). **`round5vol`** implements a long **realized-vol / microprice / markout-delay** stack with **hard caps**, per-order clips, and explicit **AVOID** sets for toxic names. **`round5ema`** adds an **EMA / trend** style layer on the same universe. **`round5pair`** runs **ratio-history** (cointegration / residual) trades between configured pairs. **`emtomcore`** carries a **broad LIMITS** table (legacy names + round-5 + vouchers) for scaffolding. **`assetvol`** ties **cross-asset** quoting to vol-style sizing.

## Portal submit helpers

Both scripts drive an **already-open** Chromium-based browser via the **Chrome DevTools Protocol** (Playwright `connect_over_cdp`). They print setup hints if the connection fails.

**Arc** — start Arc from a shell so it listens for CDP (flags and binary path depend on OS and install location). Typical pattern:

```bash
<arc> --remote-debugging-port=9222
python tools/submit.py traders/latest_trader.py
python tools/submit.py traders/latest_trader.py --inspect
```

**Google Chrome / Chromium** — quit any normal Chrome window first, then start a **debug** instance the same way (command varies by OS and package name, e.g. `chrome`, `google-chrome`, `chromium`):

```bash
<chrome-or-chromium> --remote-debugging-port=9222
python tools/submit_chrome.py traders/latest_trader.py
python tools/submit_chrome.py traders/latest_trader.py --inspect
```

Replace `<arc>` / `<chrome-or-chromium>` with whatever runs that browser on the machine (full path if the executable is not on `PATH`).

**CDP URL** — default is `http://localhost:9222`. Override if needed:

```bash
export SUBMIT_CDP_URL=http://127.0.0.1:9223
python tools/submit_chrome.py traders/latest_trader.py
```

Inspect screenshots are written under `tools/inspect_captures/` (gitignored).

## Credits

Upstream backtester: **prosperity_rust_backtester** (GeyzsoN on GitHub). Everything else here is part of a personal Prosperity 4 workspace.

## License

The **rust_backtester** crate is under the upstream **Apache-2.0** and **MIT** terms; see `backtester/LICENSE-APACHE` and `backtester/LICENSE-MIT`. Traders, research notes, and other original material in this repository are not covered by those licenses unless stated otherwise in those files.
