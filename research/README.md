# research

Scripts and notes live here; **inputs are read from the repo `datasets/` tree** (no duplicate CSVs in this folder).

Layout:

- **`REPO_ROOT`** = parent of `research/` (repository root).
- **`datasets/round2/prices_round_2_day_*.csv`** — used by `r3_correlate_round2_vs_round3.py`.
- **`datasets/round3/prices_round_3_day_*.csv`** — used by all round-3 analysis scripts.

Run scripts from anywhere; they resolve paths via `Path(__file__)`. **Outputs** (`.txt`, `.png`) are written next to the scripts in `research/`.

## Scripts → outputs

| Script | Reads | Writes |
|--------|--------|--------|
| `r3_analyze_prices_options.py` | `datasets/round3/prices_round_3_day_{0,1,2}.csv` | `r3_analyze_prices_options_out.txt` |
| `r3_correlate_hyd_velt.py` | same | `r3_correlate_hyd_velt_out.txt` |
| `r3_correlate_round2_vs_round3.py` | `datasets/round2/` + `datasets/round3/` price CSVs | `r3_correlate_round2_vs_round3_out.txt` |
| `r3_plot_implied_volatility.py` | `datasets/round3/` price CSVs | `r3_plot_iv_vs_time.png`, `r3_plot_iv_vs_moneyness.png` |

## Other files

- **`r4_report_asset_strategy.md`** — Round 4 literature / strategy notes.
- **`r0_notebook_prosper0.ipynb`** — Tutorial exploration.
- **`r3_notes_*.txt`** — Archived text exports.

Pipelines need **price** tapes (`mid_price`, etc.), not `trades_*.csv`.
