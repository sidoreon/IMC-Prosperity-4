import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_R3 = REPO_ROOT / "datasets" / "round3"
RESEARCH_DIR = Path(__file__).resolve().parent
OUT_FILE = RESEARCH_DIR / "r3_correlate_hyd_velt_out.txt"


def _require_round3_prices():
    paths = [DATASETS_R3 / f"prices_round_3_day_{d}.csv" for d in (0, 1, 2)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        print(
            "Missing under",
            DATASETS_R3,
            ":",
            ", ".join(missing),
            "\nAdd official price CSVs to datasets/round3/.",
            file=sys.stderr,
        )
        sys.exit(1)
    return paths


def main():
    paths = _require_round3_prices()
    dfs = [pd.read_csv(p, sep=";") for p in paths]

    for i, df in enumerate(dfs):
        df["day_index"] = i

    df_r3 = pd.concat(dfs, ignore_index=True)

    df_delta1 = df_r3[df_r3["product"].isin(["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"])]

    piv_r3 = df_delta1.pivot_table(
        index=["day_index", "timestamp"], columns="product", values="mid_price"
    ).reset_index()

    piv_r3["HYDROGEL_PACK_ret"] = np.log(piv_r3["HYDROGEL_PACK"]).diff()
    piv_r3["VELVETFRUIT_EXTRACT_ret"] = np.log(piv_r3["VELVETFRUIT_EXTRACT"]).diff()

    corr_price = piv_r3[["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]].corr()
    corr_ret = piv_r3[["HYDROGEL_PACK_ret", "VELVETFRUIT_EXTRACT_ret"]].corr()

    print("\nReturn Correlations:")
    print(corr_ret)

    def get_lead_lag(series1, series2, lags=10):
        cors = {}
        for l in range(-lags, lags + 1):
            if l < 0:
                cors[f"Lag {l} (Col1 leads)"] = series1.corr(series2.shift(l))
            elif l > 0:
                cors[f"Lag {l} (Col2 leads)"] = series1.corr(series2.shift(l))
            else:
                cors["Contemporaneous"] = series1.corr(series2)
        return cors

    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write("INTERNAL ROUND 3 COMMODITY CORRELATIONS\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. Price Level Correlation Matrix:\n")
        f.write(corr_price.to_string())
        f.write("\n\n")

        f.write("2. Log Return Correlation Matrix:\n")
        f.write(corr_ret.to_string())
        f.write("\n\n")

        f.write("3. Cross-Commodity Lead-Lag Indicators (Log Returns):\n")
        ll = get_lead_lag(
            piv_r3["HYDROGEL_PACK_ret"], piv_r3["VELVETFRUIT_EXTRACT_ret"], lags=10
        )
        f.write("\n--- HYDROGEL_PACK vs VELVETFRUIT_EXTRACT ---\n")
        max_cor, max_lag = 0.0, ""
        for k, v in ll.items():
            f.write(f"  {k}: {v:.4f}\n")
            if abs(v) > abs(max_cor):
                max_cor = v
                max_lag = k
        f.write(f"\n  Max Absolute Cross-Corr: {max_cor:.4f} at {max_lag}\n")

    print(f"\nDone. Results saved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
