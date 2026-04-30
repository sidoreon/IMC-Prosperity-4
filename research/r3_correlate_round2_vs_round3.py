import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_R2 = REPO_ROOT / "datasets" / "round2"
DATASETS_R3 = REPO_ROOT / "datasets" / "round3"
RESEARCH_DIR = Path(__file__).resolve().parent
OUT_FILE = RESEARCH_DIR / "r3_correlate_round2_vs_round3_out.txt"


def _require_inputs() -> None:
    need = (
        [DATASETS_R2 / f"prices_round_2_day_{d}.csv" for d in (-1, 0, 1)]
        + [DATASETS_R3 / f"prices_round_3_day_{d}.csv" for d in (0, 1, 2)]
    )
    missing = [(p.parent.name, p.name) for p in need if not p.is_file()]
    if missing:
        print(
            "Missing price CSVs:",
            ", ".join(f"{d}/{f}" for d, f in missing),
            "\nExpected under",
            REPO_ROOT / "datasets",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    _require_inputs()

    df_r2_d_minus1 = pd.read_csv(
        DATASETS_R2 / "prices_round_2_day_-1.csv", sep=";"
    )
    df_r2_d_0 = pd.read_csv(DATASETS_R2 / "prices_round_2_day_0.csv", sep=";")
    df_r2_d_1 = pd.read_csv(DATASETS_R2 / "prices_round_2_day_1.csv", sep=";")

    df_r2_d_minus1["common_day"] = 0
    df_r2_d_0["common_day"] = 1
    df_r2_d_1["common_day"] = 2
    df_r2 = pd.concat([df_r2_d_minus1, df_r2_d_0, df_r2_d_1], ignore_index=True)

    df_r3_d_0 = pd.read_csv(DATASETS_R3 / "prices_round_3_day_0.csv", sep=";")
    df_r3_d_1 = pd.read_csv(DATASETS_R3 / "prices_round_3_day_1.csv", sep=";")
    df_r3_d_2 = pd.read_csv(DATASETS_R3 / "prices_round_3_day_2.csv", sep=";")

    df_r3_d_0["common_day"] = 0
    df_r3_d_1["common_day"] = 1
    df_r3_d_2["common_day"] = 2
    df_r3 = pd.concat([df_r3_d_0, df_r3_d_1, df_r3_d_2], ignore_index=True)

    piv_r2 = df_r2.pivot_table(
        index=["common_day", "timestamp"], columns="product", values="mid_price"
    ).reset_index()
    piv_r3 = df_r3.pivot_table(
        index=["common_day", "timestamp"], columns="product", values="mid_price"
    ).reset_index()

    merged = pd.merge(piv_r2, piv_r3, on=["common_day", "timestamp"], how="inner")
    print(f"Merged aligned dataset with {len(merged)} timestamps.")

    cols = [
        "INTARIAN_PEPPER_ROOT",
        "ASH_COATED_OSMIUM",
        "HYDROGEL_PACK",
        "VELVETFRUIT_EXTRACT",
    ]

    for col in cols:
        merged[col + "_ret"] = np.log(merged[col]).diff()

    corr_price = merged[cols].corr()
    corr_ret = merged[[c + "_ret" for c in cols]].corr()

    print("\nReturn Correlations:")
    print(corr_ret)

    def get_lead_lag(series1, series2, lags=10):
        cors = {}
        for l in range(-lags, lags + 1):
            if l < 0:
                cors[f"Lag {l} (Col1 leads)"] = series1.corr(series2.shift(l))
            elif l > 0:
                cors[f"Lag {l} (Col2 leads)"] = series1.corr(series2.shift(l))
        return cors

    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write("CROSS-ROUND COMMODITY CORRELATIONS\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. Price Level Correlation Matrix:\n")
        f.write(corr_price.to_string())
        f.write("\n\n")

        f.write("2. Log Return Correlation Matrix:\n")
        f.write(corr_ret.to_string())
        f.write("\n\n")

        f.write("3. Cross-Commodity Lead-Lag Indicators (Log Returns):\n")
        pairs = [
            ("HYDROGEL_PACK", "INTARIAN_PEPPER_ROOT"),
            ("HYDROGEL_PACK", "ASH_COATED_OSMIUM"),
            ("VELVETFRUIT_EXTRACT", "INTARIAN_PEPPER_ROOT"),
            ("VELVETFRUIT_EXTRACT", "ASH_COATED_OSMIUM"),
        ]
        for c1, c2 in pairs:
            ll = get_lead_lag(merged[c1 + "_ret"], merged[c2 + "_ret"], lags=5)
            f.write(f"\n--- {c1} vs {c2} ---\n")
            max_cor, max_lag = 0.0, ""
            for k, v in ll.items():
                f.write(f"  {k}: {v:.4f}\n")
                if abs(v) > abs(max_cor):
                    max_cor = v
                    max_lag = k
            f.write(f"  Max Absolute Cross-Corr: {max_cor:.4f} at {max_lag}\n")

    print(f"\nDone. Results saved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
