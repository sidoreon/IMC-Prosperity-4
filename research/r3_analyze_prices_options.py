import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_R3 = REPO_ROOT / "datasets" / "round3"
RESEARCH_DIR = Path(__file__).resolve().parent
OUT_FILE = RESEARCH_DIR / "r3_analyze_prices_options_out.txt"
TRADING_DAYS_PER_YEAR = 250


def _require_files(paths, base_dir: Path):
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        print(
            "Missing under",
            base_dir,
            ":",
            ", ".join(missing),
            "\nAdd official price CSVs to datasets/round3/ (repo root).",
            file=sys.stderr,
        )
        sys.exit(1)


def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)


def bs_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or pd.isna(sigma):
        return np.nan, np.nan, np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = si.norm.cdf(d1)
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
        -r * T
    ) * si.norm.cdf(d2)
    vega = S * si.norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, theta, vega


def implied_volatility(S, K, T, r, price):
    if T <= 0 or pd.isna(S) or pd.isna(price):
        return np.nan
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if price < intrinsic:
        return np.nan

    def obj_func(sigma):
        return bs_call_price(S, K, T, r, sigma) - price

    try:
        if obj_func(1e-6) * obj_func(5.0) > 0:
            return np.nan
        iv = brentq(obj_func, 1e-6, 5.0)
        return iv
    except Exception:
        return np.nan


def load_data():
    paths = [DATASETS_R3 / f"prices_round_3_day_{d}.csv" for d in (0, 1, 2)]
    _require_files(paths, DATASETS_R3)
    dfs = []
    for file_path in paths:
        dfs.append(pd.read_csv(file_path, sep=";"))
    prices = pd.concat(dfs, ignore_index=True)
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    return prices


def analyze_assets(prices, fd):
    assets = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]

    for asset in assets:
        fd.write("======================================================================\n")
        fd.write(f"  DESCRIPTIVE STATISTICS: {asset}\n")
        fd.write("======================================================================\n\n")

        df = prices[prices["product"] == asset].copy()
        df.sort_values(by=["day", "timestamp"], inplace=True)
        df["log_return"] = np.log(df["mid_price"]).diff()

        fd.write("--- Mid-Price Statistics ---\n")
        fd.write(f"  Count:      {len(df)}\n")
        fd.write(f"  Mean:       {df['mid_price'].mean():.4f}\n")
        fd.write(f"  Std Dev:    {df['mid_price'].std():.4f}\n")
        fd.write(f"  Min:        {df['mid_price'].min():.4f}\n")
        fd.write(f"  Max:        {df['mid_price'].max():.4f}\n\n")

        fd.write("--- Spread Statistics ---\n")
        fd.write(f"  Mean spread:{df['spread'].mean():.4f}\n")
        fd.write(f"  Max spread: {df['spread'].max():.4f}\n\n")

        fd.write("--- Log Return Statistics ---\n")
        fd.write(f"  Mean return:{df['log_return'].mean():.8f}\n")
        fd.write(f"  Vol (std):  {df['log_return'].std():.8f}\n")
        fd.write(
            f"  Annual Vol: {df['log_return'].std() * np.sqrt(10000 * TRADING_DAYS_PER_YEAR):.6f}\n\n"
        )


def analyze_options(prices, fd):
    vev_products = [p for p in prices["product"].unique() if p.startswith("VEV_")]
    vev_products = sorted(vev_products, key=lambda x: int(x.split("_")[1]))

    vf_prices = prices[prices["product"] == "VELVETFRUIT_EXTRACT"][
        ["day", "timestamp", "mid_price"]
    ]
    vf_prices.rename(columns={"mid_price": "underlying_price"}, inplace=True)

    options_df = prices[prices["product"].isin(vev_products)].copy()
    options_df = options_df.merge(vf_prices, on=["day", "timestamp"], how="left")

    options_df["TTE_days"] = 8 - options_df["day"]
    options_df["T"] = options_df["TTE_days"] / TRADING_DAYS_PER_YEAR
    options_df["K"] = options_df["product"].apply(lambda x: float(x.split("_")[1]))
    options_df["r"] = 0.0

    fd.write("======================================================================\n")
    fd.write("  OPTIONS ANALYSIS (Velvetfruit Extract Vouchers)\n")
    fd.write("======================================================================\n\n")

    sampled = options_df.iloc[::100].copy()

    sampled["IV"] = sampled.apply(
        lambda row: implied_volatility(
            row["underlying_price"], row["K"], row["T"], row["r"], row["mid_price"]
        ),
        axis=1,
    )

    greeks = sampled.apply(
        lambda row: bs_greeks(
            row["underlying_price"], row["K"], row["T"], row["r"], row["IV"]
        ),
        axis=1,
    )

    sampled["Delta"], sampled["Gamma"], sampled["Theta"], sampled["Vega"] = zip(*greeks)

    mean_metrics = (
        sampled.groupby("product")[
            ["IV", "Delta", "Gamma", "Theta", "Vega", "mid_price", "spread"]
        ]
        .mean()
        .reset_index()
    )

    for _, row in mean_metrics.iterrows():
        fd.write(f"--- {row['product']} ---\n")
        fd.write(f"  Mean Price:    {row['mid_price']:.4f}\n")
        fd.write(f"  Mean Spread:   {row['spread']:.4f}\n")
        fd.write(f"  Mean Impl Vol: {row['IV']:.4f}  (Approx {(row['IV']*100):.2f}%)\n")
        fd.write(f"  Mean Delta:    {row['Delta']:.4f}\n")
        fd.write(f"  Mean Gamma:    {row['Gamma']:.4f}\n")
        fd.write(f"  Mean Theta:    {row['Theta']:.4f}\n")
        fd.write(f"  Mean Vega:     {row['Vega']:.4f}\n\n")

    fd.write("\n--- Head of Sampled Greeks DataFrame ---\n")
    fd.write(
        sampled[
            ["day", "timestamp", "product", "underlying_price", "mid_price", "IV", "Delta"]
        ]
        .head(10)
        .to_string()
    )
    fd.write("\n\n")


if __name__ == "__main__":
    prices = load_data()
    with OUT_FILE.open("w", encoding="utf-8") as fd:
        fd.write("FINAL DATA ANALYSIS - ROUND 3\n")
        fd.write("=" * 80 + "\n\n")
        analyze_assets(prices, fd)
        analyze_options(prices, fd)
    print(f"Analysis complete. Results written to {OUT_FILE}")
