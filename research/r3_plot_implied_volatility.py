import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as si
import seaborn as sns
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_R3 = REPO_ROOT / "datasets" / "round3"
RESEARCH_DIR = Path(__file__).resolve().parent
TRADING_DAYS_PER_YEAR = 250
DAY_TIMESTEPS = 10000


def _require_round3_prices() -> None:
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


def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)


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
    _require_round3_prices()
    dfs = []
    for day in (0, 1, 2):
        dfs.append(pd.read_csv(DATASETS_R3 / f"prices_round_3_day_{day}.csv", sep=";"))
    return pd.concat(dfs, ignore_index=True)


def main():
    prices = load_data()
    if prices is None:
        print("No data found!")
        return

    vev_products = [p for p in prices["product"].unique() if p.startswith("VEV_")]
    vev_products = sorted(vev_products, key=lambda x: int(x.split("_")[1]))

    vf_prices = prices[prices["product"] == "VELVETFRUIT_EXTRACT"][
        ["day", "timestamp", "mid_price"]
    ]
    vf_prices.rename(columns={"mid_price": "underlying_price"}, inplace=True)

    options_df = prices[prices["product"].isin(vev_products)].copy()
    options_df = options_df.merge(vf_prices, on=["day", "timestamp"], how="left")

    options_df["TTE_days"] = (
        8 - options_df["day"] - (options_df["timestamp"] / DAY_TIMESTEPS)
    )
    options_df["T"] = options_df["TTE_days"] / TRADING_DAYS_PER_YEAR
    options_df["K"] = options_df["product"].apply(lambda x: float(x.split("_")[1]))
    options_df["r"] = 0.0

    sampled = options_df.iloc[::50].copy()

    sampled["IV"] = sampled.apply(
        lambda row: implied_volatility(
            row["underlying_price"], row["K"], row["T"], row["r"], row["mid_price"]
        ),
        axis=1,
    )

    sampled = sampled.dropna(subset=["IV"])

    sampled["Moneyness"] = sampled["underlying_price"] / sampled["K"]
    sampled["Time"] = sampled["day"] * DAY_TIMESTEPS + sampled["timestamp"]

    plt.figure(figsize=(12, 6))
    for K in sampled["K"].unique():
        subset = sampled[sampled["K"] == K]
        plt.scatter(subset["Time"], subset["IV"], label=f"Strike {K}", s=5, alpha=0.6)
    plt.title("Implied Volatility vs Time")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESEARCH_DIR / "r3_plot_iv_vs_time.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        sampled["Moneyness"],
        sampled["IV"],
        c=sampled["TTE_days"],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    plt.colorbar(label="Time to Expiry (Days)")
    plt.title("Implied Volatility vs Moneyness (Smile/Smirk)")
    plt.xlabel("Moneyness (S / K)")
    plt.ylabel("Implied Volatility")
    plt.grid(True, alpha=0.3)
    plt.savefig(RESEARCH_DIR / "r3_plot_iv_vs_moneyness.png", bbox_inches="tight")
    plt.close()

    avg_iv_by_strike = sampled.groupby("K")["IV"].mean()
    print("Average IV by Strike:")
    print(avg_iv_by_strike)

    correlation = sampled["Moneyness"].corr(sampled["IV"])
    print(f"\nCorrelation between Moneyness and IV: {correlation:.4f}")

    time_corr = sampled["Time"].corr(sampled["IV"])
    print(f"Correlation between Time and IV: {time_corr:.4f}")

    print("\nSummary Stats of IV:")
    print(sampled["IV"].describe())


if __name__ == "__main__":
    main()
