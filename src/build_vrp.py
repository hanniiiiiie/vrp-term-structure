import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/vrp_daily_panel.csv")
OUTPUT = Path("data/processed/vrp_term_structure.csv")

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# 1. SPX daily log returns
# ---------------------------------------------------------
df["spx_log_return"] = np.log(df["spx"] / df["spx"].shift(1))
df["sq_log_return"] = df["spx_log_return"] ** 2

# ---------------------------------------------------------
# 2. Implied variance
#
# Cboe volatility indices are annualized volatility in %.
# Example:
# VIX = 20  -> annualized implied variance = (20/100)^2 = 0.04
# ---------------------------------------------------------
df["iv_1m"] = (df["vix"] / 100.0) ** 2
df["iv_3m"] = (df["vix3m"] / 100.0) ** 2
df["iv_6m"] = (df["vix6m"] / 100.0) ** 2


def forward_realized_variance(data, horizon_days):
    """
    Forward realized variance from date t to approximately
    t + horizon_days CALENDAR days.

    Uses daily SPX close-to-close log returns.

    The first return included is the return AFTER date t,
    so no historical information is mistakenly used in the
    forward realized variance.

    Returns annualized realized variance using 252 trading days.
    """

    dates = data["date"].to_numpy()
    sqret = data["sq_log_return"].to_numpy()

    rv = np.full(len(data), np.nan)
    end_dates = np.full(len(data), np.datetime64("NaT"), dtype="datetime64[ns]")
    n_returns = np.full(len(data), np.nan)

    for i in range(len(data)):

        target_date = data.loc[i, "date"] + pd.Timedelta(days=horizon_days)

        # First observed trading day on or after target calendar date
        j = dates.searchsorted(np.datetime64(target_date), side="left")

        if j >= len(data):
            continue

        # Returns correspond to P_k / P_(k-1).
        # At time i, future returns are i+1 through j.
        future_sq = sqret[i + 1 : j + 1]

        future_sq = future_sq[~np.isnan(future_sq)]

        if len(future_sq) == 0:
            continue

        # Annualized realized variance
        rv[i] = 252.0 * future_sq.mean()

        end_dates[i] = dates[j]
        n_returns[i] = len(future_sq)

    return rv, end_dates, n_returns


# ---------------------------------------------------------
# 3. Forward realized variance at Cboe target maturities
# ---------------------------------------------------------

# Official Cboe constant maturities:
# VIX   = 30 calendar days
# VIX3M = 93 calendar days
# VIX6M = 184 calendar days

for label, days in {
    "1m": 30,
    "3m": 93,
    "6m": 184,
}.items():

    rv, end_date, n_returns = forward_realized_variance(df, days)

    df[f"rv_{label}"] = rv
    df[f"rv_{label}_end_date"] = end_date
    df[f"rv_{label}_n_returns"] = n_returns


# ---------------------------------------------------------
# 4. Variance Risk Premium
#
# Convention used:
#
# VRP = implied variance - subsequent realized variance
#
# Positive VRP means implied variance exceeded subsequently
# realized variance.
# ---------------------------------------------------------

df["vrp_1m"] = df["iv_1m"] - df["rv_1m"]
df["vrp_3m"] = df["iv_3m"] - df["rv_3m"]
df["vrp_6m"] = df["iv_6m"] - df["rv_6m"]


# ---------------------------------------------------------
# 5. Simple term-structure measures
# ---------------------------------------------------------

df["iv_slope_3m_1m"] = df["iv_3m"] - df["iv_1m"]
df["iv_slope_6m_3m"] = df["iv_6m"] - df["iv_3m"]

df["vrp_slope_3m_1m"] = df["vrp_3m"] - df["vrp_1m"]
df["vrp_slope_6m_3m"] = df["vrp_6m"] - df["vrp_3m"]


# ---------------------------------------------------------
# 6. Save
# ---------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print("=== VRP TERM STRUCTURE DATASET ===")
print("Observations:", len(df))
print("Start:", df["date"].min().date())
print("End:", df["date"].max().date())

print("\nNon-missing VRP observations:")
print(
    df[["vrp_1m", "vrp_3m", "vrp_6m"]]
    .notna()
    .sum()
)

print("\nAverage number of future SPX returns:")
print(
    df[
        [
            "rv_1m_n_returns",
            "rv_3m_n_returns",
            "rv_6m_n_returns"
        ]
    ].mean()
)

print("\nVRP summary:")
print(
    df[["vrp_1m", "vrp_3m", "vrp_6m"]]
    .describe()
)

print("\nExample rows:")
print(
    df[
        [
            "date",
            "spx",
            "vix",
            "vix3m",
            "vix6m",
            "iv_1m",
            "rv_1m",
            "vrp_1m",
            "iv_3m",
            "rv_3m",
            "vrp_3m",
            "iv_6m",
            "rv_6m",
            "vrp_6m",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nSaved:", OUTPUT)
