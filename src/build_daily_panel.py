import pandas as pd
from functools import reduce
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

series = ["SPX", "VIX", "VIX3M", "VIX6M"]
frames = []

for t in series:
    df = pd.read_csv(
        RAW / f"{t}_daily_close.csv",
        parse_dates=["date", "close_timestamp"]
    )

    df = df[
        ["date", "close", "close_timestamp", "n_bars"]
    ].copy()

    df = df.rename(columns={
        "close": t.lower(),
        "close_timestamp": f"{t.lower()}_timestamp",
        "n_bars": f"{t.lower()}_n_bars"
    })

    frames.append(df)

# Inner join = dates available for ALL four series
panel = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="inner"),
    frames
)

panel = panel.sort_values("date").reset_index(drop=True)

# Basic quality checks
assert not panel["date"].duplicated().any(), "Duplicate dates found"
assert panel[series[0].lower()].notna().all()

price_cols = ["spx", "vix", "vix3m", "vix6m"]

print("=== COMMON DAILY PANEL ===")
print("Observations:", len(panel))
print("Start:", panel["date"].min().date())
print("End:", panel["date"].max().date())

print("\nMissing values:")
print(panel[price_cols].isna().sum())

print("\nFirst rows:")
print(panel[["date"] + price_cols].head().to_string(index=False))

print("\nLast rows:")
print(panel[["date"] + price_cols].tail().to_string(index=False))

panel.to_csv(
    OUT / "vrp_daily_panel.csv",
    index=False
)

print("\nSaved:")
print(OUT / "vrp_daily_panel.csv")
