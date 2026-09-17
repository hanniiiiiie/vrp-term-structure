import pandas as pd
from functools import reduce

series = ["SPX", "VIX", "VIX3M", "VIX6M"]

dfs = {}

for t in series:
    df = pd.read_csv(
        f"data/raw/{t}_daily_close.csv",
        parse_dates=["date", "close_timestamp"]
    )

    dfs[t] = df[[
        "date", "close", "close_timestamp", "n_bars"
    ]].rename(columns={
        "close": f"{t}_close",
        "close_timestamp": f"{t}_timestamp",
        "n_bars": f"{t}_bars"
    })

common = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="inner"),
    dfs.values()
)

print("Common dates:", len(common))
print("Start:", common.date.min())
print("End:", common.date.max())

# Difference between latest and earliest selected timestamp on each common date
times = [f"{t}_timestamp" for t in series]

common["time_spread_min"] = (
    common[times].max(axis=1) - common[times].min(axis=1)
).dt.total_seconds() / 60

print("\n=== Timestamp spread ===")
print(common["time_spread_min"].describe())

print("\n=== Largest timestamp mismatches ===")
cols = ["date", "time_spread_min"] + times
print(
    common.sort_values("time_spread_min", ascending=False)
          [cols]
          .head(30)
          .to_string(index=False)
)
