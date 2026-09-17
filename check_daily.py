import pandas as pd

d = {
    t: pd.read_csv(
        f"data/raw/{t}_daily_close.csv",
        parse_dates=["date", "close_timestamp"]
    )
    for t in ["SPX", "VIX", "VIX3M", "VIX6M"]
}

print("=== 1. Days present in VIX but missing from SPX ===")
extra = set(d["VIX"].date) - set(d["SPX"].date)

for x in sorted(extra):
    print(x)

print("\nNumber of extra VIX days:", len(extra))

print("\n=== 2. Most frequent closing times ===")
for t, df in d.items():
    print(
        t,
        df.close_timestamp.dt.time.value_counts().head(5).to_dict()
    )
