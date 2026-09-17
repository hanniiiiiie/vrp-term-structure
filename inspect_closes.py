import pandas as pd

for t in ["SPX", "VIX", "VIX3M", "VIX6M"]:
    df = pd.read_csv(
        f"data/raw/{t}_daily_close.csv",
        parse_dates=["date", "close_timestamp"]
    )

    bad = df[df["close_timestamp"].dt.time != pd.Timestamp("16:00").time()]

    print(f"\n=== {t}: non-16:00 closes ({len(bad)}) ===")
    print(
        bad[["date", "close", "close_timestamp", "n_bars"]]
        .tail(30)
        .to_string(index=False)
    )
