import pandas as pd
from pathlib import Path

path = Path("data/external/SPX_full_1min.txt")

df = pd.read_csv(
    path,
    header=None,
    names=["datetime", "open", "high", "low", "close"]
)

df["datetime"] = pd.to_datetime(df["datetime"])

dates = ["2019-12-24", "2018-12-24", "2024-11-29"]

for date in dates:
    x = df[df["datetime"].dt.strftime("%Y-%m-%d") == date].copy()

    print(f"\n===== {date} =====")
    print("bars:", len(x))
    print("first:", x["datetime"].min())
    print("last :", x["datetime"].max())

    print("\nFirst 5:")
    print(x.head().to_string(index=False))

    print("\nLast 15:")
    print(x.tail(15).to_string(index=False))
