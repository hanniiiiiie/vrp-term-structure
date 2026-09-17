import pandas as pd

df = pd.read_csv(
    "data/processed/vrp_term_structure.csv",
    parse_dates=["date"]
)

cols = [
    "date",
    "spx",
    "vix",
    "vix3m",
    "vix6m",
    "iv_1m", "rv_1m", "vrp_1m",
    "iv_3m", "rv_3m", "vrp_3m",
    "iv_6m", "rv_6m", "vrp_6m"
]

print("=== 10 LOWEST 1M VRP ===")
print(
    df.nsmallest(10, "vrp_1m")[cols]
      .to_string(index=False)
)

print("\n=== 10 HIGHEST 1M VRP ===")
print(
    df.nlargest(10, "vrp_1m")[cols]
      .to_string(index=False)
)

print("\n=== VRP POSITIVE FREQUENCY ===")
for h in ["1m", "3m", "6m"]:
    x = df[f"vrp_{h}"].dropna()
    print(
        h,
        f"{(x > 0).mean():.2%}",
        f"({(x > 0).sum()} / {len(x)})"
    )

print("\n=== AVERAGE VRP BY YEAR ===")
annual = (
    df.assign(year=df["date"].dt.year)
      .groupby("year")[["vrp_1m", "vrp_3m", "vrp_6m"]]
      .mean()
)

print(annual.to_string())

print("\n=== TERM-STRUCTURE ORDERING ===")

valid = df[
    ["vrp_1m", "vrp_3m", "vrp_6m"]
].dropna()

upward = (
    (valid["vrp_1m"] < valid["vrp_3m"]) &
    (valid["vrp_3m"] < valid["vrp_6m"])
)

downward = (
    (valid["vrp_1m"] > valid["vrp_3m"]) &
    (valid["vrp_3m"] > valid["vrp_6m"])
)

print("Strictly upward:", f"{upward.mean():.2%}")
print("Strictly downward:", f"{downward.mean():.2%}")
print("Other shapes:", f"{1 - upward.mean() - downward.mean():.2%}")
