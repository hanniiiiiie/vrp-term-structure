import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/vrp_term_structure.csv")
OUTPUT = Path("data/processed/vrp_monthly_snapshot.csv")

df = pd.read_csv(INPUT, parse_dates=["date"])

# Keep observations for which all 3 VRPs are available
x = df.dropna(subset=["vrp_1m", "vrp_3m", "vrp_6m"]).copy()

# ============================================================
# 1. FULL-SAMPLE TERM STRUCTURE
# ============================================================

print("=== FULL-SAMPLE TERM STRUCTURE ===")

summary = pd.DataFrame({
    "mean": x[["vrp_1m", "vrp_3m", "vrp_6m"]].mean(),
    "median": x[["vrp_1m", "vrp_3m", "vrp_6m"]].median(),
    "std": x[["vrp_1m", "vrp_3m", "vrp_6m"]].std(),
    "positive_freq": (x[["vrp_1m", "vrp_3m", "vrp_6m"]] > 0).mean()
})

print(summary)

# ============================================================
# 2. TERM-STRUCTURE SLOPES
# ============================================================

x["slope_3m_1m"] = x["vrp_3m"] - x["vrp_1m"]
x["slope_6m_3m"] = x["vrp_6m"] - x["vrp_3m"]
x["slope_6m_1m"] = x["vrp_6m"] - x["vrp_1m"]

print("\n=== SLOPE SUMMARY ===")
print(
    x[
        ["slope_3m_1m", "slope_6m_3m", "slope_6m_1m"]
    ].describe()
)

print("\nPositive slope frequencies:")
for c in ["slope_3m_1m", "slope_6m_3m", "slope_6m_1m"]:
    print(c, f"{(x[c] > 0).mean():.2%}")

# ============================================================
# 3. YEAR-BY-YEAR TERM STRUCTURE
# ============================================================

x["year"] = x["date"].dt.year

annual = (
    x.groupby("year")
     [["vrp_1m", "vrp_3m", "vrp_6m",
       "slope_3m_1m", "slope_6m_3m", "slope_6m_1m"]]
     .mean()
)

print("\n=== YEARLY MEANS ===")
print(annual.to_string())

# ============================================================
# 4. VOLATILITY REGIMES
#
# Define regimes using VIX quintiles rather than arbitrary
# thresholds.
# ============================================================

x["vix_regime"] = pd.qcut(
    x["vix"],
    q=5,
    labels=[
        "Q1 lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 highest"
    ]
)

regime = (
    x.groupby("vix_regime", observed=True)
     [["vix", "vrp_1m", "vrp_3m", "vrp_6m",
       "slope_3m_1m", "slope_6m_3m"]]
     .mean()
)

print("\n=== TERM STRUCTURE BY VIX REGIME ===")
print(regime.to_string())

# ============================================================
# 5. MONTH-END SNAPSHOT
#
# Daily observations use highly overlapping future windows.
# A month-end sample is useful as a descriptive robustness
# check and prevents thousands of almost-identical windows
# from dominating the story.
# ============================================================

monthly = (
    x.set_index("date")
     .groupby(pd.Grouper(freq="ME"))
     .tail(1)
     .reset_index()
)

monthly.to_csv(OUTPUT, index=False)

print("\n=== MONTH-END SAMPLE ===")
print("Observations:", len(monthly))
print("Start:", monthly["date"].min().date())
print("End:", monthly["date"].max().date())

print("\nMonthly VRP means:")
print(
    monthly[
        ["vrp_1m", "vrp_3m", "vrp_6m"]
    ].mean()
)

print("\nMonthly positive slopes:")
print(
    "3M - 1M:",
    f"{(monthly['slope_3m_1m'] > 0).mean():.2%}"
)
print(
    "6M - 3M:",
    f"{(monthly['slope_6m_3m'] > 0).mean():.2%}"
)
print(
    "6M - 1M:",
    f"{(monthly['slope_6m_1m'] > 0).mean():.2%}"
)

print("\nSaved:", OUTPUT)
