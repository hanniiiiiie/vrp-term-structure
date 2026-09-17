import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/processed/vrp_term_structure.csv")
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"])
x = df.dropna(subset=["vrp_1m", "vrp_3m", "vrp_6m"]).copy()


# ==========================================================
# FIGURE 1 — Average VRP term structure
# ==========================================================

means = x[["vrp_1m", "vrp_3m", "vrp_6m"]].mean()

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    ["1M", "3M", "6M"],
    means.values,
    marker="o",
    linewidth=2
)

ax.axhline(0, linewidth=1)
ax.set_title("Average Variance Risk Premium by Maturity")
ax.set_xlabel("Maturity")
ax.set_ylabel("Annualized Variance: IV − Subsequent RV")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "01_average_vrp_term_structure.png",
    dpi=300
)
plt.close(fig)


# ==========================================================
# FIGURE 2 — Term structure by VIX regime
# ==========================================================

x["vix_regime"] = pd.qcut(
    x["vix"],
    5,
    labels=[
        "Q1 Lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 Highest"
    ]
)

regime = (
    x.groupby("vix_regime", observed=True)
     [["vrp_1m", "vrp_3m", "vrp_6m"]]
     .mean()
)

fig, ax = plt.subplots(figsize=(8, 5))

for regime_name, row in regime.iterrows():
    ax.plot(
        ["1M", "3M", "6M"],
        row.values,
        marker="o",
        label=regime_name
    )

ax.axhline(0, linewidth=1)
ax.set_title("VRP Term Structure Across Volatility Regimes")
ax.set_xlabel("Maturity")
ax.set_ylabel("Annualized Variance: IV − Subsequent RV")
ax.legend(title="VIX Quintile")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "02_vrp_by_vix_regime.png",
    dpi=300
)
plt.close(fig)


# ==========================================================
# FIGURE 3 — 6M minus 1M slope through time
# Month-end observations reduce visual overlap.
# ==========================================================

x["slope_6m_1m"] = x["vrp_6m"] - x["vrp_1m"]

monthly = (
    x.set_index("date")
     .groupby(pd.Grouper(freq="ME"))
     .tail(1)
     .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    monthly["date"],
    monthly["slope_6m_1m"],
    linewidth=1.5
)

ax.axhline(0, linewidth=1)

ax.set_title("VRP Term-Structure Slope Through Time")
ax.set_xlabel("Date")
ax.set_ylabel("6M VRP − 1M VRP")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "03_vrp_slope_over_time.png",
    dpi=300
)
plt.close(fig)


print("Saved figures:")
for f in sorted(OUT.glob("*.png")):
    print(f)
