import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

INPUT = Path("data/processed/vrp_term_structure.csv")
OUTPUT = Path("data/processed/regime_hac_tests.csv")

df = pd.read_csv(INPUT, parse_dates=["date"])

# ---------------------------------------------------------
# 1. Main term-structure slope
# ---------------------------------------------------------

df["slope_6m_1m"] = df["vrp_6m"] - df["vrp_1m"]

x = df.dropna(
    subset=["slope_6m_1m", "vix"]
).copy()

# VIX quintiles
x["vix_regime"] = pd.qcut(
    x["vix"],
    5,
    labels=[
        "Q1_lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5_highest"
    ]
)

# ---------------------------------------------------------
# 2. Dummy regression with NO intercept
#
# Each coefficient = average 6M-1M slope in that regime.
# Keeping the entire chronological sample allows HAC
# covariance to account for serial dependence properly.
# ---------------------------------------------------------

D = pd.get_dummies(
    x["vix_regime"],
    dtype=float
)

y = x["slope_6m_1m"].astype(float)

model = sm.OLS(y, D)

result = model.fit(
    cov_type="HAC",
    cov_kwds={
        "maxlags": 127,
        "kernel": "bartlett",
        "use_correction": True,
    },
    use_t=True,
)

print("=== CONDITIONAL 6M-1M VRP SLOPE ===")
print("HAC / Newey-West, 127 lags\n")

rows = []

for regime in D.columns:

    coef = result.params[regime]
    se = result.bse[regime]
    t = result.tvalues[regime]
    p = result.pvalues[regime]

    idx = list(D.columns).index(regime)
    ci = result.conf_int(alpha=0.05).iloc[idx]

    n = int((x["vix_regime"] == regime).sum())

    rows.append({
        "regime": regime,
        "n": n,
        "mean_slope_6m_1m": coef,
        "hac_se": se,
        "t_stat": t,
        "p_value": p,
        "ci95_low": ci.iloc[0],
        "ci95_high": ci.iloc[1],
        "mean_vix": x.loc[
            x["vix_regime"] == regime, "vix"
        ].mean()
    })

table = pd.DataFrame(rows)

print(
    table.to_string(
        index=False,
        formatters={
            "mean_slope_6m_1m": "{:.6f}".format,
            "hac_se": "{:.6f}".format,
            "t_stat": "{:.3f}".format,
            "p_value": "{:.4f}".format,
            "ci95_low": "{:.6f}".format,
            "ci95_high": "{:.6f}".format,
            "mean_vix": "{:.2f}".format,
        }
    )
)

# ---------------------------------------------------------
# 3. Formal high-VIX minus low-VIX test
#
# H0:
# mean slope in Q5 = mean slope in Q1
# ---------------------------------------------------------

names = list(D.columns)

contrast = np.zeros(len(names))
contrast[names.index("Q5_highest")] = 1
contrast[names.index("Q1_lowest")] = -1

test = result.t_test(contrast)

difference = np.asarray(test.effect).item()
se = np.asarray(test.sd).item()
tstat = np.asarray(test.tvalue).item()
pvalue = np.asarray(test.pvalue).item()

print("\n=== HIGH-VIX vs LOW-VIX CONTRAST ===")
print(
    "Q5 slope - Q1 slope:",
    f"{difference:.6f}"
)
print(
    "HAC SE:",
    f"{se:.6f}"
)
print(
    "t-stat:",
    f"{tstat:.3f}"
)
print(
    "p-value:",
    f"{pvalue:.4f}"
)

# ---------------------------------------------------------
# 4. Continuous specification
#
# slope_t = alpha + beta * VIX_t + error_t
#
# This is descriptive, not causal.
# ---------------------------------------------------------

X = sm.add_constant(x["vix"].astype(float))

continuous = sm.OLS(
    y,
    X
).fit(
    cov_type="HAC",
    cov_kwds={
        "maxlags": 127,
        "kernel": "bartlett",
        "use_correction": True,
    },
    use_t=True,
)

print("\n=== CONTINUOUS VIX SPECIFICATION ===")
print(
    f"Intercept: {continuous.params['const']:.6f}"
)
print(
    f"VIX coefficient: {continuous.params['vix']:.6f}"
)
print(
    f"HAC SE: {continuous.bse['vix']:.6f}"
)
print(
    f"t-stat: {continuous.tvalues['vix']:.3f}"
)
print(
    f"p-value: {continuous.pvalues['vix']:.4f}"
)

table.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
