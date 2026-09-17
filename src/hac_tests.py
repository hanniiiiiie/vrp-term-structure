import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

INPUT = Path("data/processed/vrp_term_structure.csv")
OUTPUT = Path("data/processed/hac_tests.csv")

df = pd.read_csv(INPUT, parse_dates=["date"])

# ---------------------------------------------------------
# Construct slopes
# ---------------------------------------------------------

df["slope_3m_1m"] = df["vrp_3m"] - df["vrp_1m"]
df["slope_6m_3m"] = df["vrp_6m"] - df["vrp_3m"]
df["slope_6m_1m"] = df["vrp_6m"] - df["vrp_1m"]


def hac_mean_test(series, maxlags, name):
    """
    Test H0: mean(series) = 0

    Equivalent to an intercept-only regression:
        y_t = alpha + error_t

    HAC/Newey-West standard errors account for
    heteroskedasticity and serial correlation.
    """

    y = series.dropna().astype(float)

    X = np.ones((len(y), 1))

    model = sm.OLS(y.values, X)

    result = model.fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
            "kernel": "bartlett",
            "use_correction": True,
        },
        use_t=True,
    )

    mean = result.params[0]
    se = result.bse[0]
    t_stat = result.tvalues[0]
    p_value = result.pvalues[0]

    ci = result.conf_int(alpha=0.05)[0]

    return {
        "variable": name,
        "n": len(y),
        "mean": mean,
        "hac_lags": maxlags,
        "hac_se": se,
        "t_stat": t_stat,
        "p_value_two_sided": p_value,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
    }


# ---------------------------------------------------------
# Lag choices
#
# Use approximately the number of overlapping trading-day
# observations implied by each horizon.
#
# 1M ~ 21 trading days
# 3M ~ 65
# 6M ~ 127
#
# For a maturity difference, use the longer horizon.
# ---------------------------------------------------------

tests = [
    ("vrp_1m", 21),
    ("vrp_3m", 65),
    ("vrp_6m", 127),

    ("slope_3m_1m", 65),
    ("slope_6m_3m", 127),
    ("slope_6m_1m", 127),
]

results = []

for variable, lag in tests:
    results.append(
        hac_mean_test(
            df[variable],
            maxlags=lag,
            name=variable
        )
    )

results = pd.DataFrame(results)

# ---------------------------------------------------------
# Pretty output
# ---------------------------------------------------------

print("=== HAC / NEWEY-WEST MEAN TESTS ===")
print("H0: mean = 0")
print()

display = results.copy()

for col in [
    "mean",
    "hac_se",
    "t_stat",
    "p_value_two_sided",
    "ci95_low",
    "ci95_high",
]:
    display[col] = display[col].map(lambda x: f"{x:.6f}")

print(display.to_string(index=False))

# ---------------------------------------------------------
# Economic interpretation helper
# ---------------------------------------------------------

print("\n=== INTERPRETATION ===")

for _, row in results.iterrows():

    name = row["variable"]
    mean = row["mean"]
    p = row["p_value_two_sided"]

    if p < 0.01:
        sig = "significant at 1%"
    elif p < 0.05:
        sig = "significant at 5%"
    elif p < 0.10:
        sig = "significant at 10%"
    else:
        sig = "not significant at 10%"

    direction = "positive" if mean > 0 else "negative"

    print(
        f"{name:15s}: mean = {mean:.6f}, "
        f"{direction}, {sig}"
    )

results.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
