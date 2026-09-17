import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv(
    "data/processed/vrp_monthly_snapshot.csv",
    parse_dates=["date"]
)

df["slope_3m_1m"] = df["vrp_3m"] - df["vrp_1m"]
df["slope_6m_3m"] = df["vrp_6m"] - df["vrp_3m"]
df["slope_6m_1m"] = df["vrp_6m"] - df["vrp_1m"]


def hac_test(series, maxlags, name):
    y = series.dropna().astype(float)

    X = np.ones((len(y), 1))

    result = sm.OLS(y.values, X).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
            "kernel": "bartlett",
            "use_correction": True,
        },
        use_t=True,
    )

    ci = result.conf_int(alpha=0.05)[0]

    return {
        "variable": name,
        "n": len(y),
        "mean": result.params[0],
        "hac_se": result.bse[0],
        "t_stat": result.tvalues[0],
        "p_value": result.pvalues[0],
        "ci95_low": ci[0],
        "ci95_high": ci[1],
    }


# Approximate overlap in monthly observations:
# 1M ≈ 1 month
# 3M ≈ 3 months
# 6M ≈ 6 months

tests = [
    ("vrp_1m", 1),
    ("vrp_3m", 3),
    ("vrp_6m", 6),
    ("slope_3m_1m", 3),
    ("slope_6m_3m", 6),
    ("slope_6m_1m", 6),
]

results = []

for variable, lag in tests:
    results.append(
        hac_test(df[variable], lag, variable)
    )

results = pd.DataFrame(results)

print("=== MONTH-END HAC ROBUSTNESS ===")
print("H0: mean = 0\n")

print(
    results.to_string(
        index=False,
        formatters={
            "mean": "{:.6f}".format,
            "hac_se": "{:.6f}".format,
            "t_stat": "{:.3f}".format,
            "p_value": "{:.4f}".format,
            "ci95_low": "{:.6f}".format,
            "ci95_high": "{:.6f}".format,
        }
    )
)
