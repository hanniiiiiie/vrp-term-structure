import pandas as pd
import statsmodels.api as sm

df = pd.read_csv(
    "data/processed/vrp_monthly_snapshot.csv",
    parse_dates=["date"]
)

df["slope_6m_1m"] = df["vrp_6m"] - df["vrp_1m"]

x = df.dropna(subset=["slope_6m_1m", "vix"]).copy()

# Continuous month-end specification:
# slope_6m_1m = alpha + beta * VIX + error

X = sm.add_constant(x["vix"].astype(float))
y = x["slope_6m_1m"].astype(float)

result = sm.OLS(y, X).fit(
    cov_type="HAC",
    cov_kwds={
        "maxlags": 6,
        "kernel": "bartlett",
        "use_correction": True,
    },
    use_t=True,
)

print("=== MONTH-END REGIME ROBUSTNESS ===")
print("Observations:", len(x))

print("\nContinuous specification:")
print(f"Intercept:        {result.params['const']:.6f}")
print(f"VIX coefficient:  {result.params['vix']:.6f}")
print(f"HAC SE:           {result.bse['vix']:.6f}")
print(f"t-stat:           {result.tvalues['vix']:.3f}")
print(f"p-value:          {result.pvalues['vix']:.4f}")

ci = result.conf_int().loc["vix"]

print(
    f"95% CI:           [{ci.iloc[0]:.6f}, "
    f"{ci.iloc[1]:.6f}]"
)

# Descriptive VIX quintiles
x["vix_regime"] = pd.qcut(
    x["vix"],
    5,
    labels=[
        "Q1 lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 highest"
    ]
)

table = (
    x.groupby("vix_regime", observed=True)
     .agg(
         n=("slope_6m_1m", "size"),
         mean_vix=("vix", "mean"),
         mean_slope=("slope_6m_1m", "mean")
     )
)

print("\nMonth-end regime means:")
print(table.to_string())
