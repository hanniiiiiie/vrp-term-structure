import pandas as pd
import yfinance as yf

# ---------------------------------------------------------
# 1. Load our locally aggregated daily closes
# ---------------------------------------------------------
spx_local = pd.read_csv(
    "data/raw/SPX_daily_close.csv",
    parse_dates=["date"]
)

vix_local = pd.read_csv(
    "data/raw/VIX_daily_close.csv",
    parse_dates=["date"]
)

# Dates selected for validation
dates = pd.to_datetime([
    "2018-12-24",
    "2019-12-24",
    "2024-11-29",
    "2024-12-02",
    "2025-01-02",
])

start = dates.min() - pd.Timedelta(days=2)
end   = dates.max() + pd.Timedelta(days=3)

# ---------------------------------------------------------
# 2. Download published daily closes from Yahoo Finance
# ---------------------------------------------------------
spx_ref = yf.download(
    "^GSPC",
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False
)

vix_ref = yf.download(
    "^VIX",
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False
)

# yfinance may return MultiIndex columns
if isinstance(spx_ref.columns, pd.MultiIndex):
    spx_ref.columns = spx_ref.columns.get_level_values(0)

if isinstance(vix_ref.columns, pd.MultiIndex):
    vix_ref.columns = vix_ref.columns.get_level_values(0)

spx_ref = (
    spx_ref[["Close"]]
    .reset_index()
    .rename(columns={"Date": "date", "Close": "reference_close"})
)

vix_ref = (
    vix_ref[["Close"]]
    .reset_index()
    .rename(columns={"Date": "date", "Close": "reference_close"})
)

spx_ref["date"] = pd.to_datetime(spx_ref["date"]).dt.tz_localize(None)
vix_ref["date"] = pd.to_datetime(vix_ref["date"]).dt.tz_localize(None)

# ---------------------------------------------------------
# 3. Comparison function
# ---------------------------------------------------------
def compare(name, local, reference):

    x = (
        local[["date", "close"]]
        .merge(reference, on="date", how="inner")
    )

    x = x[x["date"].isin(dates)].copy()

    x["difference"] = x["close"] - x["reference_close"]
    x["abs_difference"] = x["difference"].abs()

    print(f"\n{'=' * 65}")
    print(name)
    print("=" * 65)

    if x.empty:
        print("No matching observations.")
        return

    print(
        x[
            ["date", "close", "reference_close",
             "difference", "abs_difference"]
        ].to_string(index=False)
    )

    print("\nMaximum absolute difference:",
          x["abs_difference"].max())

# ---------------------------------------------------------
# 4. Run validation
# ---------------------------------------------------------
compare("SPX", spx_local, spx_ref)
compare("VIX", vix_local, vix_ref)
