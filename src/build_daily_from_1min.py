"""
build_daily_from_1min.py
------------------------
Aggregates FirstRate 1-minute index files into daily closes.

Replaces the Bloomberg acquisition step for v1. Reads four index series
(SPX, VIX, VIX3M, VIX6M), applies ONE uniform close rule to all of them,
and writes one CSV per series into data/raw/ plus a source log.

Close rule (frozen): the last available bar at or before 16:00:00 US Eastern.
NOT "the 16:00 bar" -- FirstRate omits bars with no volume, so a 16:00 bar
may not exist on a given day.

This module does no cleaning, no reindexing, no forward-filling, no merging.
All of that belongs in the notebook where it is visible.

Usage:
    python build_daily_from_1min.py --src "C:/risklab/data/first_rate_data/indices" --out ./data/raw
    python build_daily_from_1min.py --src ... --out ... --force
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

# --- frozen configuration ----------------------------------------------------

TICKERS = ["SPX", "VIX", "VIX3M", "VIX6M"]
FILE_PATTERN = "{ticker}_full_1min.txt"
CLOSE_CUTOFF = dt.time(16, 0, 0)          # US Eastern; FirstRate index files are ET
COLUMNS = ["datetime", "open", "high", "low", "close"]   # index files carry no volume


def read_1min(path: Path) -> pd.DataFrame:
    """Read one FirstRate 1-minute index file.

    Header detection: FirstRate ships these without a header row, but the
    bundle is not perfectly uniform, so we sniff the first line instead of
    assuming.
    """
    with open(path, "r") as f:
        first = f.readline().strip()
    has_header = not first[:4].isdigit()      # a data row starts with a year

    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=COLUMNS,
        usecols=range(len(COLUMNS)),
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    return df


def to_daily_close(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 1-minute bars to one close per trading date.

    Keeps the timestamp of the selected bar and the bar count, both of which
    are diagnostics -- a selected bar far from 16:00, or an unusually low bar
    count, flags a day worth inspecting rather than trusting.
    """
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    eligible = df[df["time"] <= CLOSE_CUTOFF]
    eligible = eligible.sort_values("datetime")

    daily = (
        eligible.groupby("date")
        .agg(
            close=("close", "last"),
            close_timestamp=("datetime", "last"),
            n_bars=("close", "size"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def log_row(ticker: str, src: Path, daily: pd.DataFrame) -> dict:
    """One row of the source log -- deliverable #1 of the brief."""
    return {
        "series": ticker,
        "source": "FirstRateData",
        "source_file": src.name,
        "field": "close",
        "native_frequency": "1min",
        "derived_frequency": "daily",
        "close_rule": f"last bar <= {CLOSE_CUTOFF} ET",
        "timezone": "US/Eastern",
        "currency": "USD",
        "adjustment": "none",
        "n_obs": len(daily),
        "first_obs_date": daily["date"].min().date().isoformat(),
        "last_obs_date": daily["date"].max().date().isoformat(),
        "extraction_datetime": dt.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding *_full_1min.txt")
    ap.add_argument("--out", default="./data/raw", help="output folder")
    ap.add_argument("--force", action="store_true", help="rebuild existing outputs")
    args = ap.parse_args()

    src_dir, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for ticker in TICKERS:
        out_path = out_dir / f"{ticker}_daily_close.csv"
        if out_path.exists() and not args.force:
            print(f"[skip] {ticker}: {out_path.name} exists (use --force to rebuild)")
            continue

        src_path = src_dir / FILE_PATTERN.format(ticker=ticker)
        if not src_path.exists():
            print(f"[MISSING] {ticker}: {src_path}")
            continue

        print(f"[read] {ticker} ...", end=" ", flush=True)
        daily = to_daily_close(read_1min(src_path))
        daily.to_csv(out_path, index=False, date_format="%Y-%m-%d")

        row = log_row(ticker, src_path, daily)
        log.append(row)
        print(f"{row['n_obs']} days, {row['first_obs_date']} -> {row['last_obs_date']}")

    if log:
        log_path = out_dir / "source_log.csv"
        new = pd.DataFrame(log)
        if log_path.exists():
            new = pd.concat([pd.read_csv(log_path), new], ignore_index=True)
        new.to_csv(log_path, index=False)
        print(f"\n[log] {log_path}")


if __name__ == "__main__":
    main()
