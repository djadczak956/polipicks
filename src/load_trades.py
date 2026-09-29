"""Step 1: combine the raw trade parquets into one clean House transaction table.

Reads data/raw/*.parquet, writes data/interim/transactions.parquet.
No joins happen here; this script only cleans the trade source.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILES = [ROOT / f"data/raw/{y}-00000-of-00001.parquet" for y in range(2021, 2027)]
OUT = ROOT / "data/interim/transactions.parquet"

DATE_START, DATE_END = "2021-01-01", "2026-09-30"

OUT_COLS = [
    "memberId", "ticker", "assetDescription", "action", "is_sale", "partialSale",
    "td", "quarter", "amountLow", "amountHigh", "amount_mid", "owner",
    "sourceDocId", "sourceUrl",
]


def main():
    # File year is the FILING year, not the trade year, so we never use it.
    df = pd.concat([pd.read_parquet(f) for f in RAW_FILES], ignore_index=True)
    print(f"raw rows loaded                = {len(df)}")

    # transactionDate holds datetime.date objects; convert before comparing.
    df["td"] = pd.to_datetime(df.transactionDate, errors="coerce")

    # Filter 1: an amended filing keeps the old row with supersededAt set.
    # Dropping those rows prevents double-counting.
    df = df[df.supersededAt.isna()]
    print(f"after filter 1 (superseded)    = {len(df)}")

    df = df[df.chamber == "house"]
    print(f"after filter 2 (house)         = {len(df)}")

    # assetTypeCode is mostly null, so a non-null ticker is our equity filter.
    df = df[df.ticker.notna()]
    print(f"after filter 3 (ticker)        = {len(df)}")

    df = df[df.td.between(DATE_START, DATE_END)]
    print(f"after filter 4 (date window)   = {len(df)}")

    df = df.assign(
        quarter=df.td.dt.to_period("Q"),
        is_sale=df.action.isin(["sale"]),
        amount_mid=(df.amountLow + df.amountHigh) / 2,
    )[OUT_COLS].reset_index(drop=True)

    print(f"distinct memberId              = {df.memberId.nunique()}")
    print(f"distinct ticker                = {df.ticker.nunique()}")
    print(f"distinct quarter               = {df.quarter.nunique()}")
    print(f"memberId nulls                 = {df.memberId.isna().sum()}")

    top = df.memberId.value_counts().head(5)
    print("\ntop 5 members by trade count:")
    for member, n in top.items():
        print(f"  {member}  {n:>6}  {n / len(df):6.1%}")
    print(f"  top 2 combined share: {top.iloc[:2].sum() / len(df):.1%}")

    assert len(df) == 58162, len(df)
    assert df.memberId.nunique() == 201
    assert df.ticker.nunique() == 2643
    assert df.quarter.nunique() == 23
    assert df.memberId.notna().all()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
