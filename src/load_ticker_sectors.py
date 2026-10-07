from pathlib import Path
import time

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent

TRADES_FILE = ROOT / "data/interim/transactions.parquet"
OUT = ROOT / "data/interim/ticker_sectors.parquet"
OVERRIDES_FILE = ROOT / "config/ticker_sector_overrides.csv"


YAHOO_TO_GICS = {
    "Communication Services": "communication_services",
    "Consumer Cyclical": "consumer_discretionary",
    "Consumer Defensive": "consumer_staples",
    "Energy": "energy",
    "Financial Services": "financials",
    "Healthcare": "health_care",
    "Industrials": "industrials",
    "Technology": "information_technology",
    "Basic Materials": "materials",
    "Real Estate": "real_estate",
    "Utilities": "utilities",
}


def lookup_sector(ticker):

    # Yahoo usually uses - instead of . for tickers
    # such as BRK.B -> BRK-B.
    yahoo_ticker = ticker.replace(".", "-")

    try:
        info = yf.Ticker(yahoo_ticker).get_info()

        yahoo_sector = info.get("sector")

        sector = YAHOO_TO_GICS.get(
            yahoo_sector
        )

        return yahoo_sector, sector

    except Exception as error:

        print(
            f"  failed {ticker}: {error}"
        )

        return None, None


def apply_overrides(df):

    # Yahoo has no data for delisted, renamed or acquired companies
    # (FB, ATVI, SIVB, ...). The hand-made overrides file fills those in.
    # Funds/ETFs are listed with kind=fund and no sector on purpose:
    # they are not a bet on one sector.
    overrides = pd.read_csv(
        OVERRIDES_FILE,
        keep_default_na=False
    )

    # Start from Yahoo's answer every run so edits to the overrides
    # file take effect even though the lookups themselves are cached.
    df["sector"] = df["yahoo_sector"].map(YAHOO_TO_GICS)

    df = df.drop(
        columns=["kind"],
        errors="ignore"
    ).merge(
        overrides.rename(
            columns={"sector": "override_sector"}
        ),
        on="ticker",
        how="left"
    )

    fill = (
        df["sector"].isna()
        & df["override_sector"].fillna("").ne("")
    )

    df.loc[fill, "sector"] = df.loc[fill, "override_sector"]

    df["kind"] = df["kind"].fillna("company")

    print(
        f"Overrides applied: {fill.sum()} sectors, "
        f"{df['kind'].eq('fund').sum()} funds"
    )

    return df.drop(columns=["override_sector"])


def main():

    if not TRADES_FILE.exists():
        raise FileNotFoundError(
            "Run load_trades.py first."
        )

    trades = pd.read_parquet(
        TRADES_FILE
    )

    tickers = sorted(
        trades["ticker"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(f"Unique tickers: {len(tickers)}")

    # If this script was interrupted earlier,
    # continue from what was already saved.
    if OUT.exists():

        existing = pd.read_parquet(
            OUT
        )

        completed = set(
            existing["ticker"]
        )

        rows = existing.to_dict(
            "records"
        )

        print(
            f"Already completed: "
            f"{len(completed)}"
        )

    else:

        completed = set()
        rows = []

    remaining = [
        ticker
        for ticker in tickers
        if ticker not in completed
    ]

    for i, ticker in enumerate(
        remaining,
        start=1
    ):

        print(
            f"[{i}/{len(remaining)}] "
            f"{ticker}"
        )

        yahoo_sector, sector = (
            lookup_sector(ticker)
        )

        rows.append({
            "ticker": ticker,
            "yahoo_sector": yahoo_sector,
            "sector": sector,
        })

        # Save periodically so progress is not lost.
        if i % 50 == 0:

            pd.DataFrame(
                rows
            ).to_parquet(
                OUT,
                index=False
            )

        time.sleep(0.05)

    df = pd.DataFrame(rows)

    df = (
        df
        .drop_duplicates(
            subset=["ticker"],
            keep="last"
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    df = apply_overrides(df)

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUT,
        index=False
    )

    print()
    print(f"Ticker rows: {len(df)}")
    print(
        f"Sector matched: "
        f"{df['sector'].notna().sum()}"
    )
    print(
        f"Sector missing: "
        f"{df['sector'].isna().sum()}"
    )

    print(
        f"\nwrote {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()