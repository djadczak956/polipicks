from pathlib import Path
import time

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent

TRADES_FILE = ROOT / "data/interim/transactions.parquet"
OUT = ROOT / "data/interim/ticker_sectors.parquet"


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