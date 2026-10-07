from pathlib import Path

import pandas as pd

from config.model_config import (
    SECTORS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    PREDICTION_WINDOW_DAYS,
)


ROOT = Path(__file__).resolve().parent.parent

TRADES_FILE = (
    ROOT / "data/interim/transactions.parquet"
)

LEGISLATORS_FILE = (
    ROOT / "data/interim/legislators.parquet"
)

TICKER_SECTORS_FILE = (
    ROOT / "data/interim/ticker_sectors.parquet"
)

COMMITTEE_ASSIGNMENTS_FILE = (
    ROOT / "data/interim/committee_assignments.parquet"
)

COMMITTEE_SECTORS_FILE = (
    ROOT / "config/committee_sectors.csv"
)

OUT = (
    ROOT / "data/processed/model_data.parquet"
)

# STOCK Act: trades must be disclosed within 45 days.
DISCLOSURE_DEADLINE_DAYS = 45


SECTOR_NAME_MAP = {
    "communication services":
        "communication_services",

    "consumer discretionary":
        "consumer_discretionary",

    "consumer staples":
        "consumer_staples",

    "energy":
        "energy",

    "financials":
        "financials",

    "health care":
        "health_care",

    "healthcare":
        "health_care",

    "industrials":
        "industrials",

    "information technology":
        "information_technology",

    "technology":
        "information_technology",

    "materials":
        "materials",

    "real estate":
        "real_estate",

    "utilities":
        "utilities",
}


def normalize_sector(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    return SECTOR_NAME_MAP.get(
        value,
        value.replace(" ", "_")
    )


def congress_from_date(date):

    if date.year <= 2022:
        return 117

    if date.year <= 2024:
        return 118

    return 119


def build_committee_features():

    assignments = pd.read_parquet(
        COMMITTEE_ASSIGNMENTS_FILE
    )

    committee_sectors = pd.read_csv(
        COMMITTEE_SECTORS_FILE
    )

    committee_sectors["sector"] = (
        committee_sectors["sector"]
        .apply(normalize_sector)
    )

    merged = assignments.merge(
        committee_sectors,
        on="committee_code",
        how="left"
    )

    merged = merged[
        merged["sector"].isin(SECTORS)
    ]

    merged = (
        merged[
            [
                "memberId",
                "congress",
                "sector"
            ]
        ]
        .drop_duplicates()
    )

    return merged


def build_first_term_lookup():

    legislators = pd.read_parquet(
        LEGISLATORS_FILE
    )

    if "term_start" not in legislators.columns:
        raise ValueError(
            "legislators.parquet needs a "
            "'term_start' column to calculate tenure_years."
        )

    legislators["term_start"] = pd.to_datetime(
        legislators["term_start"]
    )

    first_terms = (
        legislators
        .groupby("memberId")["term_start"]
        .min()
        .to_dict()
    )

    return first_terms


def verify_no_leakage(model_df, trades):

    # Recompute features and targets for every row a second, vectorized
    # way and require an exact match. Features may only count trades that
    # were public before the prediction date; targets only trades made in
    # [prediction_date, prediction_date + window).
    pairs = model_df[
        ["memberId", "prediction_date"]
    ].merge(
        trades[
            ["memberId", "td", "available_at", "sector"]
        ],
        on="memberId"
    )

    p = pairs["prediction_date"]

    known = pairs["available_at"] < p

    checks = {
        "trades_last_365d":
            known
            & (pairs["available_at"] >= p - pd.Timedelta(days=365)),
        "trades_last_90d":
            known
            & (pairs["available_at"] >= p - pd.Timedelta(days=90)),
        "trades_last_21d":
            known
            & (pairs["available_at"] >= p - pd.Timedelta(days=21)),
    }

    keys = ["memberId", "prediction_date"]

    for column, mask in checks.items():

        expected = (
            mask.groupby([pairs[k] for k in keys]).sum()
            .reindex(pd.MultiIndex.from_frame(model_df[keys]))
            .fillna(0)
            .to_numpy()
        )

        assert (model_df[column].to_numpy() == expected).all(), (
            f"LEAKAGE CHECK FAILED: {column} does not match "
            "trades disclosed before prediction_date"
        )

    in_window = (
        (pairs["td"] >= p)
        & (pairs["td"] < p + pd.Timedelta(days=PREDICTION_WINDOW_DAYS))
    )

    traded = (
        pairs[in_window]
        .groupby(keys + ["sector"])
        .size()
        .gt(0)
        .unstack(fill_value=False)
        .reindex(pd.MultiIndex.from_frame(model_df[keys]), fill_value=False)
    )

    for sector in SECTORS:

        expected = (
            traded[sector].to_numpy()
            if sector in traded.columns
            else 0
        )

        assert (
            model_df[f"target_{sector}"].to_numpy() == expected
        ).all(), f"TARGET CHECK FAILED: target_{sector}"

    print(
        f"leakage check passed for {len(model_df)} rows"
    )


def main():

    trades = pd.read_parquet(
        TRADES_FILE
    )

    ticker_sectors = pd.read_parquet(
        TICKER_SECTORS_FILE
    )

    committee_features = (
        build_committee_features()
    )

    first_terms = (
        build_first_term_lookup()
    )

    # ---------------------------------
    # Add sector to each trade
    # ---------------------------------

    trades = trades.merge(
        ticker_sectors[
            ["ticker", "sector"]
        ],
        on="ticker",
        how="left"
    )

    trades["td"] = pd.to_datetime(
        trades["td"]
    )

    # When each trade became public. History features use this, not td:
    # the median trade is disclosed ~28 days after it happens, which is
    # longer than the prediction window.
    trades["available_at"] = pd.to_datetime(
        trades["available_at"]
    )

    trades["sector"] = (
        trades["sector"]
        .apply(normalize_sector)
    )

    trades["is_purchase"] = (
        trades["action"]
        .astype(str)
        .str.lower()
        .eq("purchase")
    )

    trades["is_sale"] = (
        trades["is_sale"]
        .astype(bool)
    )

    # ---------------------------------
    # Prediction dates
    # ---------------------------------

    first_date = trades["td"].min()

    # Trades made shortly before the data was pulled are mostly not
    # disclosed yet, so late target windows would be undercounted.
    # Stop where the whole target window is past the 45-day deadline.
    data_pulled_at = trades["available_at"].max()

    last_possible_date = (
        data_pulled_at
        - pd.Timedelta(
            days=DISCLOSURE_DEADLINE_DAYS
            + PREDICTION_WINDOW_DAYS
        )
    )

    prediction_dates = pd.date_range(
        start=first_date,
        end=last_possible_date,
        freq=f"{PREDICTION_WINDOW_DAYS}D"
    )

    rows = []

    member_ids = (
        trades["memberId"]
        .dropna()
        .unique()
    )

    print(
        f"Members: {len(member_ids)}"
    )

    print(
        f"Prediction dates: "
        f"{len(prediction_dates)}"
    )

    # ---------------------------------
    # Build one row for each
    # member + prediction date
    # ---------------------------------

    for member_number, member_id in enumerate(
        member_ids,
        start=1
    ):

        print(
            f"[{member_number}/"
            f"{len(member_ids)}] "
            f"{member_id}"
        )

        member_trades = (
            trades[
                trades["memberId"]
                == member_id
            ]
            .sort_values("td")
            .copy()
        )

        first_term = first_terms.get(
            member_id
        )

        for prediction_date in prediction_dates:

            # Only trades that were PUBLIC before the prediction date.
            history = member_trades[
                member_trades["available_at"]
                < prediction_date
            ]

            if history.empty:
                continue

            history_21 = history[
                history["available_at"]
                >= prediction_date
                - pd.Timedelta(days=21)
            ]

            history_90 = history[
                history["available_at"]
                >= prediction_date
                - pd.Timedelta(days=90)
            ]

            history_365 = history[
                history["available_at"]
                >= prediction_date
                - pd.Timedelta(days=365)
            ]

            # --------------------------
            # Next 21-day target window
            # --------------------------

            target_end = (
                prediction_date
                + pd.Timedelta(
                    days=PREDICTION_WINDOW_DAYS
                )
            )

            future = member_trades[
                (
                    member_trades["td"]
                    >= prediction_date
                )
                &
                (
                    member_trades["td"]
                    < target_end
                )
            ]

            # --------------------------
            # Overall trade features
            # --------------------------

            last_trade_date = (
                history["td"].max()
            )

            days_since_last_trade = (
                prediction_date
                - last_trade_date
            ).days

            if len(history_90) > 0:

                avg_amount_90d = (
                    history_90[
                        "amount_mid"
                    ]
                    .mean()
                )

                purchase_ratio_90d = (
                    history_90[
                        "is_purchase"
                    ]
                    .mean()
                )

                sale_ratio_90d = (
                    history_90[
                        "is_sale"
                    ]
                    .mean()
                )

            else:

                avg_amount_90d = 0.0
                purchase_ratio_90d = 0.0
                sale_ratio_90d = 0.0

            # --------------------------
            # Tenure feature
            # --------------------------

            if pd.notna(first_term):

                tenure_years = (
                    (
                        prediction_date
                        - first_term
                    ).days
                    / 365.25
                )

                tenure_years = max(
                    tenure_years,
                    0.0
                )

            else:

                tenure_years = 0.0

            row = {
                "memberId":
                    member_id,

                "prediction_date":
                    prediction_date,

                "trades_last_21d":
                    len(history_21),

                "trades_last_90d":
                    len(history_90),

                "trades_last_365d":
                    len(history_365),

                "days_since_last_trade":
                    days_since_last_trade,

                "avg_amount_90d":
                    avg_amount_90d,

                "purchase_ratio_90d":
                    purchase_ratio_90d,

                "sale_ratio_90d":
                    sale_ratio_90d,

                "tenure_years":
                    tenure_years,
            }

            # --------------------------
            # Historical sector features
            # --------------------------

            for sector in SECTORS:

                row[
                    f"{sector}_trades_90d"
                ] = int(
                    (
                        history_90[
                            "sector"
                        ]
                        == sector
                    ).sum()
                )

            # --------------------------
            # Committee-sector features
            # --------------------------

            congress = congress_from_date(
                prediction_date
            )

            member_committee_sectors = (
                committee_features[
                    (
                        committee_features[
                            "memberId"
                        ]
                        == member_id
                    )
                    &
                    (
                        committee_features[
                            "congress"
                        ]
                        == congress
                    )
                ]["sector"]
                .tolist()
            )

            for sector in SECTORS:

                row[
                    f"committee_{sector}"
                ] = int(
                    sector
                    in member_committee_sectors
                )

            # --------------------------
            # Next 21-day sector targets
            # --------------------------

            future_sectors = set(
                future[
                    "sector"
                ]
                .dropna()
                .tolist()
            )

            for sector in SECTORS:

                row[
                    f"target_{sector}"
                ] = int(
                    sector
                    in future_sectors
                )

            rows.append(row)

    # ---------------------------------
    # Final model-ready dataframe
    # ---------------------------------

    model_df = pd.DataFrame(
        rows
    )

    required = (
        FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in model_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model columns: "
            f"{missing}"
        )

    model_df = (
        model_df
        .sort_values(
            [
                "prediction_date",
                "memberId"
            ]
        )
        .reset_index(drop=True)
    )

    verify_no_leakage(
        model_df,
        trades
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_df.to_parquet(
        OUT,
        index=False
    )

    print()
    print(
        f"Model rows: "
        f"{len(model_df)}"
    )

    print(
        f"Members: "
        f"{model_df.memberId.nunique()}"
    )

    print(
        f"Prediction dates: "
        f"{model_df.prediction_date.nunique()}"
    )

    print(
        f"\nwrote "
        f"{OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()