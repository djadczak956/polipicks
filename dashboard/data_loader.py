"""Load and shape the data the dashboard shows.

Plain pandas only (no Streamlit here), so each function can be tested
or reused in a notebook. app.py wraps them in st.cache_data so they
run once per session.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.model_config import SECTORS
from dashboard.settings import (
    COMMITTEE_ASSIGNMENTS_FILE,
    COMMITTEE_SECTORS_FILE,
    LEGISLATORS_FILE,
    MODEL_DATA_FILE,
    TICKER_SECTORS_FILE,
    TRADES_FILE,
)

UNCLASSIFIED_SECTOR_LABEL = "Unclassified (ETF, fund, unknown)"


def sector_label(sector):
    """information_technology -> Information Technology"""
    return sector.replace("_", " ").title()


def sector_key(label):
    """Health Care -> health_care (inverse of sector_label)."""
    return str(label).strip().lower().replace(" ", "_")


# -------------------------------------------------------------------
# Trades
# -------------------------------------------------------------------

def load_trades():
    """One row per House trade, with its GICS sector attached."""
    trades = pd.read_parquet(TRADES_FILE)
    ticker_sectors = pd.read_parquet(TICKER_SECTORS_FILE)[["ticker", "sector"]]

    trades = trades.merge(ticker_sectors, on="ticker", how="left")
    trades["td"] = pd.to_datetime(trades["td"])
    trades["quarter"] = trades["td"].dt.to_period("Q").astype(str)
    trades["sector_label"] = trades["sector"].map(
        lambda sector: sector_label(sector) if sector in SECTORS else UNCLASSIFIED_SECTOR_LABEL
    )
    return trades


def format_amount_range(low, high):
    """Disclosures report a dollar range, not an exact amount."""
    if pd.isna(low):
        return "Not reported"
    if pd.isna(high):
        return f"Over ${low:,.0f}"
    if low == high:
        return f"${low:,.0f}"
    return f"${low:,.0f} – ${high:,.0f}"


# -------------------------------------------------------------------
# Members
# -------------------------------------------------------------------

def load_member_info():
    """Name, party, state, committees and the sectors those committees oversee.

    Uses each member's most recent Congress in the data, so committees
    reflect their latest assignments.
    """
    legislators = pd.read_parquet(LEGISLATORS_FILE)
    latest_terms = (
        legislators.sort_values("congress")
        .groupby("memberId")
        .tail(1)
        [["memberId", "congress", "name", "party", "state", "district", "term_start"]]
    )

    assignments = pd.read_parquet(COMMITTEE_ASSIGNMENTS_FILE)
    committee_sectors = pd.read_csv(COMMITTEE_SECTORS_FILE)
    committee_sectors["sector"] = committee_sectors["sector"].map(sector_key)

    latest_assignments = assignments.merge(
        latest_terms[["memberId", "congress"]], on=["memberId", "congress"]
    )

    committees_by_member = (
        latest_assignments.groupby("memberId")["committee_name"]
        .apply(lambda names: sorted(names.dropna().unique()))
        .rename("committees")
    )

    overseen_sectors_by_member = (
        latest_assignments.merge(committee_sectors, on="committee_code")
        .groupby("memberId")["sector"]
        .apply(lambda sectors: sorted(sectors.unique()))
        .rename("committee_sectors")
    )

    members = (
        latest_terms.set_index("memberId")
        .join(committees_by_member)
        .join(overseen_sectors_by_member)
        .reset_index()
    )

    # Members with no matched committees get empty lists, not NaN.
    for column in ["committees", "committee_sectors"]:
        members[column] = members[column].apply(
            lambda value: value if isinstance(value, list) else []
        )

    members["party_letter"] = members["party"].fillna("?").str[0]
    members["display_name"] = (
        members["name"] + " (" + members["party_letter"] + "-" + members["state"].fillna("?") + ")"
    )
    return members


def summarize_member_activity(trades):
    """Per-member trade counts, used for the member picker and leaderboard."""
    summary = trades.groupby("memberId").agg(
        trades=("ticker", "size"),
        first_trade=("td", "min"),
        last_trade=("td", "max"),
        tickers=("ticker", "nunique"),
        estimated_volume=("amount_mid", "sum"),
    )
    summary["share_of_all_trades"] = summary["trades"] / summary["trades"].sum()
    return summary.sort_values("trades", ascending=False).reset_index()


def member_sector_mix(member_trades, overseen_sectors):
    """Share of a member's classified trades in each sector, tagged by
    whether one of their committees oversees that sector."""
    classified = member_trades[member_trades["sector"].isin(SECTORS)]

    counts = classified["sector"].value_counts().reindex(SECTORS, fill_value=0)
    mix = pd.DataFrame({
        "sector": SECTORS,
        "sector_label": [sector_label(sector) for sector in SECTORS],
        "trades": counts.to_numpy(),
    })
    mix["share"] = mix["trades"] / max(mix["trades"].sum(), 1)
    mix["overseen_by_committee"] = mix["sector"].isin(overseen_sectors)
    return mix.sort_values("share", ascending=True)


# -------------------------------------------------------------------
# Committees vs. trading (the researcher question)
# -------------------------------------------------------------------

def build_committee_vs_trading():
    """Does sitting on a committee make a member more likely to trade the
    sectors it oversees?

    Uses model_data (one row per member and 21-day window). Only windows
    where the member traded something are kept, so we compare WHAT
    active traders buy and sell, not WHETHER they trade at all.

    For each sector:
      rate_on_committee  = share of active windows with a trade in the
                           sector, for members on an overseeing committee
      rate_off_committee = the same, for everyone else
      lift               = rate_on / rate_off  (1.0 means no difference)
    """
    model_data = pd.read_parquet(MODEL_DATA_FILE)
    target_columns = [f"target_{sector}" for sector in SECTORS]
    active_windows = model_data[model_data[target_columns].sum(axis=1) > 0]

    rows = []
    for sector in SECTORS:
        on_committee = active_windows[f"committee_{sector}"] == 1
        traded_sector = active_windows[f"target_{sector}"]

        rows.append({
            "sector": sector,
            "sector_label": sector_label(sector),
            "rate_on_committee": traded_sector[on_committee].mean(),
            "rate_off_committee": traded_sector[~on_committee].mean(),
            "windows_on_committee": int(on_committee.sum()),
            "members_on_committee": active_windows.loc[on_committee, "memberId"].nunique(),
        })

    table = pd.DataFrame(rows)
    table["lift"] = table["rate_on_committee"] / table["rate_off_committee"]

    # A sector no committee in committee_sectors.csv maps to has nothing to compare.
    unmapped_sectors = table.loc[table["windows_on_committee"] == 0, "sector_label"].tolist()
    table = table[table["windows_on_committee"] > 0].reset_index(drop=True)

    return table, len(active_windows), unmapped_sectors


# -------------------------------------------------------------------
# Optional inputs from other branches
# -------------------------------------------------------------------

def load_optional_table(path, required_columns):
    """Return (table, problem). table is None when the file is missing
    or doesn't have the columns the dashboard needs."""
    if not path.exists():
        return None, f"`{path.name}` hasn't been generated yet."

    table = pd.read_parquet(path)
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        return None, f"`{path.name}` is missing columns: {', '.join(missing)}"

    return table, None


def prediction_sectors(predictions):
    """Sectors that have both a target_<sector> and prob_<sector> column."""
    return [
        sector for sector in SECTORS
        if f"target_{sector}" in predictions.columns and f"prob_{sector}" in predictions.columns
    ]


def compute_model_metrics(predictions, threshold):
    """The metrics listed in the README: per-label F1, Hamming loss,
    subset accuracy. AUC is added because it doesn't depend on the
    threshold (the ROC/AUC material from the evaluation-metrics lecture).
    """
    sectors = prediction_sectors(predictions)
    y_true = predictions[[f"target_{sector}" for sector in sectors]].to_numpy().astype(int)
    probabilities = predictions[[f"prob_{sector}" for sector in sectors]].to_numpy()
    y_pred = (probabilities >= threshold).astype(int)

    per_sector = []
    for index, sector in enumerate(sectors):
        actual, predicted = y_true[:, index], y_pred[:, index]
        has_both_classes = 0 < actual.sum() < len(actual)
        per_sector.append({
            "sector_label": sector_label(sector),
            "positive_rate": actual.mean(),
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
            "f1": f1_score(actual, predicted, zero_division=0),
            "auc": roc_auc_score(actual, probabilities[:, index]) if has_both_classes else np.nan,
        })

    overall = {
        "Micro F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Hamming loss": hamming_loss(y_true, y_pred),
        "Subset accuracy": (y_true == y_pred).all(axis=1).mean(),
    }

    return pd.DataFrame(per_sector), overall
