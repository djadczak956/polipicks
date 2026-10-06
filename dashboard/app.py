"""PoliPicks dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Tabs
  Overview               - the dataset at a glance
  Member profile         - one representative's trading (for the public)
  Committees vs. trading - does committee jurisdiction predict trades? (researchers)
  Flagged trades         - anomaly results to review (journalists)
  Model performance      - how well the sector classifier forecasts
"""

import sys
from pathlib import Path

# `streamlit run` only puts dashboard/ on the path; add the project root
# so `config` and `dashboard` import the same way they do elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data_loader
from dashboard.settings import (
    DEFAULT_FLAGS_TO_SHOW,
    DEFAULT_PREDICTION_THRESHOLD,
    FLAGGED_TRADES_FILE,
    FLAGGED_TRADES_REQUIRED_COLUMNS,
    HIGHLIGHT_COLOR,
    MODEL_PREDICTIONS_FILE,
    MODEL_PREDICTIONS_REQUIRED_COLUMNS,
    PRIMARY_COLOR,
    RECENT_TRADES_TO_SHOW,
)


st.set_page_config(page_title="PoliPicks", layout="wide")


# -------------------------------------------------------------------
# Cached loaders (each runs once per session)
# -------------------------------------------------------------------

@st.cache_data
def get_trades():
    return data_loader.load_trades()


@st.cache_data
def get_members():
    return data_loader.load_member_info()


@st.cache_data
def get_member_activity():
    return data_loader.summarize_member_activity(get_trades())


@st.cache_data
def get_committee_vs_trading():
    return data_loader.build_committee_vs_trading()


@st.cache_data
def get_flagged_trades():
    return data_loader.load_optional_table(FLAGGED_TRADES_FILE, FLAGGED_TRADES_REQUIRED_COLUMNS)


@st.cache_data
def get_model_predictions():
    return data_loader.load_optional_table(MODEL_PREDICTIONS_FILE, MODEL_PREDICTIONS_REQUIRED_COLUMNS)


# -------------------------------------------------------------------
# Small shared helpers
# -------------------------------------------------------------------

def horizontal_bar(table, x, y, color=None, color_map=None, x_title="", hover_format=".0%"):
    """Horizontal bar chart in the dashboard's style."""
    figure = px.bar(
        table, x=x, y=y, orientation="h",
        color=color,
        color_discrete_map=color_map,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    figure.update_traces(hovertemplate=f"%{{y}}: %{{x:{hover_format}}}<extra></extra>")
    figure.update_layout(
        xaxis_title=x_title, yaxis_title=None, legend_title=None,
        xaxis_tickformat=".0%" if hover_format.endswith("%") else None,
        margin=dict(l=0, r=10, t=10, b=0), height=380,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return figure


def add_member_names(table, members):
    lookup = members.set_index("memberId")["display_name"]
    table = table.copy()
    table.insert(0, "member", table["memberId"].map(lookup).fillna(table["memberId"]))
    return table


def show_missing_input(problem, file_path, owner, required_columns_text):
    st.info(
        f"{problem}\n\n"
        f"This tab turns on automatically once **{owner}** writes "
        f"`{file_path.relative_to(file_path.parents[2])}` with these columns:\n\n"
        f"{required_columns_text}\n\nSee `dashboard/README.md` for details."
    )


# -------------------------------------------------------------------
# Tab: Overview
# -------------------------------------------------------------------

def render_overview(trades, member_activity, members):
    columns = st.columns(4)
    columns[0].metric("House trades", f"{len(trades):,}")
    columns[1].metric("Members who traded", f"{trades['memberId'].nunique():,}")
    columns[2].metric("Distinct tickers", f"{trades['ticker'].nunique():,}")
    columns[3].metric("Quarters covered", trades["quarter"].nunique())

    left, right = st.columns(2)

    with left:
        st.subheader("Trades per quarter")
        per_quarter = trades.groupby("quarter").size().rename("trades").reset_index()
        figure = px.bar(per_quarter, x="quarter", y="trades", color_discrete_sequence=[PRIMARY_COLOR])
        figure.update_traces(hovertemplate="%{x}: %{y:,} trades<extra></extra>")
        figure.update_layout(xaxis_title=None, yaxis_title="Trades", margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(figure, width="stretch")
        st.caption("The latest quarter is incomplete: members have 45 days to disclose a trade.")

    with right:
        st.subheader("Trades by sector")
        by_sector = trades["sector_label"].value_counts(normalize=True).rename("share").reset_index()
        by_sector = by_sector.sort_values("share")
        st.plotly_chart(
            horizontal_bar(by_sector, x="share", y="sector_label", x_title="Share of all trades"),
            width="stretch",
        )

    st.subheader("Most active traders")
    top_traders = add_member_names(member_activity.head(10), members)
    top_two_share = member_activity["share_of_all_trades"].head(2).sum()
    st.caption(
        f"The top two members account for {top_two_share:.0%} of all trades, "
        "which is why the pipeline weights or caps rows per member."
    )
    st.dataframe(
        top_traders[["member", "trades", "share_of_all_trades", "tickers", "first_trade", "last_trade"]],
        hide_index=True, width="stretch",
        column_config={
            "share_of_all_trades": st.column_config.ProgressColumn(
                "Share of all trades", format="percent", min_value=0, max_value=1
            ),
            "first_trade": st.column_config.DateColumn("First trade"),
            "last_trade": st.column_config.DateColumn("Last trade"),
        },
    )


# -------------------------------------------------------------------
# Tab: Member profile
# -------------------------------------------------------------------

def render_member_profile(trades, member_activity, members, flagged_trades, predictions):
    member_lookup = members.set_index("memberId")
    traders = member_activity["memberId"].tolist()   # already sorted by trade count

    selected_id = st.selectbox(
        "Representative (sorted by number of trades)",
        traders,
        format_func=lambda member_id: member_lookup["display_name"].get(member_id, member_id),
    )

    member = member_lookup.loc[selected_id] if selected_id in member_lookup.index else None
    member_trades = trades[trades["memberId"] == selected_id].sort_values("td", ascending=False)
    activity = member_activity.set_index("memberId").loc[selected_id]

    columns = st.columns(4)
    columns[0].metric("Trades disclosed", f"{int(activity['trades']):,}")
    columns[1].metric("Distinct tickers", f"{int(activity['tickers']):,}")
    columns[2].metric("Years active", f"{activity['first_trade']:%Y}–{activity['last_trade']:%Y}")
    columns[3].metric("Est. volume (range midpoints)", f"${activity['estimated_volume'] / 1e6:,.1f}M")

    overseen_sectors = []
    if member is not None:
        overseen_sectors = member["committee_sectors"]
        committees = ", ".join(member["committees"]) or "No committees matched"
        st.markdown(f"**Committees ({int(member['congress'])}th Congress):** {committees}")

    left, right = st.columns(2)

    with left:
        st.subheader("Sector mix")
        mix = data_loader.member_sector_mix(member_trades, overseen_sectors)
        mix["Committee oversight"] = mix["overseen_by_committee"].map(
            {True: "Overseen by their committee", False: "Other sectors"}
        )
        st.plotly_chart(
            horizontal_bar(
                mix, x="share", y="sector_label",
                color="Committee oversight",
                color_map={"Overseen by their committee": HIGHLIGHT_COLOR, "Other sectors": PRIMARY_COLOR},
                x_title="Share of this member's trades",
            ),
            width="stretch",
        )

    with right:
        st.subheader("Trades per quarter")
        per_quarter = member_trades.groupby("quarter").size().rename("trades").reset_index()
        figure = px.bar(per_quarter, x="quarter", y="trades", color_discrete_sequence=[PRIMARY_COLOR])
        figure.update_traces(hovertemplate="%{x}: %{y:,} trades<extra></extra>")
        figure.update_layout(xaxis_title=None, yaxis_title="Trades", margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(figure, width="stretch")

    if predictions is not None:
        render_member_forecast(selected_id, predictions)

    if flagged_trades is not None:
        member_flags = flagged_trades[flagged_trades["memberId"] == selected_id]
        st.subheader(f"Flagged trades ({len(member_flags)})")
        if member_flags.empty:
            st.write("None of this member's trades were flagged.")
        else:
            show_flag_table(member_flags.sort_values("anomaly_score", ascending=False), members)

    st.subheader("Recent trades")
    recent = member_trades.head(RECENT_TRADES_TO_SHOW).copy()
    recent["amount"] = [
        data_loader.format_amount_range(low, high)
        for low, high in zip(recent["amountLow"], recent["amountHigh"])
    ]
    st.dataframe(
        recent[["td", "ticker", "assetDescription", "sector_label", "action", "amount", "owner", "sourceUrl"]],
        hide_index=True, width="stretch",
        column_config={
            "td": st.column_config.DateColumn("Trade date"),
            "assetDescription": "Asset",
            "sector_label": "Sector",
            "sourceUrl": st.column_config.LinkColumn("Filing", display_text="PDF"),
        },
    )


def render_member_forecast(member_id, predictions):
    """The model's probabilities for this member's most recent window."""
    member_rows = predictions[predictions["memberId"] == member_id]
    if member_rows.empty:
        return

    latest = member_rows.sort_values("prediction_date").iloc[-1]
    sectors = data_loader.prediction_sectors(predictions)
    forecast = pd.DataFrame({
        "sector_label": [data_loader.sector_label(sector) for sector in sectors],
        "probability": [latest[f"prob_{sector}"] for sector in sectors],
    }).sort_values("probability")

    st.subheader(f"Model forecast for the window after {pd.Timestamp(latest['prediction_date']):%b %d, %Y}")
    st.plotly_chart(
        horizontal_bar(forecast, x="probability", y="sector_label", x_title="Predicted probability of a trade"),
        width="stretch",
    )


# -------------------------------------------------------------------
# Tab: Committees vs. trading
# -------------------------------------------------------------------

def render_committee_analysis():
    table, active_window_count, unmapped_sectors = get_committee_vs_trading()

    st.markdown(
        "For each sector: among member-windows where the member traded **something**, "
        "how often did they trade that sector? Compared for members **on** a committee "
        "that oversees the sector versus everyone else. "
        f"Based on {active_window_count:,} active 21-day windows."
    )

    long_table = table.melt(
        id_vars="sector_label",
        value_vars=["rate_on_committee", "rate_off_committee"],
        var_name="group", value_name="rate",
    )
    long_table["group"] = long_table["group"].map({
        "rate_on_committee": "On an overseeing committee",
        "rate_off_committee": "Not on one",
    })

    sector_order = table.sort_values("lift")["sector_label"].tolist()
    figure = px.bar(
        long_table, x="rate", y="sector_label", color="group", barmode="group", orientation="h",
        category_orders={"sector_label": sector_order[::-1]},
        color_discrete_map={"On an overseeing committee": HIGHLIGHT_COLOR, "Not on one": PRIMARY_COLOR},
    )
    figure.update_traces(hovertemplate="%{y}: %{x:.0%}<extra></extra>")
    figure.update_layout(
        xaxis_title="Share of active windows with a trade in the sector", yaxis_title=None,
        xaxis_tickformat=".0%",
        legend_title=None, legend=dict(orientation="h", y=1.06, x=0),
        margin=dict(l=0, r=10, t=10, b=0), height=520, bargap=0.25,
    )
    st.plotly_chart(figure, width="stretch")

    st.dataframe(
        table.sort_values("lift", ascending=False)[
            ["sector_label", "rate_on_committee", "rate_off_committee", "lift", "members_on_committee"]
        ],
        hide_index=True, width="stretch",
        column_config={
            "sector_label": "Sector",
            "rate_on_committee": st.column_config.NumberColumn("On committee", format="percent"),
            "rate_off_committee": st.column_config.NumberColumn("Not on committee", format="percent"),
            "lift": st.column_config.NumberColumn("Lift (×)", format="%.2f"),
            "members_on_committee": "Members on committee",
        },
    )
    st.caption(
        "Lift above 1 means committee members trade that sector more often. This is a "
        "correlation, not evidence of wrongdoing; members also choose committees that "
        "match their interests. Sectors with few committee members are noisy."
    )
    if unmapped_sectors:
        st.caption(
            f"Not shown (no committee mapped to it in `config/committee_sectors.csv`): "
            f"{', '.join(unmapped_sectors)}."
        )


# -------------------------------------------------------------------
# Tab: Flagged trades
# -------------------------------------------------------------------

FLAG_TABLE_OPTIONAL_COLUMNS = [
    "assetDescription", "sector_label", "amount", "flag_reasons", "sourceUrl",
]


def show_flag_table(flags, members):
    table = add_member_names(flags, members)
    if "sector_label" not in table.columns:
        table["sector_label"] = table["sector"].map(
            lambda sector: data_loader.sector_label(sector) if isinstance(sector, str) else sector
        )
    if {"amountLow", "amountHigh"} <= set(table.columns):
        table["amount"] = [
            data_loader.format_amount_range(low, high)
            for low, high in zip(table["amountLow"], table["amountHigh"])
        ]

    columns = ["member", "td", "ticker"] + [
        column for column in FLAG_TABLE_OPTIONAL_COLUMNS if column in table.columns
    ] + ["anomaly_score"]

    st.dataframe(
        table[columns], hide_index=True, width="stretch",
        column_config={
            "td": st.column_config.DateColumn("Trade date"),
            "assetDescription": "Asset",
            "sector_label": "Sector",
            "flag_reasons": st.column_config.TextColumn("Why it was flagged", width="large"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly score", format="%.2f"),
            "sourceUrl": st.column_config.LinkColumn("Filing", display_text="PDF"),
        },
    )


def render_flagged_trades(flagged_trades, problem, members):
    if flagged_trades is None:
        show_missing_input(
            problem, FLAGGED_TRADES_FILE, "the anomaly-detection branch",
            "- required: `memberId`, `td` (trade date), `ticker`, `sector`, `anomaly_score` "
            "(higher = more unusual)\n"
            "- optional: `is_flagged`, `flag_reasons`, `assetDescription`, `amountLow`, "
            "`amountHigh`, `sourceUrl`",
        )
        return

    flags = flagged_trades.copy()
    flags["td"] = pd.to_datetime(flags["td"])
    # Take party from our member table so the filter works whatever the input file has.
    flags = flags.drop(columns=["party"], errors="ignore").merge(
        members[["memberId", "party"]], on="memberId", how="left"
    )

    filter_columns = st.columns(4)
    only_flagged = filter_columns[0].toggle(
        "Only trades flagged by the detector", value=True, disabled="is_flagged" not in flags.columns
    )
    parties = filter_columns[1].multiselect("Party", sorted(flags["party"].dropna().unique()))
    sectors = filter_columns[2].multiselect("Sector", sorted(flags["sector"].dropna().unique()),
                                            format_func=data_loader.sector_label)
    rows_to_show = filter_columns[3].slider("Rows to show", 10, 500, DEFAULT_FLAGS_TO_SHOW, step=10)

    if only_flagged and "is_flagged" in flags.columns:
        flags = flags[flags["is_flagged"]]
    if parties:
        flags = flags[flags["party"].isin(parties)]
    if sectors:
        flags = flags[flags["sector"].isin(sectors)]

    columns = st.columns(3)
    columns[0].metric("Trades shown", f"{len(flags):,}")
    columns[1].metric("Members", f"{flags['memberId'].nunique():,}")
    if "sourceUrl" in flags.columns:
        columns[2].metric("Filings to review", f"{flags['sourceUrl'].nunique():,}")

    st.caption("Sorted by anomaly score, most unusual first. A flag means 'worth a look', not wrongdoing.")
    show_flag_table(flags.sort_values("anomaly_score", ascending=False).head(rows_to_show), members)


# -------------------------------------------------------------------
# Tab: Model performance
# -------------------------------------------------------------------

def render_model_performance(predictions, problem):
    if predictions is None:
        show_missing_input(
            problem, MODEL_PREDICTIONS_FILE, "the model-tuning work",
            "- `memberId`, `prediction_date`\n"
            "- `target_<sector>` (0/1, what actually happened) for each of the 11 sectors\n"
            "- `prob_<sector>` (the model's sigmoid output) for each of the 11 sectors\n\n"
            "Only out-of-sample rows (each fold's validation set), so the numbers are honest.",
        )
        return

    threshold = st.slider(
        "Decision threshold (predict 'will trade' when probability ≥ threshold)",
        0.05, 0.95, DEFAULT_PREDICTION_THRESHOLD, step=0.05,
    )
    per_sector, overall = data_loader.compute_model_metrics(predictions, threshold)

    columns = st.columns(len(overall))
    for column, (name, value) in zip(columns, overall.items()):
        column.metric(name, f"{value:.3f}")
    st.caption(
        "Hamming loss = share of individual sector labels the model got wrong (lower is better). "
        "Subset accuracy = share of windows where all 11 labels are right. Because most labels "
        "are 0, both look good even for a weak model, so focus on F1 and AUC."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("F1 by sector")
        st.plotly_chart(
            horizontal_bar(per_sector.sort_values("f1"), x="f1", y="sector_label",
                           x_title="F1 at this threshold", hover_format=".2f"),
            width="stretch",
        )
    with right:
        st.subheader("AUC by sector")
        st.plotly_chart(
            horizontal_bar(per_sector.sort_values("auc"), x="auc", y="sector_label",
                           x_title="ROC AUC (threshold-free, 0.5 = random)", hover_format=".2f"),
            width="stretch",
        )

    st.dataframe(
        per_sector, hide_index=True, width="stretch",
        column_config={
            "sector_label": "Sector",
            "positive_rate": st.column_config.NumberColumn("How often it happens", format="percent"),
            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
        },
    )


# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------

def main():
    st.title("PoliPicks")
    st.caption(
        "Which sectors House members trade, whether it tracks their committees, "
        "and which disclosures break the pattern. Source: STOCK Act Periodic Transaction Reports."
    )

    trades = get_trades()
    members = get_members()
    member_activity = get_member_activity()
    flagged_trades, flagged_problem = get_flagged_trades()
    predictions, predictions_problem = get_model_predictions()

    tabs = st.tabs([
        "Overview", "Member profile", "Committees vs. trading", "Flagged trades", "Model performance",
    ])

    with tabs[0]:
        render_overview(trades, member_activity, members)
    with tabs[1]:
        render_member_profile(trades, member_activity, members, flagged_trades, predictions)
    with tabs[2]:
        render_committee_analysis()
    with tabs[3]:
        render_flagged_trades(flagged_trades, flagged_problem, members)
    with tabs[4]:
        render_model_performance(predictions, predictions_problem)


main()
