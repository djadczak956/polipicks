"""Print worked examples from data/processed/anomaly_scores.parquet.

Run src/anomaly.py first, then from the project root:
    python -m src.anomaly_examples
    python -m src.anomaly_examples --member Goldman --top 5
"""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd

from config.model_config import SECTORS, ROUTINE_MIN_TRADES, ROUTINE_WINDOW_DAYS


SCORES_FILE = (
    Path(__file__).resolve().parent.parent
    / "data/processed/anomaly_scores.parquet"
)


def label(sector):

    return sector.replace("_", " ").title()


def load_scores():

    scores = pd.read_parquet(SCORES_FILE)

    scores["td"] = pd.to_datetime(scores["td"])

    # p is never saved, but score_history = -log(p), so it can be recovered.
    scores["p_sector"] = np.exp(-scores["score_history"])

    return scores


def unique_trades(scores):

    # One filing can list the same ticker twice on a day; show it once.
    return scores.drop_duplicates(["memberId", "td", "ticker", "sourceDocId"])


def show_top_flags(scores, n):

    top = (
        unique_trades(scores)
        .sort_values("anomaly_score", ascending=False)
        .head(n)
    )

    print(f"\nTop {n} flagged trades (of {len(scores):,} scored)\n")

    print(
        top.assign(
            date=top["td"].dt.date,
            sector=top["sector"].map(label),
            p=top["p_sector"].map(lambda p: f"{p * 100:.2g}%"),
            score=top["anomaly_score"].round(1),
        )[["name", "party", "date", "ticker", "sector", "p", "score"]]
        .to_string(index=False)
    )


def show_walkthrough(scores, trade):

    history = scores[
        (scores["memberId"] == trade["memberId"])
        & (scores["td"] < trade["td"])
    ]

    counts = (
        history["sector"]
        .value_counts()
        .reindex(SECTORS, fill_value=0)
        .sort_values(ascending=False)
    )

    print(
        f"\nWalkthrough: {trade['name']} ({trade['party']}), "
        f"{trade['action']} {trade['ticker']} on {trade['td']:%Y-%m-%d}"
    )
    print(f"  Asset: {trade['assetDescription']}")
    print(f"  Sector: {label(trade['sector'])}")

    print(f"\n  Step 1. Their {len(history)} earlier trades by sector:")
    for sector, count in counts[counts > 0].items():
        marker = "  <- this trade" if sector == trade["sector"] else ""
        print(f"    {label(sector):<24} {count:>5}{marker}")
    if counts[trade["sector"]] == 0:
        print(f"    {label(trade['sector']):<24} {0:>5}  <- this trade")

    print(
        f"\n  Step 2. Smoothed probability of {label(trade['sector'])}: "
        f"{trade['p_sector']:.4%}"
    )
    print(f"  Step 3. Surprise = -log(p) = {trade['score_history']:.2f}")
    print(
        f"  Step 4. Trades in this sector in the past {ROUTINE_WINDOW_DAYS} days: "
        f"{trade['recent_sector_trades']} "
        f"({'routine, score set to 0' if trade['is_routine'] else f'under {ROUTINE_MIN_TRADES}, not routine'})"
    )
    print(
        f"  Final anomaly score: {trade['anomaly_score']:.2f} "
        f"(higher than {trade['anomaly_rank']:.1%} of all trades)"
    )

    committees = trade["committee_sectors"] or "none mapped"
    print(f"  Sectors their committees oversee: {committees}")
    print(f"  Filing: {trade['sourceUrl']}")


def show_routine_example(scores):

    # The trade the routine rule rescued that would otherwise rank highest.
    trade = (
        scores[scores["is_routine"]]
        .sort_values("score_history", ascending=False)
        .iloc[0]
    )

    raw_rank = (scores["score_history"] < trade["score_history"]).mean()

    print(
        f"\nRoutine rule example: {trade['name']}, {trade['ticker']} "
        f"({label(trade['sector'])}) on {trade['td']:%Y-%m-%d}"
    )
    print(
        f"  Only {trade['p_sector']:.2%} of their history, so surprise is "
        f"{trade['score_history']:.2f} (higher than {raw_rank:.1%} of trades)"
    )
    print(
        f"  But they made {trade['recent_sector_trades']} trades in this sector "
        f"in the past year, so the final score is 0"
    )


def show_model_comparison(scores):

    scored = unique_trades(scores).dropna(subset=["score_model"])

    disagree = scored.assign(
        gap=scored["score_model"] - scored["score_history"]
    ).sort_values("gap")

    print("\nHistory vs. neural net, trades where they disagree most\n")

    for title, rows in [
        ("Net finds it more surprising:", disagree.tail(3)),
        ("History finds it more surprising:", disagree.head(3)),
    ]:
        print(f"  {title}")
        for _, trade in rows.iterrows():
            print(
                f"    {trade['name']:<28} {trade['ticker']:<6} "
                f"{label(trade['sector']):<24} "
                f"history {trade['score_history']:5.2f}   net {trade['score_model']:5.2f}"
            )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--member",
        help="Part of a member's name; walks through their most unusual trade"
    )
    args = parser.parse_args()

    scores = load_scores()

    show_top_flags(scores, args.top)

    candidates = scores
    if args.member:
        candidates = scores[
            scores["name"].str.contains(args.member, case=False, na=False)
        ]
        if candidates.empty:
            raise SystemExit(f"No member matching '{args.member}'")

    show_walkthrough(
        scores,
        candidates.sort_values("anomaly_score", ascending=False).iloc[0]
    )

    show_routine_example(scores)

    show_model_comparison(scores)


if __name__ == "__main__":
    main()
