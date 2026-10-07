"""Does sitting on a committee change which sectors a member trades?

Uses data/processed/model_data.parquet, keeping only windows where the
member traded something (so this asks "given they traded, which sectors"),
and counts every member equally so the two heaviest traders (~55% of all
trades) don't decide the answer.

Two comparisons per sector:
  between members: average rate for member-Congresses ON an overseeing
                   committee vs OFF one (each member-Congress weighted equally)
  within member:   members who were ON in one Congress and OFF in another;
                   mean of (rate when on - rate when off). Controls for each
                   person's own habits.

A bootstrap over members gives a 95% interval for the between-member lift.

Run: python -m src.committee_analysis   (or: make committees-analysis)
Writes data/processed/committee_effect.csv
"""

import numpy as np
import pandas as pd

from config.model_config import SECTORS, DATE_COLUMN
from src.build_model_data import ROOT, congress_from_date


MODEL_DATA_FILE = ROOT / "data/processed/model_data.parquet"
OUT = ROOT / "data/processed/committee_effect.csv"

N_BOOTSTRAP = 1000
SEED = 0


def member_congress_rates(model_df):

    targets = [f"target_{s}" for s in SECTORS]

    active = model_df[model_df[targets].sum(axis=1) > 0].copy()

    active["congress"] = (
        pd.to_datetime(active[DATE_COLUMN]).map(congress_from_date)
    )

    # One row per member and Congress: share of their active windows with a
    # trade in each sector, and whether they sat on an overseeing committee.
    return active.groupby(["memberId", "congress"]).agg(
        windows=(DATE_COLUMN, "size"),
        **{f"rate_{s}": (f"target_{s}", "mean") for s in SECTORS},
        **{f"on_{s}": (f"committee_{s}", "max") for s in SECTORS},
    ).reset_index()


def between_lift(rates, sector):

    on = rates[f"on_{sector}"] == 1
    rate_on = rates.loc[on, f"rate_{sector}"].mean()
    rate_off = rates.loc[~on, f"rate_{sector}"].mean()

    return rate_on, rate_off, rate_on / rate_off


def within_member(rates, sector):

    by_member = rates.groupby("memberId")

    diffs = []

    for _, group in by_member:

        on = group[f"on_{sector}"] == 1

        if on.any() and (~on).any():
            diffs.append(
                group.loc[on, f"rate_{sector}"].mean()
                - group.loc[~on, f"rate_{sector}"].mean()
            )

    return len(diffs), (np.mean(diffs) if diffs else np.nan)


def bootstrap_lift(rates, sector, rng):

    members = rates["memberId"].unique()
    grouped = {m: g for m, g in rates.groupby("memberId")}

    lifts = []

    for _ in range(N_BOOTSTRAP):

        sample = pd.concat(
            [grouped[m] for m in rng.choice(members, len(members))]
        )

        on = sample[f"on_{sector}"] == 1

        if on.sum() == 0 or (~on).sum() == 0:
            continue

        off_rate = sample.loc[~on, f"rate_{sector}"].mean()

        if off_rate > 0:
            lifts.append(
                sample.loc[on, f"rate_{sector}"].mean() / off_rate
            )

    return np.percentile(lifts, [2.5, 97.5])


def main():

    model_df = pd.read_parquet(MODEL_DATA_FILE)

    rates = member_congress_rates(model_df)

    rng = np.random.default_rng(SEED)

    rows = []

    for sector in SECTORS:

        n_on = int((rates[f"on_{sector}"] == 1).sum())

        if n_on == 0:
            rows.append({"sector": sector, "member_congresses_on": 0})
            continue

        rate_on, rate_off, lift = between_lift(rates, sector)
        low, high = bootstrap_lift(rates, sector, rng)
        n_switch, within = within_member(rates, sector)

        rows.append({
            "sector": sector,
            "member_congresses_on": n_on,
            "rate_on": rate_on,
            "rate_off": rate_off,
            "lift": lift,
            "lift_ci_low": low,
            "lift_ci_high": high,
            "switchers": n_switch,
            "within_member_diff": within,
        })

    result = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print(
        f"{len(rates)} member-Congresses with at least one trade, "
        f"{rates.memberId.nunique()} members\n"
    )

    with pd.option_context("display.width", 200):
        print(result.round(3).to_string(index=False))

    print(
        "\nlift < 1: members on an overseeing committee trade that sector "
        "LESS often. An interval that contains 1 means no clear effect."
    )
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
