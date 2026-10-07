"""Repeat the injection test with different random swaps and average how well the
F/U flag (anomaly score above ANOMALY_FLAG_SCORE) separates swapped trades from untouched ones.

From the project root:
    python -m src.injection_test
    python -m src.injection_test --runs 20
"""

import argparse

import numpy as np
import pandas as pd

from config.model_config import SECTORS, ANOMALY_TEST_START, ANOMALY_FLAG_SCORE
from src.anomaly import (
    load_trades,
    house_probs,
    history_probs,
    recent_counts,
    routine_mask,
    surprise,
    inject,
    tune_alpha
)


def run(trades, rows, p_history, recent, seed):

    swapped, injected = inject(trades, rows, np.random.default_rng(seed), recent)

    score = np.where(
        routine_mask(recent, swapped),
        0.0,
        surprise(p_history, swapped)
    )[rows]

    flagged = score > ANOMALY_FLAG_SCORE
    injected = injected[rows]

    return {
        "seed": seed,
        "swapped": injected.sum(),
        "accuracy": (flagged == injected).mean(),
        "catch_rate": flagged[injected].mean(),
        "false_alarm_rate": flagged[~injected].mean(),
        "precision": injected[flagged].mean(),
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    trades = load_trades()

    house = house_probs(trades)
    alpha = tune_alpha(trades, house)

    p_history = history_probs(trades, alpha, house)
    recent = recent_counts(trades)

    rows = np.flatnonzero((trades["td"] >= ANOMALY_TEST_START).to_numpy())

    results = pd.DataFrame([
        run(trades, rows, p_history, recent, seed)
        for seed in range(args.runs)
    ])

    print(
        f"\n{args.runs} injection runs on {len(rows)} trades from {ANOMALY_TEST_START}, "
        f"flag = score above {ANOMALY_FLAG_SCORE}\n"
    )
    print(results.to_string(index=False, float_format="{:.3f}".format))

    metrics = results.drop(columns=["seed", "swapped"])

    print("\nMean (std) across runs:")
    for name in metrics:
        print(f"  {name:<17} {metrics[name].mean():.3f} ({metrics[name].std():.3f})")


if __name__ == "__main__":
    main()
