import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from config.model_config import (
    DATE_COLUMN,
    ROUTINE_MIN_TRADES,
    SECTORS,
)
from src.anomaly import (
    _match_prediction_windows,
    counts_before,
    evaluate,
    inject,
    routine_mask,
)
from src.model import ResidualPoliPickModel


class MatchPredictionWindowsTests(unittest.TestCase):

    def test_window_start_is_included_and_next_start_is_excluded(self):

        first_start = pd.Timestamp("2023-01-01")
        second_start = pd.Timestamp("2023-01-22")
        trades = pd.DataFrame({
            "td": [
                first_start,
                first_start + pd.Timedelta(days=20),
                second_start,
            ]
        })
        window_starts = pd.DataFrame({
            DATE_COLUMN: [first_start, second_start]
        })

        windows = _match_prediction_windows(trades, window_starts)

        self.assertEqual(
            windows[DATE_COLUMN].tolist(),
            [first_start, first_start, second_start],
        )


class AnomalyPipelineTests(unittest.TestCase):

    def test_history_counts_exclude_same_day_trades_and_other_members(self):

        trades = pd.DataFrame({
            "memberId": ["A", "A", "A", "B"],
            "td": pd.to_datetime([
                "2023-01-01",
                "2023-01-01",
                "2023-01-02",
                "2023-01-02",
            ]),
            "sector": [SECTORS[0], SECTORS[1], SECTORS[0], SECTORS[2]],
        })

        counts = counts_before(trades, ["memberId"])

        np.testing.assert_array_equal(counts[0], np.zeros(len(SECTORS)))
        np.testing.assert_array_equal(counts[1], np.zeros(len(SECTORS)))
        self.assertEqual(counts[2, SECTORS.index(SECTORS[0])], 1)
        self.assertEqual(counts[2, SECTORS.index(SECTORS[1])], 1)
        np.testing.assert_array_equal(counts[3], np.zeros(len(SECTORS)))

    def test_evaluation_injections_are_independent_of_routine_sectors(self):

        sectors = [sector for sector in SECTORS for _ in range(20)]
        trades = pd.DataFrame({
            "memberId": "member-1",
            "td": pd.Timestamp("2023-02-01"),
            "sector": sectors,
        })
        rows = np.arange(len(trades))
        recent = np.full(
            (len(trades), len(SECTORS)),
            ROUTINE_MIN_TRADES,
        )
        p_history = np.full(
            (len(trades), len(SECTORS)),
            1 / len(SECTORS),
        )
        p_model = np.full_like(p_history, np.nan)
        original_inject = inject
        captured = {}

        def capture_injections(*args, **kwargs):
            result = original_inject(*args, **kwargs)
            captured["swapped"], captured["injected"] = result
            return result

        with (
            patch("src.anomaly.inject", side_effect=capture_injections) as inject_spy,
            patch("src.anomaly.auc", return_value=(0.5, 0.5)),
        ):
            evaluate(trades, p_history, p_model, recent)

        self.assertEqual(len(inject_spy.call_args.args), 3)
        self.assertNotIn("recent", inject_spy.call_args.kwargs)
        injected = captured["injected"]
        swapped = captured["swapped"]
        self.assertTrue(injected.any())
        self.assertTrue(routine_mask(recent, swapped)[injected].all())


class ModelArchitectureTests(unittest.TestCase):

    def test_residual_model_returns_one_logit_per_sector(self):

        model = ResidualPoliPickModel(input_size=30, num_sectors=11)
        logits = model(torch.zeros(4, 30))

        self.assertEqual(tuple(logits.shape), (4, 11))


if __name__ == "__main__":
    unittest.main()