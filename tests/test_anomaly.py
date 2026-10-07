import unittest

import pandas as pd

from config.model_config import DATE_COLUMN
from src.anomaly import _match_prediction_windows


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


if __name__ == "__main__":
    unittest.main()