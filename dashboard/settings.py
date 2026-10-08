"""Constants for the PoliDetect dashboard: file paths, colors, and the
column "contracts" the dashboard expects from the other branches.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# Inputs that already exist on main (built by the src/ scripts)
# ---------------------------------------------------------------
TRADES_FILE = ROOT / "data/interim/transactions.parquet"
TICKER_SECTORS_FILE = ROOT / "data/interim/ticker_sectors.parquet"
LEGISLATORS_FILE = ROOT / "data/interim/legislators.parquet"
COMMITTEE_ASSIGNMENTS_FILE = ROOT / "data/interim/committee_assignments.parquet"
COMMITTEE_SECTORS_FILE = ROOT / "config/committee_sectors.csv"
MODEL_DATA_FILE = ROOT / "data/processed/model_data.parquet"

# ---------------------------------------------------------------
# Inputs produced by other branches (optional)
# ---------------------------------------------------------------
# The dashboard runs without these. When a file shows up, its tab
# turns on. See dashboard/README.md for the exact columns.

# From src/anomaly.py (merged in PR #4): one row per trade with a sector.
ANOMALY_SCORES_FILE = ROOT / "data/processed/anomaly_scores.parquet"
ANOMALY_SCORES_REQUIRED_COLUMNS = [
    "memberId", "td", "ticker", "sector",
    "score_history", "recent_sector_trades", "anomaly_score", "flag",
]

# Not produced yet: out-of-sample predictions, one row per member-window.
# The Model performance tab only appears once this file exists.
MODEL_PREDICTIONS_FILE = ROOT / "data/processed/model_predictions.parquet"
MODEL_PREDICTIONS_REQUIRED_COLUMNS = ["memberId", "prediction_date"]  # plus target_* and prob_*

# ---------------------------------------------------------------
# Display
# ---------------------------------------------------------------
# Same blue/orange as eda/descriptive_stats.ipynb so slides match.
PRIMARY_COLOR = "#2a78d6"     # main series
HIGHLIGHT_COLOR = "#eb6834"   # committee-overseen sectors, flagged items
MUTED_COLOR = "#8a8984"

# main.py now tunes a threshold per fold; 0.5 is just the slider's starting point.
DEFAULT_PREDICTION_THRESHOLD = 0.5
DEFAULT_FLAGS_TO_SHOW = 50
RECENT_TRADES_TO_SHOW = 25
