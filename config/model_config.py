SECTORS = [
    "communication_services",
    "consumer_discretionary",
    "consumer_staples",
    "energy",
    "financials",
    "health_care",
    "industrials",
    "information_technology",
    "materials",
    "real_estate",
    "utilities",
]


TARGET_COLUMNS = [
    f"target_{sector}"
    for sector in SECTORS
]


FEATURE_COLUMNS = [
    # Overall trading behavior
    "trades_last_21d",
    "trades_last_90d",
    "trades_last_365d",
    "days_since_last_trade",
    "avg_amount_90d",
    "purchase_ratio_90d",
    "sale_ratio_90d",

    # Legislator information
    "tenure_years",
]


FEATURE_COLUMNS += [
    f"{sector}_trades_90d"
    for sector in SECTORS
]


FEATURE_COLUMNS += [
    f"committee_{sector}"
    for sector in SECTORS
]


DATE_COLUMN = "prediction_date"

PREDICTION_WINDOW_DAYS = 21

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 30
THRESHOLD = 0.5
ANOMALY_ALPHAS = [1.0, 2.0, 5.0, 10.0, 20.0]
ANOMALY_TEST_START = "2023-01-01"
# Fixed by definition, not tuned: the injection test can't judge what counts as routine.
ROUTINE_WINDOW_DAYS = 365
ROUTINE_MIN_TRADES = 3
