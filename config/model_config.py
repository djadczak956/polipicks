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

# Defaults chosen by a 48-combination sweep (tuned on the last training
# year of each fold, not on the test years). Mean test micro F1 ~0.50.
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
EPOCHS = 30

# Loss weight on positive labels per sector: "none", "sqrt" or "full".
POS_WEIGHT = "sqrt"

# The decision threshold is chosen per fold: train on all but the last
# TUNING_DAYS of the training period, pick the threshold with the best F1
# there, then retrain on the full training period and score the test year.
TUNING_DAYS = 365
THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 19)]

SEED = 0
