SECTOR_COLUMNS = [
    "target_communication_services",
    "target_consumer_discretionary",
    "target_consumer_staples",
    "target_energy",
    "target_financials",
    "target_healthcare",
    "target_industrials",
    "target_information_technology",
    "target_materials",
    "target_real_estate",
    "target_utilities",
]

FEATURE_COLUMNS = [
    "trades_last_21d",
    "trades_last_90d",
    "trades_last_365d",
    "days_since_last_trade",
    "avg_trade_amount_90d",
    "purchase_ratio_90d",
    "sale_ratio_90d",

    "communication_services_trades_90d",
    "consumer_discretionary_trades_90d",
    "consumer_staples_trades_90d",
    "energy_trades_90d",
    "financials_trades_90d",
    "healthcare_trades_90d",
    "industrials_trades_90d",
    "information_technology_trades_90d",
    "materials_trades_90d",
    "real_estate_trades_90d",
    "utilities_trades_90d",

    "committee_communication_services",
    "committee_consumer_discretionary",
    "committee_consumer_staples",
    "committee_energy",
    "committee_financials",
    "committee_healthcare",
    "committee_industrials",
    "committee_information_technology",
    "committee_materials",
    "committee_real_estate",
    "committee_utilities",

    "tenure_years",
    "is_house",
    "is_senate",
]

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 30
THRESHOLD = 0.5