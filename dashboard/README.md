# PoliPicks dashboard

Streamlit app for the three audiences in the main README: the public (member profiles), researchers (committees vs. trading), and journalists (flagged trades).

## Run it

From the project root, after the `src/` pipeline has produced `data/interim/*` and `data/processed/model_data.parquet`:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Files

| File | What it does |
|---|---|
| `app.py` | Page layout. One `render_<tab>()` function per tab, all called from `main()` at the bottom. |
| `data_loader.py` | Loads and reshapes data with plain pandas (no Streamlit), so each function also works in a notebook. |
| `settings.py` | File paths, colors, defaults. |

## Tabs

| Tab | Needs | Shows |
|---|---|---|
| Overview | main pipeline only | Dataset size, trades per quarter, sector shares, most active traders |
| Member profile | main pipeline only | One member's committees, sector mix (committee-overseen sectors in orange), trades per quarter, recent filings with PDF links |
| Committees vs. trading | `model_data.parquet` | For each sector: how often members on an overseeing committee trade it vs. everyone else |
| Flagged trades | `flagged_trades.parquet` (anomaly branch) | Most unusual trades first, filterable by party and sector |
| Model performance | `model_predictions.parquet` (model tuning) | Per-sector precision / recall / F1 / AUC, micro and macro F1, Hamming loss, subset accuracy, with a threshold slider |

The last two tabs show setup instructions until their file exists, so the dashboard runs on `main` today.

## What the other branches need to write

### `data/processed/flagged_trades.parquet` (anomaly detection)

One row per scored trade.

| Column | Required | Meaning |
|---|---|---|
| `memberId` | yes | Bioguide ID |
| `td` | yes | Trade date |
| `ticker` | yes | |
| `sector` | yes | Snake-case sector, same names as `SECTORS` in `config/model_config.py` |
| `anomaly_score` | yes | Higher = more unusual. The table sorts by this. |
| `is_flagged` | no | True/False. Enables the "only flagged" toggle. |
| `flag_reasons` | no | Plain-English reason, shown in the table |
| `assetDescription`, `amountLow`, `amountHigh`, `sourceUrl` | no | Copied from `transactions.parquet`; shown if present |

### `data/processed/model_predictions.parquet` (model tuning)

One row per member-window, **validation rows only** (each walk-forward fold's out-of-sample predictions stacked together).

| Column | Meaning |
|---|---|
| `memberId`, `prediction_date` | Same as `model_data.parquet` |
| `target_<sector>` | 0/1, what actually happened (11 columns) |
| `prob_<sector>` | `torch.sigmoid(logits)` for that sector (11 columns) |

In `main.py` this means saving `val_df[["memberId", "prediction_date"] + TARGET_COLUMNS]` next to the fold's probabilities and writing all folds at the end.
