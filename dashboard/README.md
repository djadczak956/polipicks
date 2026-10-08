# PoliPicks dashboard

Streamlit app for the three audiences in the main README: the public (member profiles), researchers (committees vs. trading), and journalists (flagged trades).

## Run it

From the project root, after `make data` and `python -m src.anomaly`:

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

## Tabs and where their data comes from

| Tab | Reads | Shows |
|---|---|---|
| Overview | `data/interim/transactions.parquet`, `ticker_sectors.parquet` | Dataset size, trades per quarter, sector shares, most active traders |
| Member profile | the above + `legislators.parquet`, `committee_assignments.parquet`, `config/committee_sectors.csv`, `anomaly_scores.parquet` | One member's committees, sector mix (committee-overseen sectors in orange), trades per quarter, their flagged trades, recent filings with PDF links |
| Committees vs. trading | `data/processed/model_data.parquet` | For each sector: how often members on an overseeing committee trade it vs. everyone else |
| Flagged trades | `data/processed/anomaly_scores.parquet` (from `src/anomaly.py`) | Trades flagged by the anomaly detector, most unusual first, with the reason and a link to the filing |
| Model performance | `data/processed/model_predictions.parquet` | Only appears once that file exists (see below) |

## Flagged trades

`src/anomaly.py` writes one row per trade with a known sector. The dashboard uses these columns:

| Column | Meaning |
|---|---|
| `score_history` | −log(p), where p is the member's share of earlier trades in this sector, blended with the House-wide mix |
| `recent_sector_trades` | Trades by the member in this sector in the 12 months before |
| `anomaly_score` | `score_history`, set to 0 when the sector is routine (`ROUTINE_MIN_TRADES` or more recent trades) |
| `flag` | `"F"` when `anomaly_score > ANOMALY_FLAG_SCORE`, else `"U"` |
| `committee_sectors` | Sectors the member's committees oversaw in that Congress |

The flag threshold and routine rule are read from `config/model_config.py`, so changing them there and rerunning `python -m src.anomaly` updates the dashboard.

## Model performance (not wired up yet)

`main.py` prints metrics but doesn't save predictions. To turn this tab on, save each fold's test rows to `data/processed/model_predictions.parquet` with `memberId`, `prediction_date`, `target_<sector>` (0/1) and `prob_<sector>` (sigmoid output) for all 11 sectors.
