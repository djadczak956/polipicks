# PoliPicks

Predicts which market sectors a member of Congress will trade in next quarter, and flags stock-trade disclosures that don't fit the pattern.

CS4342 (Machine Learning) project by team PoliDetect: Vincent Grassi, Krish Patel, Andrew Melton, Damian Jadczak, Antoine Pham.

## Motivation

Under the STOCK Act, members of Congress must disclose stock trades over $1,000 within 45 days. The filings are split across two portals as one-off documents, and many are scanned PDFs. That makes it hard to ask whether a member's trading tracks the jurisdiction of their committees, or to find the filings that break a member's own pattern.

PoliPicks is meant for:

- **Journalists and accountability groups**: a screening tool that pulls a handful of unusual filings out of thousands of routine ones.
- **Researchers**: a quantified answer to whether committee jurisdiction predicts trading behavior.
- **The public**: a readable trading profile for any representative.

## Data

| Source | Contents | Role |
|---|---|---|
| [congressional-stock-trades](https://huggingface.co/datasets/austin-starks/congressional-stock-trades) | House and Senate Periodic Transaction Reports already extracted from the official filings (electronic and scanned), with bioguide IDs, tickers, amount ranges and amendment history; refreshed about every 20 hours | Transactions, 2021–2026 (about 24 quarters) |
| [Congress.gov API](https://api.congress.gov/) | Members of the 117th–119th Congresses with term history | Tenure |
| [congress-legislators](https://github.com/unitedstates/congress-legislators) | Committee membership, recovered per Congress from the repo's git history | Committee features |
| `config/committee_sectors.csv` (hand-written) | House committees → GICS sectors | Committee–sector mapping |
| yfinance + `config/ticker_sector_overrides.csv` | Sector per ticker, cached; overrides cover delisted/renamed companies and mark funds | Ticker → sector |

All sources join on bioguide ID. Labels come from the filings themselves: the set of sectors a member traded in a given quarter.

We use the dataset instead of parsing the House Clerk PDFs ourselves. We checked a random sample of 50 filings against the Clerk's PDFs: every source PDF matched the dataset's SHA-256, and every row in the 40 electronic filings matched an independent parse. Rows from scanned filings are model-read and can misread the day of the month, which doesn't affect quarterly labels. The data is restricted to non-commercial use under 5 U.S.C. 13107(c) (see the dataset's `LICENSE_DATA.md`).

To get it:

```bash
git clone https://huggingface.co/datasets/austin-starks/congressional-stock-trades
```

## Pipeline

Setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "CONGRESS_API_KEY=<your key>" > .env   # free key from api.congress.gov; never commit it
make data    # builds data/processed/model_data.parquet (~1 min with the sector cache)
make train       # neural net: per-fold and mean precision / recall / F1 / macro F1
make baselines   # same folds: repeat-last-90d, logistic regression, gradient boosting (~4 min)
```

Hyperparameters live in `config/model_config.py`. Each fold picks its decision threshold on the last year of its training period, then is scored once on its test period. Compare runs by the `mean f1` line; accuracy is not reported because predicting "no trade" everywhere is ~96% accurate.

`make data` runs these in order; each writes one file:

| Script | Output | What it does |
|---|---|---|
| `src/load_trades.py` | `data/interim/transactions.parquet` | Stacks the six yearly parquets in `data/raw/`; drops amended-away rows, Senate, rows without a ticker, and trades outside 2021-01-01–2026-09-30. Asserts 58,162 trades, 201 members, 2,643 tickers. |
| `src/load_legislators.py` | `data/interim/legislators.parquet` | Members and first House term from Congress.gov. |
| `src/load_committee_snapshots.py` | `data/interim/committee_assignments.parquet` | One committee snapshot per Congress from the legislators repo's git history (needs a full clone; `make` clones it). |
| `src/load_ticker_sectors.py` | `data/interim/ticker_sectors.parquet` | yfinance sector per ticker, then the overrides file. |
| `src/build_model_data.py` | `data/processed/model_data.parquet` | All joins. One row per member per 21-day prediction date. |

Decisions worth knowing:

- **Features only use disclosed trades.** History features count trades whose `available_at` (when the filing became public) is before the prediction date. The median trade is disclosed 28 days after it happens, longer than the 21-day window, so using trade dates would leak. Targets are the sectors traded in `[prediction_date, prediction_date + 21 days)`. `verify_no_leakage` recomputes every row independently and fails the build on any mismatch.
- **No incomplete targets at the end.** Prediction dates stop 45 days (the STOCK Act deadline) plus one window before the data was pulled, since recent trades aren't all disclosed yet.
- **Committee snapshot dates.** 117th: 2021-03-15 (House assignments were added to the repo on 2021-03-01). 118th: 2023-03-01. 119th: 2025-04-01 (the China select committee was added on 2025-03-13). Each Congress uses one snapshot, so mid-Congress committee changes aren't reflected.
- **Committee mapping.** Committees without a clear sector (Appropriations, Budget, Rules, Ethics, Judiciary, Oversight, House Administration, Foreign Affairs, Homeland Security, Intelligence, Small Business, Education and Workforce, most select committees) set no jurisdiction bits. No committee maps to Consumer Discretionary yet.
- **Ticker sectors.** Yahoo has no data for delisted or renamed companies (FB, ATVI, SIVB, ...). `config/ticker_sector_overrides.csv` assigns those by hand and marks funds/ETFs as `fund` with no sector. 97% of trades have a sector, 1.6% are funds, 1.3% are unresolved.
- **Data is committed.** `data/raw`, `data/interim` and `data/processed` are small (~2 MB) and checked in so everyone trains on the same files.

## Modeling

### Sector prediction (multi-label classification)

- One row per member-quarter, 11 GICS sectors as labels.
- Primary model: a small feedforward network with 11 sigmoid outputs and binary cross-entropy weighted by inverse label frequency (Information Technology is far more common than Utilities, for example).
- Benchmarks: logistic regression and XGBoost (scikit-learn, XGBoost). We expect gradient-boosted trees to score highest on a few thousand rows of mostly categorical features, and we'll report the comparison as it comes out.
- Baseline: predict each member's historical sector mix.
- Metrics: per-label F1, Hamming loss, subset accuracy.
- Chronological train/test splits, since the task is forecasting.

### Anomaly detection

A filing is flagged when either:

- the classifier assigns low probability to the sector actually traded, or
- the trade diverges from the member's historical sector distribution (z-score or KL divergence via `scipy.stats`).

Flags above a configurable threshold are shown along with the features behind them.

## Interface

Streamlit, or a FastAPI + React app.
