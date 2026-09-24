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
| [House Clerk financial disclosures](https://disclosures-clerk.house.gov/FinancialDisclosure) | One ZIP per filing year: XML index plus filing documents | Training history, 2021–2026 (about 24 quarters) |
| Tracefour API | Keyless REST API over STOCK Act disclosures, refreshed hourly, rolling six-month window | Recent filings, without PDF parsing |
| [congress-legislators](https://github.com/unitedstates/congress-legislators) | Committee assignments, party, state, chamber, tenure | Member features |
| Hand-written CSV | ~20 House committees → GICS sectors | Committee–sector mapping |
| yfinance | Sector per ticker, fetched once and cached | Ticker → sector |

All sources join on bioguide ID. Labels come from the filings themselves: the set of sectors a member traded in a given quarter.

## Pipeline

1. Download the annual ZIPs and parse the XML index into a filing table.
2. Keep Periodic Transaction Reports and parse them into a transaction table (pdfplumber).
3. Resolve members against the legislators file.
4. Map tickers to sectors.
5. Aggregate to one row per member-quarter. Features come strictly from before the label quarter.

Slow steps are cached so they run once. Storage and processing use pandas and DuckDB.

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
