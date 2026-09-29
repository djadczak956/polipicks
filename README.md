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
| [congress-legislators](https://github.com/unitedstates/congress-legislators) | Committee assignments, party, state, chamber, tenure | Member features |
| Hand-written CSV | ~20 House committees → GICS sectors | Committee–sector mapping |
| yfinance | Sector per ticker, fetched once and cached | Ticker → sector |

All sources join on bioguide ID. Labels come from the filings themselves: the set of sectors a member traded in a given quarter.

We use the dataset instead of parsing the House Clerk PDFs ourselves. We checked a random sample of 50 filings against the Clerk's PDFs: every source PDF matched the dataset's SHA-256, and every row in the 40 electronic filings matched an independent parse. Rows from scanned filings are model-read and can misread the day of the month, which doesn't affect quarterly labels. The data is restricted to non-commercial use under 5 U.S.C. 13107(c) (see the dataset's `LICENSE_DATA.md`).

To get it:

```bash
git clone https://huggingface.co/datasets/austin-starks/congressional-stock-trades
```

## Pipeline

1. Load `political_trade_events` (one row per trade, with repeated reports merged) and keep House members.
2. Map tickers to sectors. Scanned filings have no asset-type code and sometimes no ticker, so those rows are matched on asset name.
3. Aggregate to one row per member-quarter. Features use only trades with `availableAt` before the label quarter.
4. Weight or cap per member where row counts matter: two members account for over half of all House rows.

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
