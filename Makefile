# Run with the project venv active: source .venv/bin/activate
# `make data` rebuilds everything the model reads; `make train` trains on it.

PY ?= python
LEGISLATORS_REPO = data/external/congress-legislators

.PHONY: data train baselines anomalies committee-analysis dashboard trades legislators committees sectors model_data clean

data: model_data

trades:
	$(PY) src/load_trades.py

# Needs CONGRESS_API_KEY in .env (free key from api.congress.gov).
legislators:
	$(PY) src/load_legislators.py

# Needs a FULL clone of the legislators repo (snapshots come from git history).
$(LEGISLATORS_REPO):
	git clone https://github.com/unitedstates/congress-legislators $(LEGISLATORS_REPO)

committees: $(LEGISLATORS_REPO)
	$(PY) src/load_committee_snapshots.py

# Slow on a cold cache (~15 min of yfinance lookups); resumes from the saved file.
sectors: trades
	$(PY) src/load_ticker_sectors.py

model_data: trades legislators committees sectors
	$(PY) -m src.build_model_data

train:
	$(PY) main.py

# Same folds and threshold tuning as `train`; compare its mean f1 to these.
baselines:
	$(PY) -m src.baselines

# Scores every trade; writes data/processed/anomaly_scores.parquet.
anomalies:
	$(PY) -m src.anomaly

# Committee seat vs. sectors traded, each member weighted equally.
committee-analysis:
	$(PY) -m src.committee_analysis

# Needs train + anomalies first so its tabs have data.
dashboard:
	streamlit run dashboard/app.py

# Keeps ticker_sectors.parquet: it is a slow-to-rebuild cache.
clean:
	rm -f data/interim/transactions.parquet data/interim/legislators.parquet \
	      data/interim/committee_assignments.parquet data/processed/model_data.parquet
