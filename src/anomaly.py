import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from config.model_config import (
    SECTORS,
    FEATURE_COLUMNS,
    DATE_COLUMN,
    ANOMALY_ALPHAS,
    ANOMALY_TEST_START,
    ROUTINE_WINDOW_DAYS,
    ROUTINE_MIN_TRADES,
    ANOMALY_FLAG_SCORE
)

from src.build_model_data import (
    ROOT,
    TRADES_FILE,
    TICKER_SECTORS_FILE,
    LEGISLATORS_FILE,
    normalize_sector,
    congress_from_date,
    build_committee_features
)

from src.splits import walk_forward_splits
from src.train import fit_fold, get_device


MODEL_DATA_FILE = (
    ROOT / "data/processed/model_data.parquet"
)

OUT = (
    ROOT / "data/processed/anomaly_scores.parquet"
)

EPS = 1e-6


def load_trades():

    trades = pd.read_parquet(TRADES_FILE)

    ticker_sectors = pd.read_parquet(
        TICKER_SECTORS_FILE
    )

    trades = trades.merge(
        ticker_sectors[["ticker", "sector"]],
        on="ticker",
        how="left"
    )

    trades["td"] = pd.to_datetime(trades["td"]).astype("datetime64[ns]")

    trades["sector"] = (
        trades["sector"]
        .apply(normalize_sector)
    )

    known = trades["sector"].isin(SECTORS)

    print(
        f"Dropping {(~known).sum()} of {len(trades)} "
        f"trades with no known sector"
    )

    return (
        trades[known]
        .sort_values(["td", "memberId"])
        .reset_index(drop=True)
    )


def daily_counts(trades, keys):

    onehot = pd.get_dummies(
        trades["sector"]
    ).reindex(columns=SECTORS, fill_value=0).astype(float)

    daily = (
        pd.concat([trades[keys + ["td"]], onehot], axis=1)
        .groupby(keys + ["td"])[SECTORS]
        .sum()
    )

    if keys:
        cumulative = daily.groupby(level=keys).cumsum()
    else:
        cumulative = daily.cumsum()

    return daily, cumulative


def counts_before(trades, keys):

    # Counts strictly before each trade's date, so same-day trades don't see each other.
    daily, cumulative = daily_counts(trades, keys)

    before = (cumulative - daily).reset_index()

    return (
        trades[keys + ["td"]]
        .merge(before, on=keys + ["td"], how="left")[SECTORS]
        .to_numpy()
    )


def recent_counts(trades, days=ROUTINE_WINDOW_DAYS):

    _, cumulative = daily_counts(trades, ["memberId"])

    window_starts = (
        trades[["memberId", "td"]]
        .assign(start=trades["td"] - pd.Timedelta(days=days))
        .reset_index()
        .sort_values("start")
    )

    before_window = (
        pd.merge_asof(
            window_starts,
            cumulative.reset_index().sort_values("td"),
            left_on="start",
            right_on="td",
            by="memberId",
            allow_exact_matches=False
        )
        .sort_values("index")[SECTORS]
        .fillna(0)
        .to_numpy()
    )

    return counts_before(trades, ["memberId"]) - before_window


def routine_mask(recent, sector_index):

    return recent[np.arange(len(recent)), sector_index] >= ROUTINE_MIN_TRADES


def house_probs(trades):

    _, cumulative = daily_counts(trades, ["memberId"])

    members = trades["memberId"].unique()

    grid = (
        cumulative
        .unstack("memberId")
        .reindex(columns=pd.MultiIndex.from_product([SECTORS, members]))
        .ffill()
        .fillna(0)
    )

    counts = grid.to_numpy().reshape(len(grid), len(SECTORS), len(members))
    totals = counts.sum(axis=1)

    shares = np.divide(
        counts,
        totals[:, None, :],
        out=np.zeros_like(counts),
        where=totals[:, None, :] > 0
    )

    # Mean of member shares so far, so each member counts equally and heavy traders don't define "normal".
    through_date = (
        (shares.sum(axis=2) + 1.0)
        / ((totals > 0).sum(axis=1) + len(SECTORS))[:, None]
    )

    before_date = np.vstack([
        np.full((1, len(SECTORS)), 1.0 / len(SECTORS)),
        through_date[:-1]
    ])

    return before_date[grid.index.get_indexer(trades["td"])]


def history_probs(trades, alpha, house=None):

    member_counts = counts_before(
        trades,
        ["memberId"]
    )

    if house is None:
        house = house_probs(trades)

    return (
        (member_counts + alpha * house)
        / (member_counts.sum(axis=1, keepdims=True) + alpha)
    )


def _match_prediction_windows(trades, window_starts):

    return pd.merge_asof(
        trades[["td"]].reset_index(),
        window_starts,
        left_on="td",
        right_on=DATE_COLUMN,
        allow_exact_matches=True
    )


def model_probs(trades, model_df):

    torch.manual_seed(0)

    model_df = model_df.copy()
    model_df[DATE_COLUMN] = pd.to_datetime(model_df[DATE_COLUMN]).astype("datetime64[ns]")

    device = get_device()

    fold_probs = []

    for fold in walk_forward_splits(model_df):

        if fold["train"].empty or fold["validation"].empty:
            continue

        model, scaler = fit_fold(
            fold["train"],
            device
        )

        val_df = fold["validation"]

        features = torch.tensor(
            scaler.transform(val_df[FEATURE_COLUMNS]),
            dtype=torch.float32
        ).to(device)

        model.eval()

        with torch.no_grad():
            probs = torch.sigmoid(model(features)).cpu().numpy()

        fold_probs.append(
            pd.concat(
                [
                    val_df[["memberId", DATE_COLUMN]].reset_index(drop=True),
                    pd.DataFrame(probs, columns=SECTORS)
                ],
                axis=1
            )
        )

    fold_probs = pd.concat(fold_probs)

    # Targets cover [d, d + 21), so a trade on d uses the prediction dated d.
    window_starts = pd.DataFrame({
        DATE_COLUMN: np.sort(pd.to_datetime(model_df[DATE_COLUMN].unique()))
    })

    windows = _match_prediction_windows(trades, window_starts)

    joined = (
        windows
        .assign(memberId=trades["memberId"].to_numpy())
        .merge(fold_probs, on=["memberId", DATE_COLUMN], how="left")
        .sort_values("index")
    )

    probs = joined[SECTORS].to_numpy()

    # The model predicts "trades in sector s at all this window"; normalizing turns that into "this trade's sector is s".
    return probs / probs.sum(axis=1, keepdims=True)


def surprise(probs, sector_index):

    picked = probs[np.arange(len(probs)), sector_index]

    return -np.log(np.clip(picked, EPS, 1.0))


def inject(trades, rows, rng, recent=None, rate=0.05):

    sector_index = trades["sector"].map(SECTORS.index).to_numpy()

    house_mix = (
        trades["sector"]
        .value_counts(normalize=True)
        .reindex(SECTORS, fill_value=0)
        .to_numpy()
    )

    swapped = sector_index.copy()
    injected = np.zeros(len(trades), dtype=bool)

    for i in rows[rng.random(len(rows)) < rate]:

        mix = house_mix.copy()
        mix[sector_index[i]] = 0

        # Moving a trade into one of the member's routine sectors wouldn't make it anomalous.
        if recent is not None:
            mix[recent[i] >= ROUTINE_MIN_TRADES] = 0

        if mix.sum() == 0:
            continue

        swapped[i] = rng.choice(len(SECTORS), p=mix / mix.sum())
        injected[i] = True

    return swapped, injected


def auc(trades, rows, injected, score):

    # Heavy traders (one member is ~half the trades) would otherwise set the result.
    member_weights = (
        1.0 / trades.groupby("memberId")["td"].transform("size")
    ).to_numpy()[rows]

    return (
        roc_auc_score(injected[rows], score),
        roc_auc_score(injected[rows], score, sample_weight=member_weights)
    )


def tune_alpha(trades, house):

    rows = np.flatnonzero(trades["td"] < ANOMALY_TEST_START)

    swapped, injected = inject(trades, rows, np.random.default_rng(0))

    print(f"\nTuning alpha on trades before {ANOMALY_TEST_START} ({injected.sum()} injected)")

    results = {}

    for alpha in ANOMALY_ALPHAS:

        p_history = history_probs(trades, alpha, house)

        _, results[alpha] = auc(
            trades,
            rows,
            injected,
            surprise(p_history[rows], swapped[rows])
        )

        print(f"  alpha {alpha:>4}: per-member AUC {results[alpha]:.3f}")

    best = max(results, key=results.get)

    print(f"  using alpha {best}")

    return best


def evaluate(trades, p_history, p_model, recent):

    rows = np.flatnonzero((trades["td"] >= ANOMALY_TEST_START).to_numpy())

    swapped, injected = inject(
        trades,
        rows,
        np.random.default_rng(1),
        recent
    )

    score_history = surprise(p_history, swapped)

    # The model only scores trades inside its walk-forward folds, so it's compared on that subset.
    model_rows = rows[~np.isnan(p_model[rows]).any(axis=1)]

    candidates = {
        "history": (rows, score_history[rows]),
        "routine": (
            rows,
            np.where(routine_mask(recent, swapped), 0.0, score_history)[rows]
        ),
        "model": (model_rows, surprise(p_model[model_rows], swapped[model_rows])),
    }

    print(
        f"\nTest on trades from {ANOMALY_TEST_START}: {injected.sum()} of {len(rows)} "
        f"moved into a sector the member doesn't trade routinely"
    )

    print(f"  {'AUC':<9} {'pooled':>7} {'per member':>11} {'trades':>7}")

    results = {}

    for name, (subset, score) in candidates.items():
        results[name] = auc(trades, subset, injected, score)
        pooled, weighted = results[name]
        print(f"  {name:<9} {pooled:>7.3f} {weighted:>11.3f} {len(subset):>7}")

    return results


def explain_columns(trades):

    legislators = pd.read_parquet(LEGISLATORS_FILE)

    names = (
        legislators
        .sort_values("congress")
        .groupby("memberId")[["name", "party"]]
        .last()
    )

    committees = build_committee_features()

    committee_sectors = (
        committees
        .groupby(["memberId", "congress"])["sector"]
        .agg(lambda sectors: ", ".join(sorted(sectors)))
        .rename("committee_sectors")
    )

    congress = trades["td"].map(congress_from_date)

    return (
        trades[["memberId"]]
        .assign(congress=congress)
        .join(names, on="memberId")
        .join(committee_sectors, on=["memberId", "congress"])
        [["name", "party", "committee_sectors"]]
        .fillna({"committee_sectors": ""})
    )


def main():

    trades = load_trades()

    model_df = pd.read_parquet(MODEL_DATA_FILE)

    house = house_probs(trades)

    alpha = tune_alpha(trades, house)

    p_history = history_probs(trades, alpha, house)
    p_model = model_probs(trades, model_df)

    recent = recent_counts(trades)

    evaluate(trades, p_history, p_model, recent)

    sector_index = trades["sector"].map(SECTORS.index).to_numpy()

    scores = trades[[
        "memberId", "td", "ticker", "assetDescription", "action",
        "amount_mid", "sector", "sourceDocId", "sourceUrl"
    ]].copy()

    scores["score_history"] = surprise(p_history, sector_index)
    scores["score_model"] = surprise(p_model, sector_index)

    # Trades before 2023 have no out-of-sample model score.
    scores.loc[np.isnan(p_model).any(axis=1), "score_model"] = np.nan

    scores["recent_sector_trades"] = (
        recent[np.arange(len(recent)), sector_index].astype(int)
    )

    scores["is_routine"] = routine_mask(recent, sector_index)

    # Share alone keeps flagging a heavy trader's small but regular sectors.
    scores["anomaly_score"] = scores["score_history"].where(
        ~scores["is_routine"],
        0.0
    )

    scores["flag"] = np.where(
        scores["anomaly_score"] > ANOMALY_FLAG_SCORE,
        "F",
        "U"
    )

    scores["anomaly_rank"] = scores["anomaly_score"].rank(pct=True)

    scores["member_rank"] = (
        scores
        .groupby("memberId")["anomaly_rank"]
        .rank(pct=True)
    )

    scores = pd.concat(
        [scores, explain_columns(trades)],
        axis=1
    ).sort_values("anomaly_rank", ascending=False)

    scores.to_parquet(OUT, index=False)

    print(f"\nWrote {len(scores)} scored trades to {OUT}")
    print(
        f"Flagged {(scores['flag'] == 'F').sum()} trades "
        f"(score above {ANOMALY_FLAG_SCORE})"
    )

    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(
            scores.head(20)[[
                "name", "td", "ticker", "sector", "sourceDocId",
                "recent_sector_trades", "anomaly_score", "flag", "committee_sectors"
            ]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
