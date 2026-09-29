"""Baseline vs engineered features. Run: python -m src.compare"""
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import GroupKFold, cross_validate

from src.baseline import N_FOLDS, fold_metrics, heldout_metrics
from src.config import CATEGORICAL, GROUP, NUMERIC, TARGET
from src.features import FEATURES, add_features
from src.pipeline import build_pipeline
from src.split import load_split

FOLD_STD_BASELINE_RULE = "helps = fold RMSE lower by > 1 baseline fold std AND test RMSE not higher"


def spearman_table(train):
    rows = []
    for name in FEATURES:
        rho, p = spearmanr(train[name], train[TARGET])
        rows.append({"feature": name, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows).set_index("feature")


def run_variant(train, test, extra):
    numeric = NUMERIC + list(extra)
    columns = numeric + CATEGORICAL
    model = build_pipeline(numeric=numeric)
    return {**heldout_metrics(model, train, test, columns), **fold_metrics(model, train, columns)}


def paired_fold_gains(train, extra):
    """Exploratory (not pre-registered): per-fold RMSE gain of baseline minus variant, same folds."""
    def fold_rmse(numeric):
        columns = numeric + CATEGORICAL
        scores = cross_validate(build_pipeline(numeric=numeric), train[columns], train[TARGET],
                                groups=train[GROUP], cv=GroupKFold(n_splits=N_FOLDS),
                                scoring="neg_root_mean_squared_error")
        return -scores["test_score"]
    return fold_rmse(NUMERIC) - fold_rmse(NUMERIC + list(extra))


def main():
    train, test = load_split()
    train, test = add_features(train), add_features(test)
    variants = {"baseline (original columns)": []}
    variants.update({f"+ {name}": [name] for name in FEATURES})
    variants["+ all four"] = list(FEATURES)
    table = pd.DataFrame({name: run_variant(train, test, extra) for name, extra in variants.items()}).T
    base = table.iloc[0]
    table["fold_rmse_gain"] = base["rmse_mean"] - table["rmse_mean"]
    table["test_rmse_gain"] = base["rmse"] - table["rmse"]
    helps = (table["fold_rmse_gain"] > base["rmse_std"]) & (table["test_rmse_gain"] >= 0)
    table["helps"] = helps.map({True: "yes", False: "no"})
    table.loc[table.index[0], "helps"] = "-"
    print(table.round(3).to_string())
    print()
    print(spearman_table(train).round(4).to_string())
    gains = paired_fold_gains(train, ["benchmark_days"])
    print("\nbenchmark_days paired per-fold RMSE gain (exploratory):", gains.round(3).tolist())
    print(f"\nbaseline fold std: {base['rmse_std']:.3f}")
    return table


if __name__ == "__main__":
    main()
