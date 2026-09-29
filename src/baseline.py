"""Baseline: Ridge on original columns vs a predict-the-mean floor. Run: python -m src.baseline"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate

from src.config import CATEGORICAL, GROUP, NUMERIC, ROOT, TARGET
from src.pipeline import build_pipeline
from src.split import load_split

N_FOLDS = 5


def heldout_metrics(model, train, test, columns):
    model = clone(model).fit(train[columns], train[TARGET])
    pred = model.predict(test[columns])
    return {
        "rmse": float(np.sqrt(mean_squared_error(test[TARGET], pred))),
        "mae": float(mean_absolute_error(test[TARGET], pred)),
        "r2": float(r2_score(test[TARGET], pred)),
    }


def fold_metrics(model, train, columns):
    """GroupKFold on the training rows only: mean and std of RMSE and R2 across folds."""
    scores = cross_validate(
        clone(model), train[columns], train[TARGET], groups=train[GROUP],
        cv=GroupKFold(n_splits=N_FOLDS),
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
    )
    rmse, r2 = -scores["test_rmse"], scores["test_r2"]
    return {"rmse_mean": rmse.mean(), "rmse_std": rmse.std(),
            "r2_mean": r2.mean(), "r2_std": r2.std()}


def evaluate(models, train, test, columns):
    rows = []
    for name, model in models.items():
        row = {"model": name, **heldout_metrics(model, train, test, columns),
               **fold_metrics(model, train, columns)}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main():
    train, test = load_split()
    columns = NUMERIC + CATEGORICAL
    models = {"dummy (predict mean)": DummyRegressor(strategy="mean"),
              "ridge baseline": build_pipeline()}
    table = evaluate(models, train, test, columns)
    print(table.round(3).to_string())
    return table


if __name__ == "__main__":
    main()
