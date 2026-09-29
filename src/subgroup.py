"""Does the model's error differ across groups? Run: python -m src.subgroup

Uses out-of-fold predictions over all labelled rows (GroupKFold by model_name), so every
prediction comes from a model that never saw that row's model_name.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.baseline import N_FOLDS
from src.config import CATEGORICAL, GROUP, NUMERIC, TARGET
from src.features import add_features
from src.pipeline import build_pipeline
from src.split import labelled_rows

GROUPINGS = ["open_weights", "model_type", "headquarters_country", "company_name",
             "benchmark_name", "data_source_type"]
MIN_ROWS = 100
Z_FLAG = 3.0  # about 100 groups are tested, so 2 would flag several by chance alone


def out_of_fold_residuals(extra=("benchmark_days",)):
    df = add_features(labelled_rows(), list(extra))
    numeric = NUMERIC + list(extra)
    columns = numeric + CATEGORICAL
    pred = cross_val_predict(build_pipeline(numeric=numeric), df[columns], df[TARGET],
                             groups=df[GROUP], cv=GroupKFold(n_splits=N_FOLDS))
    return df.assign(pred=pred, residual=df[TARGET] - pred)


def error_by_group(df, column):
    """RMSE, mean residual (bias) and z-scores per group; bias > 0 means under-predicted."""
    overall_sd = df["residual"].std()
    overall_rmse = float(np.sqrt((df["residual"] ** 2).mean()))
    rows = []
    for level, g in df.groupby(column):
        if len(g) < MIN_ROWS:
            continue
        rows.append({
            "group": level, "rows": len(g),
            "rmse": float(np.sqrt((g["residual"] ** 2).mean())),
            "bias": float(g["residual"].mean()),
            "bias_z": float(g["residual"].mean() / (overall_sd / np.sqrt(len(g)))),
            # sampling error of an RMSE is about rmse / sqrt(2n) for roughly normal residuals
            "rmse_z": float((np.sqrt((g["residual"] ** 2).mean()) - overall_rmse)
                            / (overall_rmse / np.sqrt(2 * len(g)))),
        })
    out = pd.DataFrame(rows).set_index("group")
    out["flag"] = (out["bias_z"].abs() > Z_FLAG) | (out["rmse_z"].abs() > Z_FLAG)
    return out


def main():
    df = out_of_fold_residuals()
    print(f"rows: {len(df):,}  overall RMSE: {np.sqrt((df['residual'] ** 2).mean()):.3f}")
    tables = {col: error_by_group(df, col) for col in GROUPINGS}
    for col, t in tables.items():
        print(f"\n== {col}: {len(t)} groups, rmse {t['rmse'].min():.2f} to {t['rmse'].max():.2f}, "
              f"flagged {int(t['flag'].sum())}")
        shown = t if len(t) <= 8 else t.reindex(t["rmse_z"].abs().sort_values(ascending=False).index).head(5)
        print(shown.round(3).to_string())
    return tables


if __name__ == "__main__":
    main()
