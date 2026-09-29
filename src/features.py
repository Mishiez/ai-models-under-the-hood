"""Row-wise engineered features (hypotheses in reports/feature_hypotheses.md).

Every function looks at one row at a time, so nothing is learned from other rows and
computing them before the split cannot leak. Anything that needs fitting stays in the Pipeline.
"""
import numpy as np
import pandas as pd

TIME_ORIGIN = pd.Timestamp("2018-01-01")  # first date in the dataset


def benchmark_days(df):
    return (pd.to_datetime(df["benchmark_date"]) - TIME_ORIGIN).dt.days


FEATURES = {"benchmark_days": benchmark_days}


def add_features(df, names=None):
    out = df.copy()
    for name in names or FEATURES:
        out[name] = FEATURES[name](df)
    return out
