import numpy as np
import pandas as pd

from src.features import FEATURES, add_features


def _row(**kw):
    base = {"benchmark_date": "2018-01-11", "parameter_count_billions": 0.0,
            "context_window_tokens": 4096, "input_price_per_1m_tokens": 2.0,
            "output_price_per_1m_tokens": 6.0}
    return pd.DataFrame([{**base, **kw}])


def test_known_values():
    out = add_features(_row())
    assert out.loc[0, "benchmark_days"] == 10
    assert out.loc[0, "log_parameters"] == 0.0
    assert np.isclose(out.loc[0, "log_context_window"], np.log(4096))
    assert out.loc[0, "price_ratio"] == 3.0


def test_zero_input_price_gives_nan_not_inf():
    assert np.isnan(add_features(_row(input_price_per_1m_tokens=0.0)).loc[0, "price_ratio"])


def test_features_are_row_wise():
    df = pd.concat([_row(), _row(parameter_count_billions=500.0)], ignore_index=True)
    alone = add_features(df.iloc[[1]].reset_index(drop=True))
    together = add_features(df).iloc[[1]].reset_index(drop=True)
    for name in FEATURES:
        assert np.isclose(alone.loc[0, name], together.loc[0, name])
