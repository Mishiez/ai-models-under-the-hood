import numpy as np
import pandas as pd

from src.subgroup import MIN_ROWS, error_by_group


def _frame(n_a=200, n_b=200, shift_b=0.0, seed=0):
    rng = np.random.default_rng(seed)
    resid = np.concatenate([rng.normal(0, 10, n_a), rng.normal(shift_b, 10, n_b)])
    return pd.DataFrame({"g": ["a"] * n_a + ["b"] * n_b, "residual": resid})


def test_no_flag_when_groups_are_alike():
    assert not error_by_group(_frame(), "g")["flag"].any()


def test_biased_group_is_flagged():
    out = error_by_group(_frame(shift_b=8.0), "g")
    assert bool(out.loc["b", "flag"]) and out.loc["b", "bias"] > 0


def test_small_groups_are_skipped():
    out = error_by_group(_frame(n_b=MIN_ROWS - 1), "g")
    assert list(out.index) == ["a"]
