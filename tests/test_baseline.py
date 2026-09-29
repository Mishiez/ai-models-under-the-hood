import numpy as np

from src.config import CATEGORICAL, GROUP, NUMERIC, TARGET
from src.pipeline import build_pipeline
from src.split import load_split

COLUMNS = NUMERIC + CATEGORICAL
LEAKY = {"benchmark_score", "benchmark_rank", "benchmark_percentile", "model_value_score",
         "parameter_efficiency", "benchmark_score_per_dollar", "model_name", "model_id"}


def test_no_model_in_both_train_and_test():
    train, test = load_split()
    assert not set(train[GROUP]) & set(test[GROUP])


def test_target_is_never_missing():
    train, test = load_split()
    assert train[TARGET].notna().all() and test[TARGET].notna().all()


def test_no_leaky_columns_in_inputs():
    assert not LEAKY & set(COLUMNS)


def test_scaler_and_encoder_learn_from_train_only():
    train, test = load_split()
    pipe = build_pipeline().fit(train[COLUMNS], train[TARGET])
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_, train[NUMERIC].mean().to_numpy())
    assert not np.allclose(scaler.mean_, test[NUMERIC].mean().to_numpy())
