from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL, NUMERIC


def build_pipeline(numeric=NUMERIC, categorical=CATEGORICAL, alpha=1.0):
    """Impute + scale numerics, one-hot categoricals, then Ridge.

    Everything lives inside the Pipeline, so imputer medians, scaler means and the
    category list are learned from the training rows only, in every fit and every CV fold.
    """
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), list(numeric)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
    ])
    return Pipeline([("prep", preprocess), ("model", Ridge(alpha=alpha))])
