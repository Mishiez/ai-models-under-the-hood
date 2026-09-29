import json

from sklearn.model_selection import GroupShuffleSplit

from src.config import GROUP, SEED, SPLIT_JSON, TARGET, TEST_SIZE
from src.load import load_subset


def labelled_rows():
    """Rows that have a target; the unlabelled quarter is dropped (see the audit)."""
    return load_subset().dropna(subset=[TARGET]).reset_index(drop=True)


def make_split(df):
    """One 80/20 split where every model_name lands wholly in train or wholly in test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP]))
    return {
        "seed": SEED,
        "group": GROUP,
        "train_groups": sorted(df.iloc[train_idx][GROUP].unique()),
        "test_groups": sorted(df.iloc[test_idx][GROUP].unique()),
    }


def save_split(split):
    SPLIT_JSON.parent.mkdir(exist_ok=True)
    SPLIT_JSON.write_text(json.dumps(split, indent=1))


def load_split():
    """Return (train_df, test_df) using the saved group lists, so every run uses the same rows."""
    df = labelled_rows()
    split = json.loads(SPLIT_JSON.read_text())
    is_test = df[GROUP].isin(split["test_groups"])
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


if __name__ == "__main__":
    df = labelled_rows()
    save_split(make_split(df))
    train, test = load_split()
    print(f"train: {len(train):,} rows / {train[GROUP].nunique()} models")
    print(f"test:  {len(test):,} rows / {test[GROUP].nunique()} models")
    print("overlap:", set(train[GROUP]) & set(test[GROUP]))
