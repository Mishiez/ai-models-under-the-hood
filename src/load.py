import pandas as pd

from src.config import COLUMNS, RAW_CSV, ROW_TYPE, SUBSET_PKL, TARGET

CHUNK_ROWS = 50_000


def build_subset(path=RAW_CSV):
    """Stream the big CSV, keep only benchmark_result rows and the chosen columns."""
    parts = []
    for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=CHUNK_ROWS, low_memory=False):
        parts.append(chunk[chunk["row_type"] == ROW_TYPE])
    return pd.concat(parts, ignore_index=True)


def load_subset(rebuild=False):
    """Return the cached benchmark subset, building it from the raw CSV if needed."""
    if SUBSET_PKL.exists() and not rebuild:
        return pd.read_pickle(SUBSET_PKL)
    df = build_subset()
    df.to_pickle(SUBSET_PKL)
    return df


if __name__ == "__main__":
    df = load_subset(rebuild=True)
    print(f"rows: {len(df):,}  columns: {df.shape[1]}")
    print(f"target populated: {df[TARGET].notna().sum():,} ({df[TARGET].notna().mean():.1%})")
