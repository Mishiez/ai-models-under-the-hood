"""Data-quality audit of the benchmark_result subset. Run: python -m src.audit"""
import pandas as pd

from src.config import TARGET
from src.load import load_subset


def missingness(df):
    """Share of missing values per column, worst first (columns with none are dropped)."""
    miss = df.isna().mean().sort_values(ascending=False)
    return miss[miss > 0].rename("missing_share").to_frame()


def target_redundancy(df):
    """Is benchmark_score just a copy of the target? Compare them where both exist."""
    both = df.dropna(subset=["benchmark_score", TARGET])
    return {
        "rows_with_both": len(both),
        "identical_share": float((both["benchmark_score"] == both[TARGET]).mean()),
        "rows_missing_score_but_have_target": int(
            (df["benchmark_score"].isna() & df[TARGET].notna()).sum()
        ),
    }


def main():
    df = load_subset()
    print(f"rows: {len(df):,}")
    print("\n== missingness ==")
    print(missingness(df).to_string())

    print("\n== score vs normalized ==")
    print(target_redundancy(df))


if __name__ == "__main__":
    main()
