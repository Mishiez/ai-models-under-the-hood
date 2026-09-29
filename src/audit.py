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
        "max_abs_difference": float((both["benchmark_score"] - both[TARGET]).abs().max()),
        "equal_within_1e-6_share": float(
            ((both["benchmark_score"] - both[TARGET]).abs() < 1e-6).mean()
        ),
        "rows_missing_score_but_have_target": int(
            (df["benchmark_score"].isna() & df[TARGET].notna()).sum()
        ),
    }


def duplicates(df):
    """Repeated rows for the same model and benchmark, and whether IDs identify anything."""
    labelled = df.dropna(subset=[TARGET])
    return {
        "model_id_unique_per_row": bool(df["model_id"].is_unique),
        "distinct_model_names": int(df["model_name"].nunique()),
        "distinct_model_families": int(df["model_family"].nunique()),
        "labelled_rows": len(labelled),
        "distinct_model_benchmark_pairs": len(labelled.groupby(["model_name", "benchmark_name"])),
        "rows_repeating_a_model_benchmark_pair": int(
            labelled.duplicated(["model_name", "benchmark_name"]).sum()
        ),
        "benchmark_name_aliases": sorted(
            n for n in df["benchmark_name"].unique() if n in {"BBH", "BIG-Bench-Hard"}
        ),
    }


def main():
    df = load_subset()
    print(f"rows: {len(df):,}")
    print("\n== missingness ==")
    print(missingness(df).to_string())

    print("\n== score vs normalized ==")
    print(target_redundancy(df))

    print("\n== duplicates and identifiers ==")
    print(duplicates(df))


if __name__ == "__main__":
    main()
