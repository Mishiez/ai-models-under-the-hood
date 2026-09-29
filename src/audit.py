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


def group_structure(df):
    """How much target variance does each grouping explain, versus what chance alone gives?

    With k groups and n rows, groups of pure noise still "explain" about (k-1)/(n-1).
    """
    labelled = df.dropna(subset=[TARGET])
    n, total_var = len(labelled), labelled[TARGET].var()
    rows = []
    for cols in (["benchmark_name"], ["company_name"], ["model_family"], ["model_name"],
                 ["model_name", "benchmark_name"]):
        grouped = labelled.groupby(cols)[TARGET]
        rows.append({
            "grouping": " + ".join(cols),
            "groups": grouped.ngroups,
            "variance_explained": grouped.transform("mean").var() / total_var,
            "expected_by_chance": (grouped.ngroups - 1) / (n - 1),
        })
    return pd.DataFrame(rows).round(4)


def model_attribute_consistency(df):
    """Max distinct values of an attribute within one model_name (a real model would have 1)."""
    cols = ["company_name", "model_type", "open_weights", "parameter_count_billions",
            "context_window_tokens", "input_price_per_1m_tokens", "release_date"]
    return df.groupby("model_name")[cols].nunique().max().rename("max_distinct_per_model_name").to_frame()


def group_structure(df):
    """How much target variance does each grouping explain, versus what chance alone gives?

    With k groups and n rows, groups of pure noise still "explain" about (k-1)/(n-1).
    """
    labelled = df.dropna(subset=[TARGET])
    n, total_var = len(labelled), labelled[TARGET].var()
    rows = []
    for cols in (["benchmark_name"], ["company_name"], ["model_family"], ["model_name"],
                 ["model_name", "benchmark_name"]):
        grouped = labelled.groupby(cols)[TARGET]
        rows.append({
            "grouping": " + ".join(cols),
            "groups": grouped.ngroups,
            "variance_explained": grouped.transform("mean").var() / total_var,
            "expected_by_chance": (grouped.ngroups - 1) / (n - 1),
        })
    return pd.DataFrame(rows).round(4)


def model_attribute_consistency(df):
    """Max distinct values of an attribute within one model_name (a real model would have 1)."""
    cols = ["company_name", "model_type", "open_weights", "parameter_count_billions",
            "context_window_tokens", "input_price_per_1m_tokens", "release_date"]
    return df.groupby("model_name")[cols].nunique().max().rename("max_distinct_per_model_name").to_frame()


def main():
    df = load_subset()
    print(f"rows: {len(df):,}")
    print("\n== missingness ==")
    print(missingness(df).to_string())

    print("\n== score vs normalized ==")
    print(target_redundancy(df))

    print("\n== duplicates and identifiers ==")
    print(duplicates(df))

    print("\n== target variance by grouping ==")
    print(group_structure(df))

    print("\n== attributes within one model_name ==")
    print(model_attribute_consistency(df))


if __name__ == "__main__":
    main()
