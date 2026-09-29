"""Data-quality audit of the benchmark_result subset. Run: python -m src.audit"""
import pandas as pd

from src.config import ROW_TYPE, SAMPLE_CSV, TARGET
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


def source_mix(df):
    """Source, verification and data-layer mix, with mean target per level and label overlap."""
    labelled = df.dropna(subset=[TARGET])
    out = {}
    for col in ("data_source_type", "verification_status", "data_layer", "benchmark_verified"):
        out[col] = pd.DataFrame({
            "rows": df[col].value_counts(),
            "share": df[col].value_counts(normalize=True).round(3),
            "mean_target": labelled.groupby(col)[TARGET].mean().round(2),
        })
    out["verification_vs_source"] = pd.crosstab(df["data_source_type"], df["verification_status"])
    return out


def target_distribution(df):
    """Range, skew and clipping of the target, overall and by benchmark date year."""
    labelled = df.dropna(subset=[TARGET])
    year = pd.to_datetime(labelled["benchmark_date"]).dt.year
    by_year = labelled.groupby(year)[TARGET].agg(["count", "mean", "std"]).round(2)
    summary = labelled[TARGET].describe().round(2).to_dict()
    summary["skew"] = round(float(labelled[TARGET].skew()), 3)
    summary["share_at_100"] = float((labelled[TARGET] >= 100).mean())
    return {"summary": summary, "by_benchmark_year": by_year,
            "by_benchmark": labelled.groupby("benchmark_name")[TARGET].mean().round(1).describe().round(2)}


def group_balance(df):
    """Row counts and mean target for the groups where sampling skew or bias could hide."""
    labelled = df.dropna(subset=[TARGET])
    out = {}
    for col in ("open_weights", "model_type", "headquarters_country", "benchmark_category"):
        counts = df[col].value_counts()
        out[col] = pd.DataFrame({
            "rows": counts,
            "share": (counts / len(df)).round(3),
            "mean_target": labelled.groupby(col)[TARGET].mean().round(2),
        }).head(10)
    per_company = df["company_name"].value_counts()
    out["rows_per_company"] = per_company.describe().round(1)
    return out


GENERATED_ON = pd.Timestamp("2026-09-14")  # from AI_Wars_Validation_Report.txt


def date_checks(df):
    """Do the date columns agree with each other and with the dataset's generation date?"""
    bench = pd.to_datetime(df["benchmark_date"])
    release = pd.to_datetime(df["release_date"])
    start = pd.Timestamp(bench.min().year, 1, 1)
    return {
        "benchmark_date_equals_release_date_share": float((bench == release).mean()),
        "benchmark_date_after_generation_share": float((bench > GENERATED_ON).mean()),
        "latest_benchmark_date": str(bench.max().date()),
        "model_age_days_vs_dates_corr": float(df["model_age_days"].corr((bench - start).dt.days)),
        "model_age_days_range": (int(df["model_age_days"].min()), int(df["model_age_days"].max())),
    }


LEAK_KEYWORDS = ("score", "rank", "percentile", "performance_to", "leaderboard", "efficiency", "per_dollar")


def leakage_scan():
    """Name-scan all 483 columns for target-like words, then test the hits on benchmark rows.

    Uses the 10k sample (all columns, ~1.4k labelled benchmark rows) instead of the full CSV.
    """
    sample = pd.read_csv(SAMPLE_CSV, low_memory=False)
    rows = sample[(sample["row_type"] == ROW_TYPE) & sample[TARGET].notna()]
    hits = [c for c in sample.columns if c != TARGET and any(k in c for k in LEAK_KEYWORDS)]
    table = pd.DataFrame({
        "populated_share": rows[hits].notna().mean(),
        "abs_corr_with_target": rows[hits].apply(pd.to_numeric, errors="coerce").corrwith(rows[TARGET]).abs(),
    }).round(3)
    return table[table["populated_share"] > 0].sort_values("abs_corr_with_target", ascending=False)


def leakage_formulas():
    """Spot-check how the strongest leaks relate to the target (ratio to target, per row)."""
    sample = pd.read_csv(SAMPLE_CSV, low_memory=False)
    rows = sample[(sample["row_type"] == ROW_TYPE) & sample[TARGET].notna()]
    ratios = {
        "model_value_score / target": rows["model_value_score"] / rows[TARGET],
        "benchmark_score_per_dollar * input_price / target":
            rows["benchmark_score_per_dollar"] * rows["input_price_per_1m_tokens"] / rows[TARGET],
    }
    return pd.DataFrame({k: v.describe()[["mean", "std", "min", "max"]] for k, v in ratios.items()}).round(3).T


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

    print("\n== source and verification mix ==")
    for name, table in source_mix(df).items():
        print(f"-- {name}")
        print(table.to_string())

    print("\n== target distribution ==")
    dist = target_distribution(df)
    print(dist["summary"])
    print(dist["by_benchmark_year"].to_string())
    print("mean target across the 23 benchmarks:")
    print(dist["by_benchmark"].to_string())

    print("\n== group balance ==")
    for name, table in group_balance(df).items():
        print(f"-- {name}")
        print(table.to_string())

    print("\n== date consistency ==")
    print(date_checks(df))

    print("\n== leakage scan ==")
    print(leakage_scan().head(12).to_string())
    print(leakage_formulas().to_string())


if __name__ == "__main__":
    main()
