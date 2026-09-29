from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "AI_Wars_Dataset.csv"
SAMPLE_CSV = DATA_DIR / "AI_Wars_Sample_10000.csv"
SUBSET_PKL = DATA_DIR / "benchmark_subset.pkl"

SEED = 42
ROW_TYPE = "benchmark_result"
TARGET = "benchmark_score_normalized"

# Columns read from the 483-column CSV. Everything else is skipped at read time.
COLUMNS = [
    "record_id", "row_type",
    "data_layer", "data_source_type", "verification_status", "confidence_score",
    "model_id", "model_name", "model_family", "model_type",
    "company_name", "headquarters_country",
    "open_weights", "open_source",
    "parameter_count_billions", "context_window_tokens",
    "input_price_per_1m_tokens", "output_price_per_1m_tokens",
    "release_date", "benchmark_date", "model_age_days", "context_to_parameter_ratio",
    "benchmark_name", "benchmark_category", "benchmark_verified",
    "benchmark_score", TARGET, "benchmark_rank", "benchmark_percentile",
]
