from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "AI_Wars_Dataset.csv"
SAMPLE_CSV = DATA_DIR / "AI_Wars_Sample_10000.csv"
SOURCES_CSV = DATA_DIR / "AI_Wars_Sources.csv"
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

GROUP = "model_name"  # model_id is unique per row, so it cannot define a group
TEST_SIZE = 0.2
SPLIT_JSON = ROOT / "splits" / "split.json"

# Baseline inputs: original columns only. Excluded on purpose:
#   benchmark_score                         copy of the target
#   benchmark_rank, benchmark_percentile    describe the outcome of the score
#   model_name, model_id, record_id         identifiers / group key
#   release_date, benchmark_date            raw dates; reserved for the feature branch
NUMERIC = [
    "parameter_count_billions", "context_window_tokens",
    "input_price_per_1m_tokens", "output_price_per_1m_tokens",
    "model_age_days", "context_to_parameter_ratio", "confidence_score",
    "open_weights", "open_source", "benchmark_verified",
]
CATEGORICAL = [
    "benchmark_name", "benchmark_category", "model_type", "model_family",
    "company_name", "headquarters_country", "data_source_type",
]
