# ai-models-under-the-hood
What really separates one AI model from another? This project digs beneath the surface of modern AI models, exploring the relationship between capability, cost, performance, and technical characteristics—and using data to uncover the signals hiding underneath.

## I2: Feature engineering and data quality

Question: can basic model characteristics predict `benchmark_score_normalized` on `benchmark_result` rows of the AI Wars dataset?

Short answer: barely. Original columns do no better than predicting the mean; the one engineered feature with signal is the benchmark date (test RMSE 16.30 to 16.14).

| Step | Where |
|---|---|
| Data-quality audit (leakage, missing target, unusable IDs, fake sources) | `reports/data_quality_audit.md`, `src/audit.py` |
| Baseline (group split, Ridge pipeline, dummy floor) | `reports/baseline_metrics.md`, `src/split.py`, `src/pipeline.py`, `src/baseline.py` |
| Feature hypotheses, written before modelling | `reports/feature_hypotheses.md` |
| Before/after comparison | `reports/comparison.md`, `src/features.py`, `src/compare.py` |

Run (with `data/AI_Wars_Dataset.csv` in place): `python -m src.audit`, `python -m src.split`, `python -m src.baseline`, `python -m src.compare`, `python -m pytest`.
