# Baseline metrics

Task: predict `benchmark_score_normalized` on labelled `benchmark_result` rows (29,986).
Split: `GroupShuffleSplit` by `model_name`, 80/20, seed 42, saved in `splits/split.json`.
Train 24,048 rows / 395 models; test 5,938 rows / 99 models; no model in both.
Inputs: 10 numeric + 7 categorical original columns (see `src/config.py`); leaky and identifier columns excluded per the audit.
Model: median impute + StandardScaler + one-hot, then Ridge(alpha=1), all inside one Pipeline. "Folds" are 5-fold GroupKFold on the training rows only.

| Model | Test RMSE | Test MAE | Test R² | Fold RMSE (mean ± std) | Fold R² (mean ± std) |
|---|---|---|---|---|---|
| Dummy (predict train mean) | 16.270 | 13.321 | -0.000 | 16.180 ± 0.165 | -0.000 ± 0.000 |
| Ridge baseline | 16.303 | 13.326 | -0.004 | 16.232 ± 0.149 | -0.007 ± 0.003 |

## Reading
The baseline does not beat the floor: RMSE is 0.03 worse on test and 0.05 worse across folds (within one fold std), and R² is slightly negative. The original columns carry essentially no signal about the target, which matches the audit (groupings explain only chance-level variance; feature correlations below 0.05). This is the number the engineered features have to beat. Any gain should be judged against the fold std of about 0.15 RMSE, not the test score alone.

Reproduce: `python -m src.split && python -m src.baseline`
