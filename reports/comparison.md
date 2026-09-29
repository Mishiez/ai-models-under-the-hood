# Before / after: engineered features

Setup is identical to `reports/baseline_metrics.md` (group split by `model_name`, Ridge pipeline fitted on train only, 5-fold GroupKFold on train). Each engineered feature is added to the baseline inputs alone, then all four together. Hypotheses and decision rules were committed first in `reports/feature_hypotheses.md`. Reproduce with `python -m src.compare`.

| Variant | Test RMSE | Test R² | Fold RMSE (mean ± std) | Fold RMSE gain | Test RMSE gain | Meets pre-registered rule |
|---|---|---|---|---|---|---|
| Baseline | 16.303 | -0.004 | 16.232 ± 0.149 | - | - | - |
| + benchmark_days | 16.141 | 0.015 | 16.098 ± 0.134 | 0.134 | 0.162 | no (see below) |
| + log_parameters | 16.303 | -0.004 | 16.233 ± 0.148 | -0.001 | 0.000 | no |
| + log_context_window | 16.303 | -0.004 | 16.233 ± 0.149 | -0.001 | 0.000 | no |
| + price_ratio | 16.306 | -0.005 | 16.232 ± 0.148 | 0.001 | -0.003 | no |
| + all four | 16.144 | 0.015 | 16.098 ± 0.133 | 0.134 | 0.159 | no |

Spearman correlation with the target on the training rows:

| Feature | rho | p-value | Signal (p < 0.01)? |
|---|---|---|---|
| benchmark_days | 0.120 | < 0.0001 | yes |
| log_parameters | -0.001 | 0.91 | no |
| log_context_window | -0.009 | 0.14 | no |
| price_ratio | -0.010 | 0.13 | no |

## Result
- **benchmark_days (H1) is supported.** Correlation 0.12, the only feature with signal, and it is the only one that moves the model: test RMSE 16.30 to 16.14, test R² from -0.004 to 0.015 (the first positive R² we have). Its per-fold RMSE gain, computed on the same folds as the baseline, was positive in all 5 folds (0.093, 0.107, 0.174, 0.144, 0.154). That paired check was added after seeing the results and is exploratory.
- **It narrowly fails the strict pre-registered rule.** The rule required a fold gain larger than the baseline fold std (0.149); the gain was 0.134. We keep the rule as written and do not claim it passed. The std compares across folds of different data, while the paired check compares the same folds, so the rule is arguably too strict, but that judgement is made after the fact.
- **H2, H3 and H4 are not supported.** log_parameters, log_context_window and price_ratio have no correlation with the target (|rho| < 0.011, p > 0.1) and change RMSE by less than 0.003. This matches the audit: size, context and price carry no signal in this dataset.
- **The improvement is small.** Even the best variant explains 1.5% of variance; RMSE falls by about 1%. Scores here still look mostly like noise around a slow time trend.

## Caveats
- The data is partly simulated, so "scores rise over time" describes how this file was generated, not real model progress.
- benchmark_days was motivated by a descriptive pattern seen in the audit, so it is not a fully blind hypothesis (disclosed in the hypotheses file).
- All comparisons use one split with one seed; the fold results are what guard against a lucky split.
