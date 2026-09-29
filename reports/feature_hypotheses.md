# Feature hypotheses (written before any feature was modelled)

Target: `benchmark_score_normalized`. Baseline to beat (`reports/baseline_metrics.md`): test RMSE 16.30, fold RMSE 16.23 ± 0.15, R² about 0. The data audit (`reports/data_quality_audit.md`) shows the original columns carry almost no signal, so most of these are expected to be weak. That expectation is stated here so a null result is not a surprise.

**Disclosure:** the audit showed, descriptively, that mean target rises with benchmark year (62 in 2018 to 71 in 2026). Feature 1 is motivated by that and is therefore not a blind guess; the other three come from general domain reasoning and were not looked at against the target.

| # | Feature | Formula | Hypothesis | Availability check |
|---|---|---|---|---|
| 1 | `benchmark_days` | days from 2018-01-01 to `benchmark_date` | Scores drift upward over time (newer evaluations score higher). Expected direction: positive. | `benchmark_date` has no missing values; 5.4% of dates are after the generation date, kept and noted. |
| 2 | `log_parameters` | `log1p(parameter_count_billions)` | Returns to model size diminish, so a log scale fits a linear model better than raw size. Expected direction: positive but small. | Min value 1, no missing values, no zeros. |
| 3 | `log_context_window` | `log(context_window_tokens)` | Context window spans 4k to 2M tokens, so raw values are dominated by a few huge models; the log scale is what a linear model can use. Expected direction: positive but small. | Min 4,096, no missing values. Chosen over "context per parameter" because the existing `context_to_parameter_ratio` column does not match `context_window_tokens / parameter_count` and could not be verified. |
| 4 | `price_ratio` | `output_price / input_price`, missing if input price is 0 | Output-to-input price ratio is a pricing-tier proxy for model class; premium models price output higher. Expected direction: unclear. | Input price min is 0.01, so no zero denominators occur; the guard is kept anyway. |

Spare (not built unless one above fails its availability check): none needed, all four passed.

Dropped from the original plan: "model age = benchmark date minus release date". `benchmark_date == release_date` on every row (audit finding 6), so it is always 0.

## Decision rules (fixed now)
- A feature "helps" only if adding it lowers 5-fold GroupKFold RMSE by more than one fold std of the baseline (0.15) **and** does not raise held-out test RMSE.
- Per-feature Spearman correlation with the target is reported on the training rows; p < 0.01 counts as signal (with n of about 24,000, a correlation above 0.02 is significant).
- If nothing meets the first rule, the conclusion is "no improvement", reported as such with the fold numbers.
