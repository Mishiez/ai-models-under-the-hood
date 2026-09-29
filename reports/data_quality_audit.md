# Data-quality audit: AI Wars `benchmark_result` rows

Scope: the 39,979 rows with `row_type == "benchmark_result"` (full dataset, 29 columns loaded by `src/load.py`). The prediction target is `benchmark_score_normalized`. Every number below is reproduced by `python -m src.audit`.

**Caveat that frames everything:** this dataset is partly simulated. Its own validation report states a generation seed, the sources file points only to `example.com`, and the numbers below show the target behaves like random noise for most columns. Findings are about *this file*, not about real AI models.

## Findings that change how we model

### 1. A quarter of the target is missing, and its twin column is missing on the same rows
`benchmark_score_normalized` is empty on 9,993 of 39,979 rows (25.0%). `benchmark_score` is empty on exactly the same rows. Neither the README nor the validation report mentions this (the report quotes 1.64% average missingness over all 483 columns). Rows without a target are dropped, leaving 29,986 labelled rows.

### 2. `benchmark_score` is a copy of the target (leakage)
Where both exist they differ by at most 1e-9 (correlation 1.0). It must never be a feature.

### 3. Derived columns hide the target (leakage)
A name scan over all 483 columns, tested on the benchmark rows of the 10k sample:

| Column | \|corr\| with target | What it is |
|---|---|---|
| `model_value_score` | 0.66 | about 1.0 × target with ±23% noise |
| `parameter_efficiency` | 0.42 | a fixed-range multiple of the target (0.13 to 0.59 × target) |
| `benchmark_score_per_dollar`, `benchmark_efficiency`, `performance_to_cost_ratio` | 0.17 | target ÷ input price (`score_per_dollar × input_price ≈ 0.99 × target`) |
| `cost_efficiency`, `overall_cost_efficiency`, `coding_/reasoning_cost_efficiency` | 0.17 | same family |

All are marked `derived = No` in the data dictionary, so the dictionary cannot be used to find them. All are excluded from every model. By contrast `benchmark_rank` and `benchmark_percentile` are *not* leaks (|corr| about 0.002), despite their names.

### 4. `model_id` identifies nothing, and `model_name` is not a coherent model
`model_id` is unique on every one of the 39,979 rows, although the README advertises 494 unique models (that is the count of `model_name`). Within a single `model_name`, one name can carry up to 47 different companies, 41 parameter counts, 8 model types, 110 input prices and both open and closed weights. So a "model" here is a label attached to randomly assigned attributes. Consequence: a group split on `model_id` groups nothing; we group on `model_name` (494 groups), which is the dataset's own notion of a model.

### 5. The target carries almost no group structure; only time carries signal
Share of target variance explained by a grouping, against what pure noise gives for that many groups:

| Grouping | Groups | Explained | Expected by chance |
|---|---|---|---|
| benchmark | 23 | 0.06% | 0.07% |
| company | 50 | 0.16% | 0.16% |
| model_family | 38 | 0.12% | 0.12% |
| model_name | 494 | 1.43% | 1.64% |
| model_name × benchmark | 10,544 | 35.3% | 35.2% |

Group means are indistinguishable from chance. All 23 benchmarks average 68.0 ± 0.4, so the usual "benchmarks differ in difficulty" assumption does not hold. The one real pattern is the benchmark year: mean target rises from 62.2 (2018) to 70.7 (2026), correlation about 0.13.

### 6. `benchmark_date` equals `release_date` on every row, and `model_age_days` is unrelated to either
The date columns are identical on 100% of rows, so "benchmark date minus release date" is always zero. `model_age_days` (1 to 1,999) has correlation 0.0001 with the dates it should be derived from. It cannot be treated as a real age.

### 7. Heavy repetition: about 65% of labelled rows repeat a model/benchmark pair
10,544 distinct (`model_name`, `benchmark_name`) pairs cover 29,986 labelled rows, so 19,442 rows repeat a pair (up to 10 per pair). Repeats have different dates and scores, so they are not exact duplicates, but a random row split would put the same model in train and test. `BBH` and `BIG-Bench-Hard` are also two names for the same benchmark.

## Sampling skew and possible bias

- **Geography:** 64% of rows have a US-headquartered company, 18% China, 6% South Korea, and every other country is 2 to 4%. Companies themselves are balanced (748 to 875 rows each), so the skew comes from how many US companies exist, not from oversampling.
- **Open vs closed:** 35% open weights, 65% closed. Mean target is 68.01 vs 68.03, so no bias in the label here.
- **Model type:** language is 35% and embedding 3%, but mean target ranges only from 67.7 to 68.6 across types.
- **Verification:** 85% verified, 10% medium, 5% estimated. `verification_status`, `benchmark_verified` and `data_source_type` are the same information, perfectly nested (for example `research_paper` always means verified). Mean target is about equal across levels (67.7 to 68.3), so "estimated" rows are not systematically different.
- **Protected attributes:** none exist (no people). The closest analogues are company country and open/closed, covered above.

## Trust in the documentation

- **Sources file:** 100% of the 1,000 URLs are `https://example.com/...`; 30% have an access date before the publication date; 9.5% are accessed after the dataset was generated (14 Sep 2026). Yet 33,973 benchmark rows are labelled "verified". The verification label cannot be traced to anything.
- **Future dates:** 5.4% of benchmark rows are dated after the generation date (latest 2026-12-28), and the README says the range ends in 2026 without noting it extends past "today".
- **Validation report:** its 8 checks are all structural (unique IDs, no negative prices). It has no check for target missingness, leakage or entity consistency.
- **Data dictionary:** descriptions are boilerplate ("AI Wars Dataset column: Benchmark Score") and `derived` is `No` for the columns in finding 3.

## What this means for the next branches

1. **Split:** group by `model_name`, 80/20, saved to disk. Drop unlabelled rows first.
2. **Features banned:** `benchmark_score`, every column in finding 3, and anything containing rank/percentile from the derived family.
3. **Feature ideas:** the plan's "model age" feature (benchmark date minus release date) is dead (finding 6). Date-based features (benchmark year) are the one candidate with measurable signal. Size, price and context features start with correlations below 0.05, so expect a null or tiny result and report fold statistics honestly.
4. **Benchmark name:** keep it in the baseline for completeness, but it explains 0.06% of variance, which is chance level.

## Update: does the model's error differ across groups?

`python -m src.subgroup` scores every labelled row (29,986) with an out-of-fold prediction from the baseline plus `benchmark_days`, where the model never saw that row's `model_name`. For each group of at least 100 rows it reports RMSE, mean residual (bias) and z-scores for both. A group is flagged at |z| > 3, because about 90 groups are tested and |z| > 2 would flag several by chance alone. Overall RMSE is 16.09.

| Grouping | Groups | RMSE range | Largest \|bias\| | Flagged |
|---|---|---|---|---|
| open vs closed weights | 2 | 16.08 to 16.11 | 0.004 | 0 |
| model type | 8 | 15.68 to 16.29 | 0.018 | 0 |
| headquarters country | 8 | 15.07 to 16.61 | 0.025 | 0 |
| company | 50 | 15.07 to 17.21 | 0.052 | 0 |
| benchmark | 23 | 15.38 to 16.52 | 0.039 | 0 |
| data source type | 5 | 15.77 to 16.36 | 0.019 | 0 |

- **No bias, but not evidence of fairness.** Bias is under 0.06 points in every group (on a 0 to 100 scale with a standard deviation of 16). The model is equally wrong everywhere because it is barely better than predicting the mean, so it has nothing group-specific to get wrong.
- **RMSE spread matches chance.** The widest gap is Suno AI (17.21, z = 2.5) against Mistral AI (15.07, z = -2.2), among 50 companies of about 600 rows each. That is what noise produces at this sample size. Country RMSE ranges from 15.07 (France, 584 rows) to 16.61 (Canada, 568 rows), also within sampling error.
- **The "estimated" rows are not worse.** The 5% of rows the dataset labels as estimated have RMSE 15.96 against 16.1 overall, so nothing says they should be down-weighted.
- **Limit of the check:** the model was given company, country and open-weights as inputs, so near-zero bias for those groups is partly by construction. The meaningful result is that RMSE, which is not forced to zero, is also flat.
