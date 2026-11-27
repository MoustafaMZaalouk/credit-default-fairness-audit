# credit-default-fairness-audit

[![ci](https://github.com/MoustafaMZaalouk/credit-default-fairness-audit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MoustafaMZaalouk/credit-default-fairness-audit/actions/workflows/ci.yml)

A small audit of a credit-default classifier: how well it ranks risk, and how its errors differ by sex and by age group. It compares logistic regression with a gradient-boosting model, each with and without the `SEX` column, and tries one simple post-hoc adjustment. CPU only, runs in about 5 seconds.

This is an AI-assisted educational project written by a student. It is not a credit-scoring product, not client work, and not legal or compliance advice. The data is one Taiwan bank's 2005 card portfolio.

## Results (30,000 clients, 5 random 60/20/20 splits, mean (std over seeds), test split)

| model | ROC AUC | average precision | F1 at chosen threshold |
|---|---|---|---|
| `logreg_without_sex` | 0.717 (0.012) | 0.501 (0.015) | 0.509 (0.016) |
| `logreg_with_sex` | 0.719 (0.012) | 0.502 (0.015) | 0.511 (0.015) |
| `hgb_without_sex` | 0.782 (0.010) | 0.561 (0.017) | 0.544 (0.016) |
| `hgb_with_sex` | 0.782 (0.010) | 0.560 (0.018) | 0.541 (0.019) |

Gaps between groups at the validation-chosen threshold (max minus min across groups; "flag" means predicted to default):

| model | sex flag-rate gap | sex TPR gap | sex FPR gap | age TPR gap | age FPR gap |
|---|---|---|---|---|---|
| `logreg_without_sex` | 0.034 (0.013) | 0.023 (0.040) | 0.018 (0.006) | 0.095 (0.027) | 0.041 (0.021) |
| `logreg_with_sex` | 0.068 (0.020) | 0.061 (0.032) | 0.051 (0.018) | 0.073 (0.017) | 0.039 (0.018) |
| `hgb_without_sex` | 0.043 (0.016) | 0.042 (0.034) | 0.023 (0.009) | 0.081 (0.050) | 0.025 (0.009) |
| `hgb_with_sex` | 0.054 (0.020) | 0.048 (0.043) | 0.034 (0.018) | 0.082 (0.049) | 0.026 (0.011) |

Post-hoc check: choose a separate threshold per sex on validation data so both sexes are flagged at the same overall rate.

| model | F1 before | F1 after | sex flag gap before | after | sex TPR gap before | after |
|---|---|---|---|---|---|---|
| `logreg_without_sex` | 0.509 | 0.509 | 0.034 | 0.017 | 0.023 | 0.029 |
| `logreg_with_sex` | 0.511 | 0.511 | 0.068 | 0.015 | 0.061 | 0.027 |
| `hgb_without_sex` | 0.544 | 0.542 | 0.043 | 0.018 | 0.042 | 0.036 |
| `hgb_with_sex` | 0.541 | 0.540 | 0.054 | 0.022 | 0.048 | 0.046 |

Plots: `results/group_gaps.png`. Raw numbers, including per-group counts, base rates, TPR, FPR and precision for seed 0: `results/results.json`.

## How to read this
- Gradient boosting ranks better than logistic regression (AUC about 0.78 against 0.72). That gap is much larger than the seed-to-seed std.
- Dropping `SEX` does not remove the gap. The group gaps are small (a few points) and about as large as the seed-to-seed std, so the numbers here do not show that adding or removing `SEX` changes fairness reliably. Other columns can act as proxies for sex.
- In the data, the observed default rate differs by group (seed-0 test split: men 24.5%, women 20.5%). Equal flag rates therefore would not be calibrated to the base rates. Which fairness definition is appropriate is a policy question this repo does not answer.
- The per-sex threshold step lowers the flag-rate gap by construction and costs little F1 here. It is a description of a trade-off on this one split protocol, not a recommendation to use group-specific thresholds. Using sex that way in real lending is restricted in many places.
- Age groups are `<=29`, `30-39`, `>=40`. Age buckets are my choice and the gaps depend on them.
- Thresholds maximise F1 on validation data. Nothing is tuned on the test split. Hyperparameters are fixed in advance.
- Labels are "default payment next month" as given by the dataset. There is no information on how the labels were produced or on selection into the portfolio.

## Data
Not included. `make run` downloads the archive from the UCI repository and checks its SHA-256 (`56c885...727602`) before use. Dataset: Yeh, I. C., and Lien, C. H. (2009), "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients", Expert Systems with Applications 36(2). UCI lists it under CC BY 4.0. Code here is MIT only.

## Run
```
pip install -r requirements.txt
make test     # 7 tests; downloads the dataset once
make smoke    # 2,000-row, 1-seed run
make run      # full run, rewrites results/
```

## Layout
`src/creditaudit/` data, audit, run | `tests/` | `results/` committed output of the last full run | `.github/workflows/ci.yml`

## License
MIT, code only. Copyright Moustafa Mouhamed Zaalouk.
