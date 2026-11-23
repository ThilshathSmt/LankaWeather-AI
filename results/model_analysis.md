# LankaWeather AI: Model Analysis

## Experiment design

Chronological 80/20 split. Training dates: 2016-01-01 to 2024-09-17 (3,183 rows); test dates: 2024-09-18 to 2026-11-22 (796 rows). Models were evaluated on the same untouched test period.

The stacked model used expanding-window forward-only out-of-fold predictions for its logistic regression meta learner. Gradient Boosting used balanced sample weights derived from training labels only.

## Held-out metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Balanced accuracy | Train seconds | Predict seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy Baseline | 0.7789 | 0.7789 | 1.0000 | 0.8757 | 0.5000 | 0.5000 | 0.070 | 0.001 |
| Random Forest | 0.8065 | 0.8552 | 0.9048 | 0.8793 | 0.7965 | 0.6825 | 3.137 | 0.426 |
| Gradient Boosting | 0.7701 | 0.9009 | 0.7919 | 0.8429 | 0.8125 | 0.7426 | 1.535 | 0.014 |
| Stacking | 0.7613 | 0.9072 | 0.7726 | 0.8345 | 0.8104 | 0.7471 | 26.665 | 0.545 |
| Hybrid Soft Voting | 0.7927 | 0.8785 | 0.8516 | 0.8649 | 0.8091 | 0.7184 | 4.817 | 0.542 |

## Findings

- Highest accuracy: **Random Forest** (0.8065).
- Highest precision: **Stacking** (0.9072).
- Highest recall: **Dummy Baseline** (1.0000).
- Highest f1 score: **Random Forest** (0.8793).
- Highest roc auc: **Gradient Boosting** (0.8125).
- Highest balanced accuracy: **Stacking** (0.7471).
- Training class counts: `{0: 781, 1: 2402}`; test class counts: `{0: 176, 1: 620}`.
- Stacking versus Random Forest: F1 0.8345 versus 0.8793; ROC-AUC 0.8104 versus 0.7965.
- Hybrid versus Random Forest: F1 0.8649 versus 0.8793; ROC-AUC 0.8091 versus 0.7965.
- The most-frequent baseline accuracy was 0.7789; compare it with balanced accuracy, recall, and ROC-AUC because raw accuracy can obscure class-specific errors.

## Limitations and interpretation

This is a single held-out time period at one Colombo location, based on Open-Meteo reanalysis data. Chronological evaluation is more realistic than random splitting, but performance may vary across seasons, locations, and later periods. The split does not measure geographic transfer. The target marks whether following-day precipitation exceeds 1 mm; it is a binary event target, not a rainfall amount forecast.

The date-derived calendar predictors and recent weather may help capture seasonal persistence. Any metric differences among ensembles are empirical for this test period and do not establish a general advantage for stacking or voting. Hyperparameters were not tuned against the test set. The rainfall class proportions and each model's precision/recall trade-off should guide interpretation.

## Conclusion

For this split, the best observed model by F1 was **Random Forest** and by ROC-AUC was **Gradient Boosting**. Use the complete metric table and confusion matrices alongside these rankings; no model is established as universally superior by this single experiment.
