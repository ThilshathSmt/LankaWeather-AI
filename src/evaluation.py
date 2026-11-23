"""Write a concise evidence-based analysis from measured holdout results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_analysis(results: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                   train_dates: pd.Series, test_dates: pd.Series) -> Path:
    """Create model_analysis.md using only metrics and class counts from this run."""
    ranked_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "balanced_accuracy"]
    best = {metric: results.loc[results[metric].idxmax(), "model"] for metric in ranked_metrics}
    baseline = results.loc[results["model"] == "Dummy Baseline"].iloc[0]
    stacking = results.loc[results["model"] == "Stacking"].iloc[0]
    hybrid = results.loc[results["model"] == "Hybrid Soft Voting"].iloc[0]
    lines = [
        "# LankaWeather AI: Model Analysis", "",
        "## Experiment design", "",
        f"Chronological 80/20 split. Training dates: {train_dates.iloc[0].date()} to {train_dates.iloc[-1].date()} "
        f"({len(train_dates):,} rows); test dates: {test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()} "
        f"({len(test_dates):,} rows). Models were evaluated on the same untouched test period.", "",
        "The stacked model used expanding-window forward-only out-of-fold predictions for its logistic regression "
        "meta learner. Gradient Boosting used balanced sample weights derived from training labels only.", "",
        "## Held-out metrics", "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Balanced accuracy | Train seconds | Predict seconds |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} "
            f"| {row['f1_score']:.4f} | {row['roc_auc']:.4f} | {row['balanced_accuracy']:.4f} "
            f"| {row['training_time_seconds']:.3f} | {row['prediction_time_seconds']:.3f} |"
        )
    lines.extend(["", "## Findings", ""])
    for metric, model in best.items():
        score = results.loc[results["model"] == model, metric].iloc[0]
        lines.append(f"- Highest {metric.replace('_', ' ')}: **{model}** ({score:.4f}).")
    lines.extend([
        f"- Training class counts: `{y_train.value_counts().sort_index().to_dict()}`; test class counts: "
        f"`{y_test.value_counts().sort_index().to_dict()}`.",
        f"- Stacking versus Random Forest: F1 {stacking['f1_score']:.4f} versus "
        f"{results.loc[results.model == 'Random Forest', 'f1_score'].iloc[0]:.4f}; ROC-AUC "
        f"{stacking['roc_auc']:.4f} versus {results.loc[results.model == 'Random Forest', 'roc_auc'].iloc[0]:.4f}.",
        f"- Hybrid versus Random Forest: F1 {hybrid['f1_score']:.4f} versus "
        f"{results.loc[results.model == 'Random Forest', 'f1_score'].iloc[0]:.4f}; ROC-AUC "
        f"{hybrid['roc_auc']:.4f} versus {results.loc[results.model == 'Random Forest', 'roc_auc'].iloc[0]:.4f}.",
        f"- The most-frequent baseline accuracy was {baseline['accuracy']:.4f}; compare it with balanced accuracy, "
        "recall, and ROC-AUC because raw accuracy can obscure class-specific errors.",
        "", "## Limitations and interpretation", "",
        "This is a single held-out time period at one Colombo location, based on Open-Meteo reanalysis data. "
        "Chronological evaluation is more realistic than random splitting, but performance may vary across seasons, "
        "locations, and later periods. The split does not measure geographic transfer. The target marks whether "
        "following-day precipitation exceeds 1 mm; it is a binary event target, not a rainfall amount forecast.", "",
        "The date-derived calendar predictors and recent weather may help capture seasonal persistence. Any metric "
        "differences among ensembles are empirical for this test period and do not establish a general advantage "
        "for stacking or voting. Hyperparameters were not tuned against the test set. The rainfall class proportions "
        "and each model's precision/recall trade-off should guide interpretation.", "",
        "## Conclusion", "",
        f"For this split, the best observed model by F1 was **{best['f1_score']}** and by ROC-AUC was **{best['roc_auc']}**. "
        "Use the complete metric table and confusion matrices alongside these rankings; no model is established as "
        "universally superior by this single experiment.", "",
    ])
    output = PROJECT_ROOT / "results" / "model_analysis.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    """Evaluation is executed as part of src/train_models.py."""
    raise SystemExit("Run the full evaluation with: python src/train_models.py")


if __name__ == "__main__":
    main()
