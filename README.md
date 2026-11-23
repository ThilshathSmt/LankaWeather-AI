# LankaWeather AI

LankaWeather AI is a university machine learning assignment that explores next-day rainfall classification for Colombo, Sri Lanka. The project collects historical weather data from Open-Meteo, prepares leakage-aware features, compares four ensemble classifiers with a simple baseline, evaluates them on a chronological holdout period, and presents the data and saved results in a Streamlit dashboard.

> **Scope:** This is an academic experiment, not an official or operational weather forecast. Historical values come from Open-Meteo Historical Weather API reanalysis products. They are not independently collected ground-station measurements.

## Problem statement

Rainfall affects transport, agriculture, water management, and everyday planning. This project studies whether recent daily weather and calendar features can classify whether precipitation on the following day will exceed 1 mm at one Colombo location. It frames the assignment as a binary classification problem; it does not predict rainfall amount or provide a geographically general Sri Lankan forecast.

## Objectives

- Collect and inspect historical daily weather data for Colombo.
- Create calendar features and a next-day rainfall target without using future-day measurements as predictors.
- Compare Bagging, Boosting, Stacking, and soft-voting Hybridization against a Dummy baseline.
- Evaluate all models on the same chronological 20% test period.
- Present the saved data, trained models, measured results, and evaluation plots in an interactive dashboard.

## Location and Sri Lankan weather context

The selected point is Colombo, Sri Lanka, at **latitude 6.9271° N and longitude 79.8612° E**, using the `Asia/Colombo` timezone for data collection. Sri Lanka has varied local weather and rainfall patterns; this single coastal location cannot represent the whole island.

The engineered `sri_lankan_season` is a simplified calendar feature: **Maha** covers October through March, and **Yala** covers April through September. It provides broad seasonal context for this assignment and is not a detailed meteorological classification of local rainfall regimes.

## Dataset and source

The data collection script requests the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) from **2016-01-01 through the latest date returned by the API**. The saved snapshot in this repository contains:

| Dataset | Records | Date coverage | Columns |
|---|---:|---|---:|
| Raw daily weather | 3,980 | 2016-01-01 to 2026-11-23 | 7 |
| Processed model features | 3,979 | 2016-01-01 to 2026-11-22 | 14 |

The final processed date is one day before the final raw observation because the last raw row has no observed next-day rainfall target. The raw CSV contains daily maximum, minimum and mean 2 m temperature, daily precipitation sum, maximum 10 m wind speed, and daily mean 2 m relative humidity aggregated from hourly values.

The processed table adds `year`, `month`, `day`, `day_of_week`, `day_of_year`, `sri_lankan_season`, and `next_day_rain`. Its target is **1 when the following calendar day's precipitation sum is greater than 1 mm; otherwise it is 0**. Records are sorted and deduplicated by date. Missing weather observations are not filled with invented values; rows without required weather values or a next-day target are omitted. Calendar predictors describe the observation day.

The processed target counts are **957 non-rainy (0; 24.05%)** and **3,022 rainy (1; 75.95%)**. This is the class distribution of the processed dataset, not a test-only distribution. On the chronological training period, counts were 781 non-rainy and 2,402 rainy; the test period had 176 non-rainy and 620 rainy examples.

### Model predictors

Training metadata identifies `next_day_rain` as the target and `date` as the ordering field, not a predictor. Saved model predictors are:

`temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`, `year`, `month`, `day`, `day_of_week`, `day_of_year`, and `sri_lankan_season`.

Training expands `sri_lankan_season` into `sri_lankan_season_Maha` and `sri_lankan_season_Yala`, producing 12 encoded model columns. Although `relative_humidity_2m` is present in the processed dataset and available in the explorer, it is absent from the saved predictor metadata. The current training code's broad identifier-name filter matches the letters `id` within `humidity` and excludes the column. The dashboard follows the saved metadata exactly; the existing models were not changed for this documentation update.

## Preprocessing and feature engineering

`src/data_collection.py` requests daily variables and hourly relative humidity, averages humidity by Colombo calendar date, validates the API response, retries transient HTTP failures, and writes `data/raw/sri_lanka_weather.csv`.

`src/preprocessing.py` parses and sorts dates, removes duplicate dates, creates calendar and season features, and uses a one-day shift to create the target only when the next row is the next calendar date. It drops the final row and rows missing required weather or target information. It does not impute missing observations.

The model trainer discovers the binary target from the dataset, excludes the date, target, identifier-like, and future-named fields, then sorts by date. Categorical values are one-hot encoded using training data categories, and test columns are aligned to training column order. No random shuffling is used.

## Modeling approaches

- **Bagging — Random Forest:** combines many decision trees. The model uses `class_weight="balanced_subsample"` to account for class imbalance in each tree's bootstrap sample.
- **Boosting — Gradient Boosting:** builds shallow trees sequentially. Balanced sample weights are calculated from training labels only.
- **Stacking:** combines Random Forest and Gradient Boosting class probabilities with a Logistic Regression meta-learner. Meta-learner training data comes from expanding-window chronological out-of-fold predictions; base learners are then refit on the full training set.
- **Hybridization — soft voting:** averages class probabilities from Random Forest and Gradient Boosting. The Gradient Boosting member uses balanced training-label sample weights.
- **Baseline — Dummy Classifier:** predicts the most frequent training class. It is a reference point, not a trained weather model.

## Train/test split and evaluation

Models were evaluated on a **chronological 80/20 split**, without shuffling. The training period is **2016-01-01 to 2024-09-17** (3,183 rows); the untouched test period is **2024-09-18 to 2026-11-22** (796 rows). Every model uses the same test set.

The saved evaluation reports Accuracy, Precision, Recall, F1-score, ROC-AUC calculated from probabilities, Balanced Accuracy, confusion matrices, classification reports, and training/prediction time. The dashboard and CSV use the actual saved measurements.

### Measured test results

Values below are read from `results/model_comparison.csv` and rounded to four decimal places for display.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Baseline | 0.7789 | 0.7789 | 1.0000 | 0.8757 | 0.5000 | 0.5000 |
| Random Forest | 0.8065 | 0.8552 | 0.9048 | 0.8793 | 0.7965 | 0.6825 |
| Gradient Boosting | 0.7701 | 0.9009 | 0.7919 | 0.8429 | 0.8125 | 0.7426 |
| Stacking | 0.7613 | 0.9072 | 0.7726 | 0.8345 | 0.8104 | 0.7471 |
| Hybrid Soft Voting | 0.7927 | 0.8785 | 0.8516 | 0.8649 | 0.8091 | 0.7184 |

### Findings: measurements and interpretation

**Measured findings** from the held-out period:

- Random Forest had the highest accuracy (**0.8065**) and F1 (**0.8793**) among the trained models.
- Gradient Boosting had the highest ROC-AUC (**0.8125**).
- Stacking had the highest precision (**0.9072**) and balanced accuracy (**0.7471**).
- Hybrid Soft Voting's F1 (**0.8649**) did not exceed Random Forest's (**0.8793**).
- The Dummy Baseline had 0.7789 accuracy and 1.0000 recall, but also 0.5000 balanced accuracy and 0.5000 ROC-AUC.

**Interpretation:** the target is imbalanced, with rainy days comprising 75.95% of all processed examples and 77.89% of the test period. The most-frequent baseline predicts every example as rainy, so its accuracy reflects the majority-class prevalence and its recall is perfect for that class, while it has no ability to distinguish dry days. Its accuracy and F1 should not be read as evidence of useful discrimination; balanced accuracy and ROC-AUC expose that limitation.

Stacking and Hybrid Soft Voting did not improve F1 over Random Forest on this test interval, although their ROC-AUC values were slightly higher than Random Forest's (0.8104 and 0.8091 versus 0.7965). This is an observed result for this split, not proof that either ensemble method is generally better or worse. Class-weighting, the chronological period, selected features, and model hyperparameters may affect the precision/recall balance; the experiment does not establish which factor explains the differences.

See [`results/model_analysis.md`](results/model_analysis.md) for the complete measured analysis, timing, and discussion.

## Dashboard

The Streamlit application has five sections:

1. **Overview:** Colombo project summary, dataset coverage and size, class distribution, and best trained model by accuracy, F1, and ROC-AUC.
2. **Dataset Explorer:** date filters, temperature and rainfall trends, monthly rainfall totals, humidity, predictor correlations, and an interactive data table.
3. **Model Comparison:** saved metrics for all models, with the Dummy baseline separated from trained ensembles.
4. **Model Evaluation:** saved confusion matrices, ROC and precision-recall curves, and tree-based feature importance plots.
5. **Rainfall Prediction:** selects one of the four saved models and provides default inputs from the latest observation. Calendar values are derived from the selected observation date; `date` and the target are not model inputs. Predictions are academic demonstrations, not official forecasts.

Launch from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Technology stack

Python, pandas, NumPy, scikit-learn, joblib, Requests, Matplotlib, Plotly, Streamlit, Jupyter, and IPython. The repository's `requirements.txt` records the installed Python package versions used by this project.

## Repository structure

```text
LankaWeather-AI/
├── data/
│   ├── raw/sri_lanka_weather.csv
│   └── processed/weather_features.csv
├── dashboard/app.py
├── models/
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── stacking_model.pkl
│   ├── hybrid_model.pkl
│   └── training_metadata.json
├── notebooks/ensemble_analysis.ipynb
├── results/
│   ├── figures/                 # Confusion matrices, ROC/PR and importance plots
│   ├── model_analysis.md
│   └── model_comparison.csv
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── ensemble_estimators.py
│   └── evaluation.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Windows installation

Clone the repository, create a virtual environment, activate it in PowerShell, and install dependencies:

```powershell
git clone YOUR_REPOSITORY_URL
cd LankaWeather-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's Python executable directly as shown in the execution commands below.

## Execution commands

Run these from the repository root. Data collection contacts Open-Meteo and overwrites the raw CSV with a fresh API snapshot. Preprocessing overwrites the processed CSV. Training overwrites the four model files, metadata, comparison CSV, plots, and analysis report; it is listed for reproducibility and is not needed to launch the existing dashboard.

```powershell
# Collect current historical API data
.\.venv\Scripts\python.exe src\data_collection.py

# Build processed features and the next-day target
.\.venv\Scripts\python.exe src\preprocessing.py

# Train/evaluate models and regenerate saved artifacts and results
.\.venv\Scripts\python.exe src\train_models.py

# Launch the dashboard using existing saved artifacts
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The evaluation report and plots are produced by `src/train_models.py`; `src/evaluation.py` contains the report-writing function. There is no standalone evaluation run required.

### Notebook instructions

Open `notebooks/ensemble_analysis.ipynb` in VS Code or Jupyter after installing requirements. Run cells from the repository root. The notebook includes dataset inspection, target/predictor discovery, exploratory class distribution, pipeline execution, metric review, analysis, and saved figures. **Its pipeline cell calls `run_pipeline()` and therefore retrains models and overwrites model and result artifacts.** Skip that cell if you only want to review the already generated results.

## Limitations and future improvements

- The data covers one Colombo coordinate and is based on Open-Meteo historical/reanalysis products; the result does not represent station observations or all Sri Lankan regions.
- Evaluation uses one chronological holdout. Seasonal drift, other locations, and future periods need separate evaluation.
- The processed target is binary rainfall occurrence above a fixed 1 mm threshold, not a precipitation amount forecast.
- The target is imbalanced, and the Dummy baseline demonstrates why accuracy alone is insufficient.
- The saved model feature schema currently omits relative humidity even though the feature exists in the processed data; see the model predictors note above.
- Model hyperparameters were not tuned against the test period. Calibration, threshold selection, and repeated rolling-origin evaluation could be explored in future work.
- Future improvements include a verified station-data comparison, multiple Sri Lankan locations, clearer geographic/seasonal validation, probability calibration, uncertainty communication, and a carefully tested feature-selection pipeline.

## Student Contributions and Project Ownership

### 1. Problem Identification and Project Scope

I adopted Sri Lankan weather prediction as a locally meaningful real-world context for studying ensemble learning, and focused the project on classifying whether Colombo would receive more than 1 mm of precipitation on the next day. Initial project ideation and architecture suggestions were AI-assisted. I set and refined the assignment scope, selected the project direction from those discussions, and directed the work through its data, modeling, evaluation, and dashboard stages. I do not present AI-assisted ideation as independent discovery.

### 2. Dataset Selection and Justification

I chose Colombo historical weather as the study setting because it gives the assignment a concrete Sri Lankan location and a time-indexed dataset suited to next-day classification. Daily temperature, precipitation, wind, and humidity fields provide weather context; the next-day threshold creates a clear binary outcome. The temporal ordering also makes chronological splitting and leakage prevention essential. Open-Meteo was used as the historical source during the AI-assisted project planning and development process; I do not claim that I independently discovered the dataset or API. The 1 mm threshold defines the assignment's target and should not be interpreted as an official rainfall warning criterion. The downloaded history consists of Open-Meteo historical/reanalysis values rather than independent ground-station measurements.

### 3. System Architecture and Technical Decisions

I coordinated a staged design that separates API collection, preprocessing, model training and evaluation, saved artifacts, and the Streamlit dashboard. The pipeline collects weather records, parses and orders dates, creates calendar features and the simplified Maha/Yala season label, and constructs the next-day target using a one-day shift. The target and date are excluded from model inputs. A chronological 80/20 split was selected to retain time order and evaluate on a later period.

I adopted four complementary ensemble approaches for comparison: Random Forest for Bagging, Gradient Boosting, Stacking with Random Forest and Gradient Boosting probabilities feeding a Logistic Regression meta-learner, and a soft-voting Hybrid that averages probabilities from the two tree ensembles. Stacking uses expanding-window chronological out-of-fold predictions. The models and training metadata are serialized with Joblib, and Streamlit was selected to expose the dataset, saved measurements, figures, and model predictions through a local interactive interface. ChatGPT and Codex assisted substantially with the architecture and implementation; these choices were reviewed and directed as part of the project, not independently coded in full by me.

### 4. Experimental Analysis and Critical Evaluation

The following are **measured results** from the shared chronological test set, as recorded in `results/model_comparison.csv` and `results/model_analysis.md`:

- Random Forest had the highest accuracy (**0.8065**) and F1 (**0.8793**) among the trained models.
- Gradient Boosting had the highest ROC-AUC (**0.8125**).
- Stacking had the highest precision (**0.9072**) and balanced accuracy (**0.7471**).
- Hybrid Soft Voting had F1 **0.8649**, below Random Forest's **0.8793**.
- Dummy Baseline accuracy was **0.7789**, with recall **1.0000**, balanced accuracy **0.5000**, and ROC-AUC **0.5000**.

**Possible explanations, not proven causes:** the target is majority-rain (620 rainy versus 176 non-rainy cases in the test set), and the Dummy Baseline predicts only the majority class. Its accuracy therefore largely reflects class prevalence; perfect recall does not mean it identifies rainy and non-rainy cases well, as its balanced accuracy and ROC-AUC show. The ensembles make different class decisions and probability rankings, and class weighting can affect their precision/recall trade-offs. These factors may help explain the measured differences, but this single test period does not establish why each model scored as it did or which approach will generalize best elsewhere.

### 5. AI-Assisted Development and Student Involvement

ChatGPT assisted with initial ideation, planning, explanations, architecture suggestions, and debugging. OpenAI Codex assisted substantially with project structure and Python implementation, including preprocessing, model implementation, evaluation, and dashboard development. I coordinated staged development, adopted and reviewed design decisions, executed the project workflows, examined the dataset and experimental results, and ran and tested the dashboard. I did not independently implement all generated code and do not claim independent authorship of AI-generated implementation. I remain responsible for understanding the workflow, critically checking the outputs, and presenting the AI assistance transparently in accordance with course requirements.

### 6. Learning Outcomes and Technical Understanding

Working through and evaluating this project helped me understand how to collect historical weather data from an API and why reanalysis estimates differ from station observations; how to parse and preprocess daily records; and how calendar and season features support a classification task. The target shift provided a concrete example of why future observations must be kept out of predictors to avoid leakage. Sorting by date and splitting chronologically showed how evaluation changes when the test period follows the training period, while expanding-window out-of-fold predictions demonstrated a time-aware way to train a stacking meta-learner.

I also developed understanding of how Bagging reduces dependence on an individual tree through a forest, how Boosting builds trees sequentially, how Stacking learns from base-model probabilities, and how Hybrid soft voting combines probabilities. The imbalanced test distribution illustrated why accuracy should be read alongside precision, recall, F1, ROC-AUC, and balanced accuracy. Finally, the saved Joblib models and their metadata showed how model serialization connects training code to a Streamlit prediction interface, where input names, types, and order must match the trained feature schema.

The four ensemble methods produced different results because they learn and combine their tree predictions differently; the observed precision, recall, ranking, and class-balance scores therefore varied on this test period. That is a high-level interpretation of the measured comparisons, not a causal conclusion. A longer holdout, additional Sri Lankan locations, and further validation would be needed to assess whether those differences persist.

## References

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Plotly Python documentation](https://plotly.com/python/)
