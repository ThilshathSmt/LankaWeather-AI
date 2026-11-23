# LankaWeather AI

LankaWeather AI is a university machine learning assignment scaffold for exploring weather prediction in Sri Lanka. The repository is organized to support data collection, preprocessing, ensemble model training, evaluation, and a Streamlit dashboard.

> **Project status:** Initial scaffold. The raw data file is empty, model training and evaluation are not implemented, and no model artifacts or experimental results have been created.

## Project structure

```text
LankaWeather-AI/
├── data/
│   ├── raw/                      # Source data; currently an empty CSV placeholder
│   └── processed/                # Intended for validated, transformed datasets
├── notebooks/                    # Exploratory analysis and assignment work
├── models/                       # Trained model artifacts (created after training)
├── src/                          # Data and modeling workflow modules
├── dashboard/                    # Streamlit application
├── results/
│   ├── figures/                  # Generated plots and visualizations
│   └── model_comparison.csv       # Results schema; currently contains no measurements
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview and setup
└── .gitignore                    # Generated files and local configuration exclusions
```

## Setup

Use Python 3.10 or newer, then create and activate a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Current workflow status

1. Add the selected Sri Lankan weather dataset to `data/raw/sri_lanka_weather.csv` and document its source and license.
2. Implement and run the data collection and preprocessing steps in `src/`.
3. Train the assignment's models and write artifacts to `models/` only after successful training.
4. Evaluate those trained models and record measured results in `results/model_comparison.csv`.
5. Extend `notebooks/ensemble_analysis.ipynb` and `dashboard/app.py` to use the real data and trained artifacts.

The source modules intentionally contain scaffolding only. No model definitions, predictions, scores, or sample results are included.

## Run the dashboard

Once the dashboard has been extended to use real data and trained artifacts, launch it with:

```bash
streamlit run dashboard/app.py
```

At this initial stage, it displays a project status message and does not make predictions.