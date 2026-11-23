"""Streamlit dashboard for exploring LankaWeather AI data and trained models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    # Custom estimators used by the serialized stacking and voting artifacts.
    sys.path.insert(0, str(SRC_PATH))
DATA_PATH = ROOT / "data" / "processed" / "weather_features.csv"
RAW_PATH = ROOT / "data" / "raw" / "sri_lanka_weather.csv"
METRICS_PATH = ROOT / "results" / "model_comparison.csv"
FIGURES_PATH = ROOT / "results" / "figures"
MODEL_PATHS = {
    "Random Forest": ROOT / "models" / "random_forest.pkl",
    "Gradient Boosting": ROOT / "models" / "gradient_boosting.pkl",
    "Stacking": ROOT / "models" / "stacking_model.pkl",
    "Hybrid Soft Voting": ROOT / "models" / "hybrid_model.pkl",
}
METRIC_COLUMNS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1-score": "f1_score",
    "ROC-AUC": "roc_auc",
    "Balanced accuracy": "balanced_accuracy",
}
CHART_TEMPLATE = "plotly_dark"
PALETTE = ["#42d392", "#45b8ff", "#a78bfa", "#fbbf24", "#fb7185"]

st.set_page_config(page_title="LankaWeather AI", page_icon="🌦️", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at 78% 0%, #102c37 0%, #09141d 38%, #071019 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d2029, #0a151d); border-right: 1px solid #1b3842; }
    .block-container { padding-top: 1.7rem; max-width: 1500px; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #21414b; border-radius: 20px;
            background: linear-gradient(115deg, rgba(28,91,80,.44), rgba(24,68,104,.35)); margin-bottom: 1.1rem; }
    .hero h1 { margin: 0; letter-spacing: -.04em; font-size: clamp(2rem, 5vw, 3.2rem); }
    .hero p { color: #b3c7d2; margin: .55rem 0 0; }
    .eyebrow { color: #55d9a2; text-transform: uppercase; font-size: .76rem; letter-spacing: .16em; font-weight: 700; }
    div[data-testid="stMetric"] { background: rgba(17,37,47,.78); border: 1px solid #1d3a45; padding: 1rem; border-radius: 15px; }
    div[data-testid="stMetricLabel"] { color: #aec3ce; }
    .note { border-left: 3px solid #45b8ff; padding: .7rem 1rem; background: rgba(26,68,91,.22); border-radius: 0 10px 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    """Load processed observations and parse the date column once per cache key."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    if "date" not in frame:
        raise ValueError("Processed dataset has no date column.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the raw daily weather table for field-level exploration."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")
    frame = pd.read_csv(RAW_PATH)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Load actual held-out metrics from the model training run."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Model comparison results not found: {METRICS_PATH}")
    return pd.read_csv(METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any]:
    """Read exact predictor and encoded column names saved during training."""
    path = ROOT / "models" / "training_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Training metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading selected trained model…")
def load_model(path_text: str) -> Any:
    """Cache a joblib model by path; raise a clear error for absent artifacts."""
    model_path = Path(path_text)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    return joblib.load(model_path)


def make_plot(fig: go.Figure) -> None:
    """Apply shared responsive chart layout and display a Plotly figure."""
    fig.update_layout(template=CHART_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15, r=15, t=55, b=15),
                      legend_title_text="")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})


def section_header(title: str, subtitle: str) -> None:
    """Show a consistent heading for the current navigation section."""
    st.markdown(f"<div class='eyebrow'>LankaWeather AI / {title}</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def available_feature_data() -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    """Load core dashboard data and convert recoverable file errors to a message."""
    try:
        data = load_processed_data()
        metadata = load_metadata()
        return data, metadata, None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return pd.DataFrame(), {}, str(exc)


def show_overview(data: pd.DataFrame, metadata: dict[str, Any], metrics: pd.DataFrame) -> None:
    """Render project summary, dataset facts, class balance, and top trained models."""
    section_header("Overview", "A compact view of the Colombo dataset and measured model performance.")
    st.markdown(
        "<div class='hero'><div class='eyebrow'>University machine learning assignment · Colombo, Sri Lanka</div>"
        "<h1>Weather patterns, measured honestly.</h1><p>Explore historical weather features and compare four trained "
        "ensemble classifiers against a simple baseline.</p></div>", unsafe_allow_html=True,
    )
    target = metadata.get("target")
    if data.empty or not target or target not in data:
        st.error("Dataset or target metadata is unavailable. Check the processed CSV and training metadata.")
        return
    n_rainy = int((data[target] == 1).sum())
    n_dry = int((data[target] == 0).sum())
    start, end = data["date"].min().date(), data["date"].max().date()
    trained = metrics[metrics["model"] != "Dummy Baseline"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily records", f"{len(data):,}")
    c2.metric("Date coverage", f"{start} → {end}")
    c3.metric("Rainy next days", f"{n_rainy:,}", f"{n_rainy / len(data):.1%}")
    c4.metric("Non-rainy next days", f"{n_dry:,}", f"{n_dry / len(data):.1%}")
    try:
        raw = load_raw_data()
        st.caption(f"Raw source table: {len(raw):,} records · {RAW_PATH.relative_to(ROOT)}")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.caption(f"Raw source table is unavailable: {exc}")

    st.subheader("Best trained model by metric")
    columns = st.columns(3)
    for column, (label, metric_name) in zip(columns, [("Accuracy", "accuracy"), ("F1-score", "f1_score"), ("ROC-AUC", "roc_auc")]):
        if metric_name in trained and not trained.empty:
            winner = trained.loc[trained[metric_name].idxmax()]
            column.metric(label, f"{winner[metric_name]:.3f}", str(winner["model"]))
        else:
            column.metric(label, "Unavailable")
    dist = pd.DataFrame({"Outcome": ["Non-rainy", "Rainy"], "Days": [n_dry, n_rainy]})
    make_plot(px.pie(dist, names="Outcome", values="Days", hole=.62, color="Outcome",
                     color_discrete_map={"Rainy": PALETTE[0], "Non-rainy": PALETTE[1]}, title="Next-day rainfall target distribution"))
    st.markdown("<div class='note'>The target records whether next-day precipitation exceeds 1 mm. Model scores below come from the saved chronological test evaluation.</div>", unsafe_allow_html=True)


def show_explorer(data: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Render filtered history, weather trends, monthly rainfall, correlation, and table."""
    section_header("Dataset Explorer", "Filter the historical observations and inspect weather patterns.")
    if data.empty:
        st.error("Processed dataset is unavailable.")
        return
    lo, hi = data["date"].min().date(), data["date"].max().date()
    selected = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi, key="explorer_dates")
    if isinstance(selected, tuple) and len(selected) == 2:
        view = data[data["date"].dt.date.between(selected[0], selected[1])].copy()
    else:
        view = data[data["date"].dt.date == selected].copy()
    st.caption(f"Showing {len(view):,} of {len(data):,} daily records.")

    temperature_cols = [c for c in ("temperature_2m_max", "temperature_2m_mean", "temperature_2m_min") if c in view]
    if temperature_cols and not view.empty:
        long = view.melt("date", value_vars=temperature_cols, var_name="Measure", value_name="Temperature (°C)")
        long["Measure"] = long["Measure"].str.replace("temperature_2m_", "", regex=False).str.title()
        make_plot(px.line(long, x="date", y="Temperature (°C)", color="Measure", color_discrete_sequence=PALETTE, title="Daily temperature trends"))
    rainfall_col = "precipitation_sum" if "precipitation_sum" in view else None
    if rainfall_col and not view.empty:
        left, right = st.columns(2)
        with left:
            make_plot(px.area(view, x="date", y=rainfall_col, color_discrete_sequence=[PALETTE[1]], title="Daily precipitation"))
        with right:
            monthly = view.assign(month_start=view["date"].dt.to_period("M").dt.to_timestamp()).groupby("month_start", as_index=False)[rainfall_col].sum()
            monthly["month"] = monthly["month_start"].dt.strftime("%Y-%m")
            make_plot(px.bar(monthly, x="month", y=rainfall_col, color_discrete_sequence=[PALETTE[0]], title="Monthly rainfall total"))
    if "relative_humidity_2m" in view and not view.empty:
        make_plot(px.line(view, x="date", y="relative_humidity_2m", color_discrete_sequence=[PALETTE[2]], title="Daily mean relative humidity"))
    predictors = metadata.get("predictors", [])
    numeric = [c for c in predictors if c in view and pd.api.types.is_numeric_dtype(view[c])]
    if len(numeric) > 1 and not view.empty:
        corr = view[numeric].corr()
        make_plot(px.imshow(corr, text_auto=".2f", color_continuous_scale="Tealrose", title="Predictor correlations"))
    st.subheader("Filtered observations")
    st.dataframe(view, width="stretch", hide_index=True)


def show_comparison(metrics: pd.DataFrame) -> None:
    """Compare actual saved metrics and distinguish the non-trained baseline."""
    section_header("Model Comparison", "Held-out scores from the saved chronological evaluation.")
    expected = {"model", *METRIC_COLUMNS.values()}
    missing = expected.difference(metrics.columns)
    if missing:
        st.error(f"Metrics file is missing columns: {sorted(missing)}")
        return
    metrics = metrics.copy()
    metrics["Model type"] = metrics["model"].apply(lambda name: "Baseline" if name == "Dummy Baseline" else "Trained ensemble")
    selected_metrics = st.multiselect("Metrics", list(METRIC_COLUMNS), default=list(METRIC_COLUMNS), key="comparison_metrics")
    plotted = metrics.melt(id_vars=["model", "Model type"], value_vars=[METRIC_COLUMNS[x] for x in selected_metrics],
                           var_name="Metric", value_name="Score")
    label_map = {v: k for k, v in METRIC_COLUMNS.items()}
    plotted["Metric"] = plotted["Metric"].map(label_map)
    if not plotted.empty:
        make_plot(px.bar(plotted, x="model", y="Score", color="Metric", barmode="group", facet_col="Metric",
                         facet_col_wrap=3, color_discrete_sequence=PALETTE, title="Model metrics on the shared test period"))
    baseline, trained = st.tabs(["Baseline", "Trained ensembles"])
    with baseline:
        st.caption("DummyClassifier(strategy='most_frequent') is a reference baseline, not a trained weather model.")
        st.dataframe(metrics[metrics["Model type"] == "Baseline"].drop(columns="Model type"), width="stretch", hide_index=True)
    with trained:
        st.dataframe(metrics[metrics["Model type"] == "Trained ensemble"].drop(columns="Model type"), width="stretch", hide_index=True)
    if "classification_report" in metrics:
        with st.expander("Classification reports"):
            for _, row in metrics.iterrows():
                st.markdown(f"**{row['model']}**")
                try:
                    report = json.loads(row["classification_report"])
                    st.dataframe(pd.DataFrame(report).T, width="stretch")
                except (TypeError, json.JSONDecodeError):
                    st.warning("A saved classification report could not be parsed.")


def show_evaluation() -> None:
    """Display all saved evaluation plots grouped by artifact type."""
    section_header("Model Evaluation", "Confusion matrices and diagnostic plots generated by the training run.")
    if not FIGURES_PATH.exists():
        st.warning(f"Evaluation figure folder is missing: {FIGURES_PATH}")
        return
    images = sorted(FIGURES_PATH.glob("*.png"))
    if not images:
        st.info("No evaluation figures have been generated yet.")
        return
    groups = [
        ("Confusion matrices", [p for p in images if p.name.startswith("confusion_matrix_")]),
        ("ROC curve", [p for p in images if "roc" in p.name.lower()]),
        ("Precision–recall curve", [p for p in images if "precision_recall" in p.name.lower()]),
        ("Feature importance", [p for p in images if "feature_importance" in p.name.lower()]),
    ]
    for title, paths in groups:
        if not paths:
            continue
        st.subheader(title)
        cols = st.columns(2)
        for index, path in enumerate(paths):
            with cols[index % 2]:
                st.image(str(path), caption=path.stem.replace("_", " ").title(), width="stretch")


def _build_prediction_input(data: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Collect predictor values and encode them in the saved training schema/order."""
    predictors = metadata.get("predictors", [])
    encoded_columns = metadata.get("encoded_predictors", [])
    if not predictors or not encoded_columns:
        raise ValueError("Training metadata does not define predictor columns.")
    latest = data.iloc[-1]
    st.markdown("Enter conditions for the observation day. Calendar predictors are derived from the selected date.")
    weather_fields = [name for name in predictors if name in data and pd.api.types.is_numeric_dtype(data[name])
                      and name not in {"year", "month", "day", "day_of_week", "day_of_year"}]
    cols = st.columns(2)
    values: dict[str, Any] = {}
    for index, feature in enumerate(weather_fields):
        latest_value = float(latest[feature])
        with cols[index % 2]:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(), value=latest_value,
                help=f"Default is the latest recorded value ({latest_value:g}).",
                key=f"prediction_{feature}",
            )
    observation_date = st.date_input(
        "Observation date", value=latest["date"].date(),
        help="Calendar features describe the day whose conditions are entered.", key="prediction_date",
    )
    dt = pd.Timestamp(observation_date)
    calendar_values = {
        "year": dt.year, "month": dt.month, "day": dt.day,
        "day_of_week": dt.day_of_week, "day_of_year": dt.day_of_year,
    }
    for name in predictors:
        if name in calendar_values:
            values[name] = calendar_values[name]
        elif name == "sri_lankan_season":
            season_values = sorted(data[name].dropna().astype(str).unique())
            season_default = str(latest[name]) if str(latest[name]) in season_values else season_values[0]
            values[name] = st.selectbox("Sri Lankan season", season_values, index=season_values.index(season_default), key="prediction_season")
        elif name not in values:
            raise ValueError(f"No prediction input is configured for saved predictor {name!r}.")

    raw_row = pd.DataFrame([{name: values[name] for name in predictors}], columns=predictors)
    categorical = raw_row.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    encoded = pd.get_dummies(raw_row, columns=categorical, dtype="int8")
    encoded = encoded.reindex(columns=encoded_columns, fill_value=0)
    return encoded.astype(float)


def show_prediction(data: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Render input controls, load the selected artifact, and show class/probability."""
    section_header("Rainfall Prediction", "Predict the next-day rain class using one of the saved ensemble models.")
    st.warning("Academic project tool only. This is not an official operational weather forecast.")
    if data.empty:
        st.error("Historical dataset is unavailable, so prediction defaults cannot be prepared.")
        return
    model_name = st.selectbox("Trained model", list(MODEL_PATHS), key="selected_prediction_model")
    with st.form("prediction_form"):
        try:
            input_frame = _build_prediction_input(data, metadata)
        except (ValueError, KeyError) as exc:
            st.error(f"Prediction form could not be prepared: {exc}")
            return
        submitted = st.form_submit_button("Predict next-day rainfall", type="primary")
    st.caption(f"Model input columns: {', '.join(input_frame.columns)}")
    if submitted:
        try:
            model = load_model(str(MODEL_PATHS[model_name]))
            prediction = int(model.predict(input_frame)[0])
            label = "Rain likely (>1 mm)" if prediction == 1 else "No rain expected (≤1 mm)"
            st.success(f"Predicted class: **{label}**")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_frame)[0]
                classes = list(model.classes_)
                positive_index = classes.index(1)
                probability = float(probabilities[positive_index])
                st.metric("Probability of next-day rain", f"{probability:.1%}")
                st.progress(probability)
            with st.expander("Submitted predictor values"):
                st.dataframe(input_frame, width="stretch", hide_index=True)
        except (OSError, ValueError, AttributeError, KeyError) as exc:
            st.error(f"Prediction failed: {exc}")


def show_project_info() -> None:
    """Explain dataset, ensemble approaches, and evaluation measures."""
    with st.expander("About this academic project", expanded=False):
        st.markdown(
            """
            **Location and data:** Colombo, Sri Lanka (6.9271° N, 79.8612° E). Daily historical weather variables and
            hourly humidity aggregates were collected from the Open-Meteo Historical Weather API. The binary target
            is whether next-day precipitation exceeds 1 mm.

            **Bagging — Random Forest:** trains many decision trees on bootstrap samples and combines their votes;
            class-balanced subsampling was used to account for label imbalance.

            **Boosting — Gradient Boosting:** adds shallow trees sequentially to improve the ensemble; training-only
            class sample weights address the observed class proportions.

            **Stacking:** combines Random Forest and Gradient Boosting probabilities. A logistic regression meta
            learner is fitted using expanding-window chronological out-of-fold predictions.

            **Hybridization:** soft voting averages class probabilities from Random Forest and Gradient Boosting.
            The `DummyClassifier` is shown as a simple most-frequent baseline, not as an ensemble.

            **Evaluation:** models share a chronological 80/20 split with no shuffling. Accuracy, precision, recall,
            F1-score, ROC-AUC, balanced accuracy, confusion matrices, classification reports, and runtime are saved
            in the project results. This is a single-location academic experiment, not a validated operational forecast.
            """
        )


def main() -> None:
    """Load project assets and route to the selected dashboard section."""
    st.sidebar.markdown("## 🌦️ LankaWeather AI")
    st.sidebar.caption("Ensemble learning · Colombo, Sri Lanka")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Dataset Explorer", "Model Comparison", "Model Evaluation", "Rainfall Prediction"],
        label_visibility="collapsed",
    )
    try:
        data = load_processed_data()
        metadata = load_metadata()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load required project data: {exc}")
        return
    try:
        metrics = load_metrics()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load saved model metrics: {exc}")
        metrics = pd.DataFrame()

    if page == "Overview":
        show_overview(data, metadata, metrics)
    elif page == "Dataset Explorer":
        show_explorer(data, metadata)
    elif page == "Model Comparison":
        show_comparison(metrics)
    elif page == "Model Evaluation":
        show_evaluation()
    else:
        show_prediction(data, metadata)
    show_project_info()


if __name__ == "__main__":
    main()
