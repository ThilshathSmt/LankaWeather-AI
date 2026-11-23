"""Build leakage-safe calendar features and a next-day rain target."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sri_lanka_weather.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "weather_features.csv"
DATE_COLUMN = "date"
PRECIPITATION_COLUMN = "precipitation_sum"
TARGET_COLUMN = "next_day_rain"
WEATHER_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m",
]
REQUIRED_WEATHER_COLUMNS = [column for column in WEATHER_COLUMNS if column != "relative_humidity_2m"]


def preprocess_weather(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Prepare data without imputing missing observations or using future features.

    Sri Lankan season is a simplified calendar feature: Maha spans October to
    March and Yala spans April to September. This labels broad cultivation /
    monsoon seasons and is not a claim that local rainfall has only two seasons.
    Rows with missing weather values or missing next-day rainfall are excluded;
    no weather values are filled or synthesized.
    """
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise ValueError(f"Raw weather CSV is missing or empty: {raw_path}")
    frame = pd.read_csv(raw_path)
    required = {DATE_COLUMN, *REQUIRED_WEATHER_COLUMNS}
    absent = sorted(required.difference(frame.columns))
    if absent:
        raise ValueError(f"Raw CSV is missing required columns: {absent}")

    input_records = len(frame)
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce").dt.normalize()
    invalid_dates = int(frame[DATE_COLUMN].isna().sum())
    frame = frame.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    duplicates = int(frame.duplicated(DATE_COLUMN).sum())
    frame = frame.drop_duplicates(DATE_COLUMN, keep="first").reset_index(drop=True)
    for column in WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # This label uses only today's date. Target uses tomorrow's precipitation,
    # while all predictor columns retain same-day values only.
    frame["year"] = frame[DATE_COLUMN].dt.year
    frame["month"] = frame[DATE_COLUMN].dt.month
    frame["day"] = frame[DATE_COLUMN].dt.day
    frame["day_of_week"] = frame[DATE_COLUMN].dt.day_of_week
    frame["day_of_year"] = frame[DATE_COLUMN].dt.day_of_year
    frame["sri_lankan_season"] = frame["month"].map(
        lambda month: "Maha" if month >= 10 or month <= 3 else "Yala"
    )
    next_date_is_consecutive = frame[DATE_COLUMN].shift(-1).sub(frame[DATE_COLUMN]).eq(pd.Timedelta(days=1))
    next_day_precipitation = frame[PRECIPITATION_COLUMN].shift(-1).where(next_date_is_consecutive)
    frame[TARGET_COLUMN] = (next_day_precipitation > 1.0).astype("Int64")
    # Last row has no observed following day. Do not infer it.
    frame = frame.iloc[:-1].copy()

    missing_before_drop = frame[WEATHER_COLUMNS + [TARGET_COLUMN]].isna().sum()
    rows_before_missing_drop = len(frame)
    # Humidity is optional in the requested source. Keep its missing values
    # visible for downstream decisions; exclude rows missing core daily data.
    frame = frame.dropna(subset=REQUIRED_WEATHER_COLUMNS + [TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
    frame = frame.reset_index(drop=True)

    # Assert target and predictor provenance explicitly for this fixed schema.
    assert set(frame[TARGET_COLUMN].unique()).issubset({0, 1})
    predictor_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    assert "next_day_precipitation" not in predictor_columns
    assert "next_day_rain" not in predictor_columns
    if not frame.empty:
        assert frame[DATE_COLUMN].is_monotonic_increasing

    print(f"Input records: {input_records}")
    print(f"Invalid dates removed: {invalid_dates}; duplicate dates removed: {duplicates}")
    print(f"Rows removed for missing weather/target values: {rows_before_missing_drop - len(frame)}")
    print(f"Output records: {len(frame)}")
    if not frame.empty:
        print(f"Output date range: {frame[DATE_COLUMN].min().date()} to {frame[DATE_COLUMN].max().date()}")
    print("Missing values in retained features:")
    print(frame.isna().sum().to_string())
    counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    percentages = counts.div(len(frame)).mul(100) if len(frame) else counts.astype(float)
    print("Target class counts and percentages:")
    print(pd.DataFrame({"count": counts, "percent": percentages.round(2)}).to_string())
    print(f"Target contains only 0/1: {set(frame[TARGET_COLUMN].unique()).issubset({0, 1})}")
    print("Predictors use current-day weather and calendar fields only; target is next-day rainfall.")
    print("Missing weather counts before row removal:")
    print(missing_before_drop.to_string())
    return frame


def main() -> None:
    """Preprocess raw weather records and save the feature table."""
    try:
        features = preprocess_weather()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Preprocessing stopped: {exc}") from exc
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved processed features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
