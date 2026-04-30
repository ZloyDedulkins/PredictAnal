import io
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None


@dataclass
class ForecastResult:
    name: str
    forecast: pd.Series
    mae: float
    rmse: float


def load_excel(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    required_cols = {"период", "показатель"}
    lower_map = {c.lower().strip(): c for c in df.columns}

    if not required_cols.issubset(set(lower_map.keys())):
        raise ValueError("Файл должен содержать колонки: 'период' и 'показатель'.")

    period_col = lower_map["период"]
    value_col = lower_map["показатель"]

    out = df[[period_col, value_col]].copy()
    out.columns = ["period", "value"]
    out["period"] = pd.to_datetime(out["period"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("period").reset_index(drop=True)

    if len(out) < 36:
        raise ValueError("Для расчета нужно минимум 36 наблюдений (30 train + 6 test).")

    return out


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.iloc[:30].copy()
    test = df.iloc[-6:].copy()
    return train, test


def baseline_moving_average(train: pd.DataFrame, test: pd.DataFrame, window: int = 3) -> pd.Series:
    history = train["value"].tolist()
    preds = []
    for _ in range(len(test)):
        preds.append(np.mean(history[-window:]))
        history.append(preds[-1])
    return pd.Series(preds, index=test["period"], name="baseline_ma")


def holt_winters_model(train: pd.DataFrame, test: pd.DataFrame, season_length: int = 12) -> pd.Series:
    model = ExponentialSmoothing(
        train["value"], trend="add", seasonal="add", seasonal_periods=season_length
    ).fit(optimized=True)
    forecast = model.forecast(len(test))
    forecast.index = test["period"]
    forecast.name = "holt_winters"
    return forecast


def prophet_model(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    if Prophet is None:
        raise RuntimeError("Пакет prophet не установлен.")

    prop_train = train.rename(columns={"period": "ds", "value": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prop_train)

    future = pd.DataFrame({"ds": test["period"]})
    pred = model.predict(future)
    forecast = pred.set_index("ds")["yhat"]
    forecast.index.name = None
    forecast.name = "prophet"
    return forecast


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def run_models(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, ForecastResult]:
    results = {}
    y_true = test.set_index("period")["value"]

    models = {
        "Moving Average (baseline)": lambda: baseline_moving_average(train, test),
        "Holt-Winters": lambda: holt_winters_model(train, test),
        "Prophet": lambda: prophet_model(train, test),
    }

    for name, fn in models.items():
        try:
            pred = fn()
            mae, rmse = evaluate(y_true, pred)
            results[name] = ForecastResult(name=name, forecast=pred, mae=mae, rmse=rmse)
        except Exception as exc:
            st.warning(f"Модель {name} не рассчитана: {exc}")

    return results


def main() -> None:
    st.set_page_config(page_title="Сравнение моделей прогнозирования", layout="wide")
    st.title("Прогнозирование ряда: MA / Holt-Winters / Prophet")
    st.write("Загрузите Excel-файл с колонками **период** и **показатель**.")

    uploaded = st.file_uploader("Excel-файл", type=["xlsx", "xls"])

    if not uploaded:
        return

    try:
        df = load_excel(uploaded.getvalue())
        train, test = split_data(df)
        results = run_models(train, test)

        if not results:
            st.error("Не удалось рассчитать ни одну модель.")
            return

        metrics = pd.DataFrame(
            [
                {
                    "model": r.name,
                    "MAE": r.mae,
                    "RMSE": r.rmse,
                }
                for r in results.values()
            ]
        ).sort_values("RMSE")

        best_model_name = metrics.iloc[0]["model"]

        st.subheader("Метрики на тесте (последние 6 месяцев)")
        st.dataframe(metrics, use_container_width=True)
        st.success(f"Лучшая модель по RMSE: **{best_model_name}**")

        plot_df = df.set_index("period")["value"].to_frame("actual")
        plot_df.loc[test["period"], "actual_test"] = test["value"].values
        for name, res in results.items():
            plot_df[name] = np.nan
            plot_df.loc[res.forecast.index, name] = res.forecast.values

        st.subheader("Визуализация")
        st.line_chart(plot_df)

        st.subheader("Train / Test")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Train (первые 30)")
            st.dataframe(train, use_container_width=True)
        with col2:
            st.write("Test (последние 6)")
            st.dataframe(test, use_container_width=True)

    except Exception as err:
        st.error(f"Ошибка обработки файла: {err}")


if __name__ == "__main__":
    main()
