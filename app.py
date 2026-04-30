import json
import io
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from prophet import Prophet
except Exception:
    Prophet = None

MIN_ROWS = 24
TRAIN_RATIO = 0.95

app = Flask(__name__)


@dataclass
class ForecastResult:
    name: str
    test_forecast: pd.Series
    full_forecast: pd.Series
    next_month: float
    mae: float
    rmse: float


def load_excel(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    required_cols = {"период", "показатель"}
    lower_map = {c.lower().strip(): c for c in df.columns}

    if not required_cols.issubset(set(lower_map.keys())):
        raise ValueError("Файл должен содержать колонки: 'период' и 'показатель'.")

    out = df[[lower_map["период"], lower_map["показатель"]]].copy()
    out.columns = ["period", "value"]
    out["period"] = pd.to_datetime(out["period"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("period").reset_index(drop=True)

    if len(out) < MIN_ROWS:
        raise ValueError(f"Для расчета нужно минимум {MIN_ROWS} наблюдения.")

    return out


def split_data(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(int(len(df) * train_ratio), 1)
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def baseline_moving_average_forecast(train_values: pd.Series, periods: int, window: int = 3) -> np.ndarray:
    history = train_values.tolist()
    window = min(window, len(history))
    preds = []
    for _ in range(periods):
        pred = float(np.mean(history[-window:]))
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


def holt_winters_forecast(train_values: pd.Series, periods: int, season_length: int = 12) -> np.ndarray:
    if len(train_values) >= season_length * 2:
        model = ExponentialSmoothing(
            train_values, trend="add", seasonal="add", seasonal_periods=season_length
        ).fit(optimized=True)
    else:
        model = ExponentialSmoothing(train_values, trend="add", seasonal=None).fit(optimized=True)
    return np.array(model.forecast(periods))


def prophet_forecast(train_df: pd.DataFrame, periods: int) -> np.ndarray:
    if Prophet is None:
        raise RuntimeError("Prophet не установлен")
    prop_train = train_df.rename(columns={"period": "ds", "value": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prop_train)

    last_period = train_df["period"].max()
    future_periods = pd.date_range(last_period + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    future = pd.DataFrame({"ds": future_periods})
    pred = model.predict(future)
    return pred["yhat"].to_numpy()


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def run_models(train: pd.DataFrame, test: pd.DataFrame, full_df: pd.DataFrame) -> Dict[str, ForecastResult]:
    y_true = test.set_index("period")["value"]
    full_periods = len(full_df) + 1
    full_index = pd.date_range(full_df["period"].min(), periods=full_periods, freq="MS")

    model_fns = {
        "Moving Average (baseline)": lambda s, p: baseline_moving_average_forecast(s, p),
        "Holt-Winters": lambda s, p: holt_winters_forecast(s, p),
    }
    results = {}

    for name, fn in model_fns.items():
        try:
            test_pred_vals = fn(train["value"], len(test))
            test_pred = pd.Series(test_pred_vals, index=test["period"], name=name)
            mae, rmse = evaluate(y_true, test_pred)

            full_pred_vals = fn(full_df["value"], full_periods)
            full_pred = pd.Series(full_pred_vals, index=full_index, name=name)
            next_month_val = float(full_pred.iloc[-1])
            results[name] = ForecastResult(name, test_pred, full_pred, next_month_val, mae, rmse)
        except Exception:
            continue

    try:
        test_pred_vals = prophet_forecast(train[["period", "value"]], len(test))
        test_pred = pd.Series(test_pred_vals, index=test["period"], name="Prophet")
        mae, rmse = evaluate(y_true, test_pred)

        full_pred_vals = prophet_forecast(full_df[["period", "value"]], full_periods)
        full_pred = pd.Series(full_pred_vals, index=full_index, name="Prophet")
        next_month_val = float(full_pred.iloc[-1])
        results["Prophet"] = ForecastResult("Prophet", test_pred, full_pred, next_month_val, mae, rmse)
    except Exception:
        pass

    if not results:
        raise RuntimeError("Ни одна модель не смогла посчитать прогноз")
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    metrics = None
    best_model = None
    chart_payload = None
    next_month_forecast = None

    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            error = "Файл не загружен"
        else:
            try:
                df = load_excel(f.read())
                train, test = split_data(df)
                results = run_models(train, test, df)
                metrics_df = pd.DataFrame([
                    {
                        "Model": r.name,
                        "MAE": round(r.mae, 4),
                        "RMSE": round(r.rmse, 4),
                        "NextMonth": round(r.next_month, 4),
                    }
                    for r in results.values()
                ]).sort_values("RMSE")
                best_model = metrics_df.iloc[0]["Model"]
                metrics = metrics_df.to_dict(orient="records")
                next_month_forecast = metrics_df.iloc[0]["NextMonth"]

                next_period = (df["period"].max() + pd.offsets.MonthBegin(1)).strftime("%m.%Y")

                labels = [d.strftime("%Y-%m") for d in pd.date_range(df["period"].min(), periods=len(df) + 1, freq="MS")]
                fact_values = df["value"].round(4).tolist() + [None]
                datasets = [{"label": "Факт", "data": fact_values}]

                for model_name, result in results.items():
                    datasets.append({
                        "label": model_name,
                        "data": result.full_forecast.round(4).tolist(),
                    })

                chart_payload = json.dumps(
                    {
                        "labels": labels,
                        "datasets": datasets,
                        "next_period": next_period,
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                error = str(exc)

    return render_template(
        'index.html',
        error=error,
        metrics=metrics,
        best_model=best_model,
        chart_payload=chart_payload,
        next_month_forecast=next_month_forecast,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
