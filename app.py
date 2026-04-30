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
    forecast: pd.Series
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


def baseline_moving_average(train: pd.DataFrame, test: pd.DataFrame, window: int = 3) -> pd.Series:
    history = train["value"].tolist()
    window = min(window, len(history))
    preds = []
    for _ in range(len(test)):
        preds.append(np.mean(history[-window:]))
        history.append(preds[-1])
    return pd.Series(preds, index=test["period"], name="Moving Average (baseline)")


def holt_winters_model(train: pd.DataFrame, test: pd.DataFrame, season_length: int = 12) -> pd.Series:
    if len(train) >= season_length * 2:
        model = ExponentialSmoothing(
            train["value"], trend="add", seasonal="add", seasonal_periods=season_length
        ).fit(optimized=True)
    else:
        model = ExponentialSmoothing(train["value"], trend="add", seasonal=None).fit(optimized=True)
    forecast = model.forecast(len(test))
    forecast.index = test["period"]
    forecast.name = "Holt-Winters"
    return forecast


def prophet_model(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    if Prophet is None:
        raise RuntimeError("Prophet не установлен")
    prop_train = train.rename(columns={"period": "ds", "value": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prop_train)
    future = pd.DataFrame({"ds": test["period"]})
    pred = model.predict(future)
    forecast = pred.set_index("ds")["yhat"]
    forecast.name = "Prophet"
    return forecast


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def run_models(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, ForecastResult]:
    y_true = test.set_index("period")["value"]
    model_fns = {
        "Moving Average (baseline)": lambda: baseline_moving_average(train, test),
        "Holt-Winters": lambda: holt_winters_model(train, test),
        "Prophet": lambda: prophet_model(train, test),
    }
    results = {}
    for name, fn in model_fns.items():
        try:
            pred = fn()
            mae, rmse = evaluate(y_true, pred)
            results[name] = ForecastResult(name, pred, mae, rmse)
        except Exception:
            continue
    if not results:
        raise RuntimeError("Ни одна модель не смогла посчитать прогноз")
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    metrics = None
    best_model = None
    chart_payload = None

    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            error = "Файл не загружен"
        else:
            try:
                df = load_excel(f.read())
                train, test = split_data(df)
                results = run_models(train, test)
                metrics_df = pd.DataFrame([
                    {"Model": r.name, "MAE": round(r.mae, 4), "RMSE": round(r.rmse, 4)}
                    for r in results.values()
                ]).sort_values("RMSE")
                best_model = metrics_df.iloc[0]["Model"]
                metrics = metrics_df.to_dict(orient="records")

                labels = [d.strftime("%Y-%m") for d in test["period"]]
                datasets = [
                    {
                        "label": "Факт",
                        "data": test["value"].round(4).tolist(),
                    }
                ]
                for model_name, result in results.items():
                    datasets.append({
                        "label": model_name,
                        "data": result.forecast.reindex(test["period"]).round(4).tolist(),
                    })

                chart_payload = json.dumps({"labels": labels, "datasets": datasets}, ensure_ascii=False)
            except Exception as exc:
                error = str(exc)

    return render_template(
        'index.html',
        error=error,
        metrics=metrics,
        best_model=best_model,
        chart_payload=chart_payload,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
