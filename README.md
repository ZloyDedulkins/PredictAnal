# PredictAnal

Теперь это обычный **веб-сайт на Flask** (без Streamlit), чтобы не было ошибки `streamlit: command not found`.

## Запуск сайта

```bash
pip install -r requirements.txt
python app.py
```

Откройте: `http://localhost:8000`

## Что делает сайт

- Принимает Excel с колонками `период`, `показатель`.
- Делит данные: `train=95%`, `test=5%`.
- Считает 3 модели: Moving Average, Holt-Winters, Prophet (если установлен отдельно).
- Считает MAE и RMSE.
- Выбирает лучшую модель по RMSE.
- Показывает таблицу метрик и лучшую модель прямо в браузере.
