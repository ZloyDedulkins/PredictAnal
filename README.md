# PredictAnal

Веб-приложение (сайт) на Streamlit для сравнения моделей прогнозирования по Excel-файлу с колонками:

- `период`
- `показатель`

## Логика

1. Чтение Excel.
2. Разбиение данных: **95% train / 5% test**.
3. Прогноз тремя моделями:
   - Moving Average (baseline)
   - Holt-Winters
   - Prophet (если установлен)
4. Расчет метрик на test:
   - MAE
   - RMSE
5. Выбор лучшей модели по RMSE.
6. Визуализация на графике + таблицы train/test.

## Запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

Откройте адрес из консоли (обычно `http://localhost:8501`).

## Опционально: Prophet

Если хотите включить модель Prophet, установите ее отдельно:

```bash
pip install prophet
```
