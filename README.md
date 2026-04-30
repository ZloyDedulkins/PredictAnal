# PredictAnal

Небольшое Streamlit-приложение для сравнения моделей прогнозирования по Excel-файлу c колонками:

- `период`
- `показатель`

## Что делает приложение

1. Читает Excel.
2. Делит ряд на:
   - `train`: первые 30 месяцев;
   - `test`: последние 6 месяцев.
3. Строит прогнозы 3 моделями:
   - Moving Average (baseline)
   - Holt-Winters
   - Prophet
4. Считает метрики на test:
   - MAE
   - RMSE
5. Выбирает лучшую модель по RMSE и визуализирует результаты.

## Запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```
