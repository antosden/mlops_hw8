from pathlib import Path
import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset


REPORT_DIR = Path("drift/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Создаем простой датасет, похожий по структуре на пример из семинара
np.random.seed(42)

df = pd.DataFrame({
    "title": [f"movie_{i}" for i in range(200)],
    "score": np.random.normal(loc=0.0, scale=0.5, size=200),
    "published": np.random.choice(
        ["Mon", "Tue", "Wed", "Thu", "Fri"],
        size=200
    ),
    "tickers": np.random.choice(
        ["GAZP", "SBER", "YNDX", "VKCO"],
        size=200
    )
})

# 2. Кодируем категориальный столбец, как в семинаре
df["tickers"] = LabelEncoder().fit_transform(df["tickers"])

# 3. Удаляем текстовый столбец title
adult_ref = df.drop(columns=["title"], axis=1)

print("Prepared data:")
print(adult_ref.head())

# 4. Reference batch — эталонные данные
reference_batch = adult_ref.iloc[0:100].copy()

# 5. Current batch — данные со сдвигом
# Специально меняем распределение score и tickers, чтобы дрифт точно был виден
current_batch = adult_ref.iloc[100:200].copy()
current_batch["score"] = current_batch["score"] * 1000
current_batch["tickers"] = current_batch["tickers"] + 100

# 6. Строим отчет Evidently как в материалах семинара
data_report = Report(
    metrics=[
        DataDriftPreset(stattest="psi", stattest_threshold=0.3),
        DataQualityPreset(),
    ],
    timestamp=datetime.datetime.now(),
)

data_report.run(
    reference_data=reference_batch,
    current_data=current_batch,
)

report_path = REPORT_DIR / "data_drift_report.html"
data_report.save_html(str(report_path))

print()
print("Data drift report saved to:", report_path)
print("Reference batch shape:", reference_batch.shape)
print("Current batch shape:", current_batch.shape)
