from evidently.report import Report
from evidently.metrics import DataDriftPreset
import os

# Comparar train y test para detectar drift en entrada
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)

# Guardar el reporte como archivo HTML
os.makedirs("reports", exist_ok=True)
report_path = "reports/data_drift_report.html"
report.save_html(report_path)

# Loggear como artefacto en MLflow
mlflow.log_artifact(report_path, artifact_path="drift_reports")
