import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class DashboardPage(QWidget):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📊 Dashboard")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        self.info_label = QLabel()
        layout.addWidget(self.info_label)

        layout.addStretch()

        self.update_paths(settings)

    def update_paths(self, settings: dict):
        self.settings = settings

        text = (
            f"Dataset: {settings.get('dataset_dir')}\n"
            f"Runs: {settings.get('runs_dir')}\n"
            f"Models: {settings.get('models_dir')}\n"
            f"Predict Output: {settings.get('predict_output_dir')}\n"
            f"History: {settings.get('history_dir')}\n"
            f"Temp: {settings.get('temp_dir')}\n"
        )
        self.info_label.setText(text)
