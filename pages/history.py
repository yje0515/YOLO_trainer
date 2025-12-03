import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem
)


class HistoryPage(QWidget):
    def __init__(self, settings):
        super().__init__()

        self.settings = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📜 학습 히스토리")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["시간", "모델 파일", "Epoch", "Patience", "mAP50"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.table)

        self.load_history()

    def load_history(self):
        history_dir = self.settings.get("history_dir", "history")

        if not os.path.exists(history_dir):
            return

        for file in os.listdir(history_dir):
            if not file.endswith(".json"):
                continue

            with open(os.path.join(history_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)

            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(data.get("timestamp", "-")))
            self.table.setItem(row, 1, QTableWidgetItem(data.get("model_file", "-")))
            self.table.setItem(row, 2, QTableWidgetItem(str(data.get("epochs", "-"))))
            self.table.setItem(row, 3, QTableWidgetItem(str(data.get("patience", "-"))))
            self.table.setItem(row, 4, QTableWidgetItem(str(data.get("metrics", {}).get("metrics/mAP50", "-"))))
