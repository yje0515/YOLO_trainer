import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem
)


class HistoryPage(QWidget):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📜 Model History")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["시간", "경로", "YOLO 모델", "Epochs", "Patience", "mAP50"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.table)
        layout.addStretch()

        self.reload_history()

    def update_paths(self, settings: dict):
        self.settings = settings
        self.reload_history()

    def reload_history(self, *_):
        self.table.setRowCount(0)
        history_root = self.settings.get("history_dir", "./history")
        if not os.path.isdir(history_root):
            return

        entries = []
        for name in os.listdir(history_root):
            sub = os.path.join(history_root, name)
            if not os.path.isdir(sub):
                continue
            meta = os.path.join(sub, "metadata.json")
            if os.path.exists(meta):
                entries.append(meta)

        entries.sort()

        for meta_path in entries:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            row = self.table.rowCount()
            self.table.insertRow(row)

            ts = data.get("timestamp", "-")
            model_file = data.get("models_file", "-")
            base_model = data.get("base_model", "-")
            epochs = str(data.get("epochs", "-"))
            patience = str(data.get("patience", "-"))
            mAP50 = str(data.get("metrics", {}).get("metrics/mAP50(B)", data.get("metrics", {}).get("metrics/mAP50", "-")))

            self.table.setItem(row, 0, QTableWidgetItem(ts))
            self.table.setItem(row, 1, QTableWidgetItem(model_file))
            self.table.setItem(row, 2, QTableWidgetItem(base_model))
            self.table.setItem(row, 3, QTableWidgetItem(epochs))
            self.table.setItem(row, 4, QTableWidgetItem(patience))
            self.table.setItem(row, 5, QTableWidgetItem(mAP50))
