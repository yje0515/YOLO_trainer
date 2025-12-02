# pages/history.py
import os
import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit,
    QAbstractItemView
)
from PySide6.QtCore import Qt


class HistoryPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)

        title = QLabel("📚 Training History")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # 히스토리 테이블
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Timestamp", "Model File", "mAP50", "Precision", "Recall"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        # ✅ 여기 두 줄이 문제였음: 상수는 QAbstractItemView 에서 가져와야 함
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.table.itemSelectionChanged.connect(self.show_detail)

        refresh_btn = QPushButton("🔄 새로고침")
        refresh_btn.clicked.connect(self.load_history)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumHeight(180)
        self.detail.setStyleSheet("font-family: Consolas; font-size: 12px;")

        layout.addWidget(refresh_btn)
        layout.addWidget(self.table)
        layout.addWidget(self.detail)

        self.history_data = []
        self.load_history()

    def load_history(self):
        self.table.setRowCount(0)
        self.history_data = []

        if not os.path.isdir("history"):
            return

        files = [f for f in os.listdir("history") if f.endswith(".json")]
        files.sort(reverse=True)  # 최신 기록 위로

        for fname in files:
            path = os.path.join("history", fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            metrics = data.get("metrics", {})
            # YOLO results.csv 컬럼 이름은 버전에 따라 다를 수 있어서 여유있게 처리
            map50 = (
                metrics.get("metrics/mAP50(B)") or
                metrics.get("metrics/mAP50-95(B)") or
                metrics.get("metrics/mAP50-95(B)") or
                ""
            )
            prec = metrics.get("metrics/precision(B)", "")
            rec = metrics.get("metrics/recall(B)", "")

            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(data.get("timestamp", "")))
            self.table.setItem(row, 1, QTableWidgetItem(os.path.basename(data.get("model_file", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(str(map50)))
            self.table.setItem(row, 3, QTableWidgetItem(str(prec)))
            self.table.setItem(row, 4, QTableWidgetItem(str(rec)))

            self.history_data.append(data)

    def show_detail(self):
        selected = self.table.selectedIndexes()
        if not selected:
            return
        row = selected[0].row()
        if row >= len(self.history_data):
            return

        data = self.history_data[row]
        text_lines = [
            f"Timestamp : {data.get('timestamp')}",
            f"Model File: {data.get('model_file')}",
            f"Run Dir   : {data.get('run_dir')}",
            f"Epochs    : {data.get('epochs')}",
            f"Patience  : {data.get('patience')}",
            "",
            "[Metrics]",
        ]
        metrics = data.get("metrics", {})
        for k, v in metrics.items():
            text_lines.append(f"{k}: {v}")

        self.detail.setPlainText("\n".join(text_lines))
