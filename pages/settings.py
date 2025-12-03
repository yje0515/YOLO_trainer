import json
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QFileDialog
)
from PySide6.QtCore import Signal


DEFAULT_SETTINGS = {
    "dataset_dir": "C:/yolo_data/datasets",
    "runs_dir": "C:/yolo_data/runs",
    "models_dir": "C:/yolo_data/models",
    "predict_output_dir": "C:/yolo_data/predict_output",
    "history_dir": "C:/yolo_data/history",
    "temp_dir": "C:/yolo_data/temp"
}

SETTINGS_PATH = "config/settings.json"


def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_SETTINGS.copy()


def save_settings(data: dict):
    os.makedirs("config", exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class SettingsPage(QWidget):
    settings_changed = Signal(dict)

    def __init__(self):
        super().__init__()

        self.settings = load_settings()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("⚙ 설정 — 저장 경로 관리")
        title.setStyleSheet("font-size: 18px; font-weight:bold;")
        layout.addWidget(title)

        def add_row(label_text, key):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))

            edit = QLineEdit(self.settings[key])
            row.addWidget(edit)

            btn = QPushButton("📂 선택")
            btn.clicked.connect(lambda: self.select_path(edit, key))
            row.addWidget(btn)

            layout.addLayout(row)
            return edit

        self.ed_dataset = add_row("데이터셋 저장경로:", "dataset_dir")
        self.ed_runs = add_row("runs 저장경로:", "runs_dir")
        self.ed_models = add_row("모델 저장경로:", "models_dir")
        self.ed_predict = add_row("Predict 출력경로:", "predict_output_dir")
        self.ed_history = add_row("History 경로:", "history_dir")
        self.ed_temp = add_row("Temp 경로:", "temp_dir")

        btn_save = QPushButton("💾 저장")
        btn_save.clicked.connect(self.save_click)
        layout.addWidget(btn_save)
        layout.addStretch()

    def select_path(self, edit: QLineEdit, key):
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            edit.setText(folder)
            self.settings[key] = folder

    def save_click(self):
        self.settings["dataset_dir"] = self.ed_dataset.text()
        self.settings["runs_dir"] = self.ed_runs.text()
        self.settings["models_dir"] = self.ed_models.text()
        self.settings["predict_output_dir"] = self.ed_predict.text()
        self.settings["history_dir"] = self.ed_history.text()
        self.settings["temp_dir"] = self.ed_temp.text()

        save_settings(self.settings)
        self.settings_changed.emit(self.settings)
