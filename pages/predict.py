# pages/predict.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QFileDialog, QComboBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Signal

from ultralytics import YOLO
from utils.model_loader import load_model_list


class PredictPage(QWidget):

    predict_log_signal = Signal(str)

    def __init__(self):
        super().__init__()

        self.model_path = None
        self.image_path = None
        self.output_image = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🔮 Predict with YOLO Model")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # 모델 선택
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("모델 선택:"))
        self.model_combo = QComboBox()
        row1.addWidget(self.model_combo)
        layout.addLayout(row1)

        # 모델 목록 로딩
        self.refresh_model_list()

        # 이미지 선택 버튼
        self.btn_select = QPushButton("📂 이미지 선택")
        self.btn_select.clicked.connect(self.select_image)
        layout.addWidget(self.btn_select)

        # 예측 실행 버튼
        self.btn_predict = QPushButton("🚀 Predict")
        self.btn_predict.clicked.connect(self.run_predict)
        layout.addWidget(self.btn_predict)

        # 결과 이미지 박스
        self.preview = QLabel()
        self.preview.setFixedHeight(400)
        self.preview.setStyleSheet("border: 1px solid gray;")
        self.preview.setScaledContents(True)
        layout.addWidget(self.preview)

        layout.addStretch()

    #######################################################
    def refresh_model_list(self):
        """models 폴더의 모델 자동 로딩"""
        models = load_model_list()
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
        else:
            self.model_combo.addItem("(모델 없음)")

    #######################################################
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", ".", "Images (*.jpg *.png *.jpeg)"
        )
        if file_path:
            self.image_path = file_path
            self.preview.setPixmap(QPixmap(file_path))
            self.predict_log_signal.emit(f"✔ 이미지 선택됨: {file_path}")

    #######################################################
    def run_predict(self):
        if not self.image_path:
            self.predict_log_signal.emit("❌ 이미지가 선택되지 않았습니다.")
            return

        model_file = self.model_combo.currentText()
        if model_file == "(모델 없음)":
            self.predict_log_signal.emit("❌ 모델이 없습니다.")
            return

        model_path = os.path.join("models", model_file)

        self.predict_log_signal.emit(f"🔍 모델 로드: {model_path}")
        model = YOLO(model_path)

        self.predict_log_signal.emit(f"🚀 예측 실행 중...")

        results = model(self.image_path)

        # 결과 이미지 저장
        save_dir = "predict_output"
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, "result.jpg")
        results[0].save(out_path)

        self.predict_log_signal.emit(f"✔ 결과 저장됨: {out_path}")

        # UI에 표시
        self.preview.setPixmap(QPixmap(out_path))
