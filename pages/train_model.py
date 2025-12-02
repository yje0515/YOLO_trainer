import os
import shutil
import json
import datetime

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QPushButton,
    QHBoxLayout, QComboBox, QLineEdit, QFileDialog
)
from ultralytics import YOLO


# ==========================================================
# ✔ 학습을 수행하는 Worker (백그라운드)
# ==========================================================
class TrainWorker(QThread):
    log_signal = Signal(str)

    def __init__(self, model_name, data_yaml, epochs, patience):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.patience = patience

    def run(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_signal.emit(f"🧪 학습 시작 → {timestamp}")

            model = YOLO(self.model_name)

            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                patience=self.patience,
                imgsz=640,
                batch=8
            )

            # best.pt 경로
            best_weight = "runs/detect/train/weights/best.pt"
            save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)

            out_path = f"{save_dir}/model_{timestamp}.pt"
            shutil.copy(best_weight, out_path)

            self.log_signal.emit(f"✔ best.pt 저장됨: {out_path}")

            # JSON 기록 저장
            hist_dir = "history"
            os.makedirs(hist_dir, exist_ok=True)

            hist_path = f"{hist_dir}/{timestamp}.json"
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp,
                    "model_file": out_path,
                    "results": results.results_dict
                }, f, indent=4)

            self.log_signal.emit(f"📚 학습 기록 저장됨: {hist_path}")
            self.log_signal.emit("=== 학습 종료 ===")

        except Exception as e:
            self.log_signal.emit(f"❌ 오류 발생: {e}")


# ==========================================================
# ✔ 학습 페이지 UI (TrainModelPage)
# ==========================================================
class TrainModelPage(QWidget):

    train_log_signal = Signal(str)   # main.py에서 로그 출력 연결용

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # --------------------------------------------------
        # 제목
        # --------------------------------------------------
        title = QLabel("🎯 YOLO 모델 학습")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # --------------------------------------------------
        # 모델 선택 콤보박스
        # --------------------------------------------------
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ])

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("모델 선택:"))
        row1.addWidget(self.model_combo)
        layout.addLayout(row1)

        # --------------------------------------------------
        # data.yaml 선택
        # --------------------------------------------------
        self.data_path = QLineEdit()
        self.data_path.setPlaceholderText("data.yaml 파일 경로")
        btn_data = QPushButton("📂 data.yaml 선택")
        btn_data.clicked.connect(self.select_dataset)

        row2 = QHBoxLayout()
        row2.addWidget(self.data_path)
        row2.addWidget(btn_data)
        layout.addLayout(row2)

        # --------------------------------------------------
        # Epoch 입력
        # --------------------------------------------------
        self.epoch_input = QLineEdit("50")

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Epoch:"))
        row3.addWidget(self.epoch_input)
        layout.addLayout(row3)

        # --------------------------------------------------
        # Patience 입력
        # --------------------------------------------------
        self.patience_input = QLineEdit("20")

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Patience:"))
        row4.addWidget(self.patience_input)
        layout.addLayout(row4)

        # --------------------------------------------------
        # 학습 버튼
        # --------------------------------------------------
        btn_train = QPushButton("🚀 학습 시작")
        btn_train.clicked.connect(self.start_training)
        btn_train.setStyleSheet("padding: 10px; font-size: 15px;")
        layout.addWidget(btn_train)

        layout.addStretch()
        self.setLayout(layout)

        self.worker = None

    # ------------------------------------------------------
    # data.yaml 파일 선택
    # ------------------------------------------------------
    def select_dataset(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML Files (*.yaml)")
        if file:
            self.data_path.setText(file)

    # ------------------------------------------------------
    # 학습 시작
    # ------------------------------------------------------
    def start_training(self):
        model_name = self.model_combo.currentText()
        data_yaml = self.data_path.text()
        epochs = int(self.epoch_input.text())
        patience = int(self.patience_input.text())

        if not os.path.exists(data_yaml):
            self.train_log_signal.emit("❌ data.yaml 경로가 잘못되었습니다.")
            return

        self.worker = TrainWorker(model_name, data_yaml, epochs, patience)
        self.worker.log_signal.connect(self.train_log_signal.emit)
        self.worker.start()
