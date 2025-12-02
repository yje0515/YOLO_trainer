# pages/train.py

import os
import sys
import shutil
import json
import datetime
import csv
import threading
import time
import re
from io import StringIO

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QComboBox, QLineEdit
)

from ultralytics import YOLO


##############################################################
# 최신 data.yaml 자동 탐색
##############################################################
def find_latest_data_yaml(root_dir="."):
    candidates = []
    for curr_root, dirs, files in os.walk(root_dir):
        if "data.yaml" in files:
            full = os.path.join(curr_root, "data.yaml")
            mtime = os.path.getmtime(full)
            candidates.append((mtime, full))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


##############################################################
# 최신 run 디렉토리 찾기 (train/predict 둘 다 허용)
##############################################################
def get_latest_run_dir():
    search_paths = [
        os.path.abspath("runs/detect"),
        os.path.abspath("../runs/detect"),
        os.path.abspath("../../runs/detect"),
    ]

    latest = None
    latest_time = -1

    for base in search_paths:
        if not os.path.isdir(base):
            continue

        for d in os.listdir(base):
            full = os.path.join(base, d)
            if (
                os.path.isdir(full)
                and (d.startswith("train") or d.startswith("predict"))
            ):
                t = os.path.getmtime(full)
                if t > latest_time:
                    latest_time = t
                    latest = full

    return latest


##############################################################
# TrainWorker — GPU 학습 + 실시간 로그 + best 저장
##############################################################
class TrainWorker(QThread):
    log_signal = Signal(str)
    epoch_signal = Signal(str)

    def __init__(self, model_name, data_yaml, epochs, patience):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.patience = patience

    def run(self):
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
        try:
            self.log_signal.emit(f"🧪 학습 시작 → {timestamp}")
            self.log_signal.emit(f"📂 dataset: {self.data_yaml}")

            #################################################
            # YOLO 모델 로드
            #################################################
            model = YOLO(self.model_name)

            #################################################
            # stdout 후킹 (실시간 log 출력)
            #################################################
            old_stdout = sys.stdout
            buf = StringIO()
            sys.stdout = buf

            def monitor():
                last = ""
                while True:
                    text = buf.getvalue()
                    if text != last:
                        new_part = text[len(last):]
                        last = text

                        if new_part.strip():
                            self.log_signal.emit(new_part.strip())

                        # Epoch 파싱 (1/50 → 2/50 → ...)
                        m = re.search(r"(\d+)\/(\d+)", new_part)
                        if m:
                            cur = int(m.group(1))
                            tot = int(m.group(2))
                            pct = int(cur / tot * 100)
                            self.epoch_signal.emit(f"Epoch {cur}/{tot} ({pct}%)")

                    if self.isFinished():
                        break
                    time.sleep(0.1)

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()

            #################################################
            # 📌 YOLO 학습 (GPU 사용 + 저장 경로 강제)
            #################################################
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                patience=self.patience,
                imgsz=640,
                batch=8,
                device=0,                      # GPU
                project="runs/detect",         # 저장 폴더 강제
                name="train",                  # train/ 폴더로 고정
                exist_ok=True
            )

            # stdout 원상복구
            sys.stdout = old_stdout

            #################################################
            # 최신 train 폴더 찾기
            #################################################
            run_dir = get_latest_run_dir()
            if not run_dir:
                self.log_signal.emit("⚠ run 폴더를 찾지 못했습니다.")
                return

            #################################################
            # best.pt 저장
            #################################################
            best_src = os.path.join(run_dir, "weights", "best.pt")
            if not os.path.exists(best_src):
                self.log_signal.emit("⚠ best.pt 찾지 못함")
                return

            os.makedirs("models", exist_ok=True)

            best_name = f"best_{timestamp}.pt"
            best_dst = os.path.join("models", best_name)
            shutil.copy(best_src, best_dst)

            self.log_signal.emit(f"✔ best 모델 저장됨 → {best_dst}")

            #################################################
            # metrics 저장
            #################################################
            metrics = {}
            csv_path = os.path.join(run_dir, "results.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                header = rows[0]
                last = rows[-1]
                metrics = {header[i]: last[i] for i in range(len(header))}

            #################################################
            # history JSON 저장
            #################################################
            os.makedirs("history", exist_ok=True)
            hist_path = os.path.join("history", f"{timestamp}.json")

            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp,
                    "model_file": best_dst,
                    "run_dir": run_dir,
                    "epochs": self.epochs,
                    "patience": self.patience,
                    "metrics": metrics
                }, f, indent=4, ensure_ascii=False)

            self.log_signal.emit(f"📚 기록 저장됨 → {hist_path}")
            self.log_signal.emit("=== 학습 종료 ===")

        except Exception as e:
            sys.stdout = old_stdout
            self.log_signal.emit(f"❌ 오류: {e}")


##############################################################
# TrainPage — UI
##############################################################
class TrainPage(QWidget):
    train_log_signal = Signal(str)

    def __init__(self):
        super().__init__()

        self.dataset_path = None
        self.worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🧪 Train YOLO Model")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.dataset_label = QLabel("📂 선택된 데이터셋: (없음)")
        layout.addWidget(self.dataset_label)

        # 모델 선택
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        r1.addWidget(self.model_combo)
        layout.addLayout(r1)

        # Epoch
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Epochs:"))
        self.epoch_input = QLineEdit("50")
        r2.addWidget(self.epoch_input)
        layout.addLayout(r2)

        # Patience
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Patience:"))
        self.pat_input = QLineEdit("20")
        r3.addWidget(self.pat_input)
        layout.addLayout(r3)

        # Epoch 상태 표시
        self.epoch_status = QLabel("Epoch 상태: -")
        layout.addWidget(self.epoch_status)

        # 학습 버튼
        self.train_btn = QPushButton("🚀 학습 시작")
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)

        layout.addStretch()

    #################################################
    # dataset path 받기
    #################################################
    def set_dataset_path(self, path):
        self.dataset_path = path
        self.dataset_label.setText(f"📂 선택된 데이터셋: {path}")

    #################################################
    # 학습 시작
    #################################################
    def start_training(self):
        if not self.dataset_path:
            self.train_log_signal.emit("❌ 데이터셋을 선택하세요.")
            return

        model_name = self.model_combo.currentText()
        try:
            epochs = int(self.epoch_input.text())
            patience = int(self.pat_input.text())
        except ValueError:
            self.train_log_signal.emit("❌ Epoch/Patience는 정수로 입력해야 합니다.")
            return

        self.worker = TrainWorker(model_name, self.dataset_path, epochs, patience)
        self.worker.log_signal.connect(self.train_log_signal.emit)
        self.worker.epoch_signal.connect(self.update_epoch_status)
        self.worker.start()

    def update_epoch_status(self, text):
        self.epoch_status.setText(text)
