import os
import shutil
import json
import datetime
import csv
import sys
import io
import time

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QComboBox, QLineEdit, QFileDialog, QTextEdit
)

from ultralytics import YOLO


# ============================
#   학습 Worker Thread
# ============================
class TrainWorker(QThread):
    log_signal = Signal(str)        # 로그 출력
    finished_ok = Signal(str)       # best.pt 경로 전달
    stopped = Signal()              # 중단 시그널

    def __init__(self, model_name, data_yaml, epochs, patience, paths: dict):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.patience = patience
        self.paths = paths
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

        runs_dir = self.paths["runs_dir"]
        models_dir = self.paths["models_dir"]
        history_dir = self.paths["history_dir"]

        os.makedirs(runs_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)

        self.log_signal.emit(f"🧪 학습 시작 ({timestamp})")
        self.log_signal.emit(f"data.yaml: {self.data_yaml}")

        # ----- stdout redirect -----
        class Redirect(io.TextIOBase):
            def __init__(self, callback):
                self.callback = callback
                self.buffer = ""

            def write(self, text):
                self.buffer += text
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self.callback(line)
                return len(text)

            def flush(self):
                if self.buffer:
                    self.callback(self.buffer.strip())
                    self.buffer = ""

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = Redirect(self.log_signal.emit)
        sys.stderr = Redirect(self.log_signal.emit)

        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"Device: {device}")
        except:
            device = "cpu"
            self.log_signal.emit("CUDA 체크 실패 → CPU 사용")

        model = YOLO(self.model_name)

        # ----- STOP 체크를 위한 콜백 추가 -----
        def callback(trainer):
            if self.stop_flag:
                trainer.stop = True
                self.log_signal.emit("🛑 학습 중지 신호 감지 → 종료 중...")
                time.sleep(0.3)

        try:
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                patience=self.patience,
                imgsz=640,
                batch=8,
                device=device,
                project=runs_dir,
                name=f"train_{timestamp}",
                exist_ok=True,
                callbacks={"on_train_epoch_end": callback}
            )
        except Exception as e:
            self.log_signal.emit(f"❌ 학습 실패: {e}")
            return
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        if self.stop_flag:
            self.stopped.emit()
            return

        # 학습 로그 저장 폴더
        run_dir = os.path.join(runs_dir, f"train_{timestamp}")
        best_src = os.path.join(run_dir, "weights", "best.pt")

        if not os.path.exists(best_src):
            self.log_signal.emit("⚠ best.pt를 찾을 수 없습니다.")
            return

        # 모델 저장
        best_name = f"best_{timestamp}.pt"
        best_dst = os.path.join(models_dir, best_name)
        shutil.copy(best_src, best_dst)

        # history 저장
        hist_dir = os.path.join(history_dir, timestamp)
        os.makedirs(hist_dir, exist_ok=True)
        shutil.copy(best_src, os.path.join(hist_dir, "best.pt"))

        meta = {
            "timestamp": timestamp,
            "data_yaml": self.data_yaml,
            "base_model": self.model_name,
            "epochs": self.epochs,
            "patience": self.patience,
            "models_file": best_dst,
            "run_dir": run_dir
        }

        with open(os.path.join(hist_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

        self.log_signal.emit(f"✔ 학습 완료 → {best_dst}")
        self.finished_ok.emit(best_dst)


# ============================
#   Train Page (UI)
# ============================
class TrainPage(QWidget):
    model_saved_signal = Signal(str)

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None
        self.worker = None
        self.data_yaml = None
        self.update_paths(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🧪 Train Model")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # 선택된 데이터셋 표시
        self.dataset_label = QLabel("📂 data.yaml 선택되지 않음")
        layout.addWidget(self.dataset_label)

        btn_sel = QPushButton("📂 data.yaml 불러오기")
        btn_sel.clicked.connect(self.select_dataset)
        layout.addWidget(btn_sel)

        # 모델 선택
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("YOLO 모델 선택하기 :"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        row1.addWidget(self.model_combo)
        layout.addLayout(row1)

        # Epoch
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Epochs:"))
        self.epoch_input = QLineEdit("30")
        row2.addWidget(self.epoch_input)
        layout.addLayout(row2)

        # Patience
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Patience:"))
        self.pat_input = QLineEdit("10")
        row3.addWidget(self.pat_input)
        layout.addLayout(row3)

        # 버튼
        row4 = QHBoxLayout()
        self.btn_start = QPushButton("🚀 학습 시작")
        self.btn_stop = QPushButton("🛑 학습 중단")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop.clicked.connect(self.stop_training)
        row4.addWidget(self.btn_start)
        row4.addWidget(self.btn_stop)
        layout.addLayout(row4)

        # 로그창
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-family:Consolas; font-size:12px;")
        layout.addWidget(self.log_box)

        layout.addStretch()

    def set_overlay(self, overlay):
        self.overlay = overlay

    def update_paths(self, settings: dict):
        self.paths = settings

    def set_dataset_path(self, path: str):
        self.data_yaml = path
        self.dataset_label.setText(f"📂 선택된 데이터셋 data.yaml: {path}")

    def select_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", ".", "YAML (*.yaml)")
        if path:
            self.set_dataset_path(path)

    def start_training(self):
        if not self.data_yaml:
            self.log_box.append("❌ data.yaml 선택 후 학습이 가능합니다.")
            return

        epochs = int(self.epoch_input.text())
        patience = int(self.pat_input.text())
        model_name = self.model_combo.currentText()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        if self.overlay:
            self.overlay.show_overlay("🧪 모델 학습 중...")

        self.worker = TrainWorker(model_name, self.data_yaml, epochs, patience, self.paths)
        self.worker.log_signal.connect(self.log_box.append)
        self.worker.finished_ok.connect(self.on_model_saved)
        self.worker.finished.connect(self.training_done)
        self.worker.stopped.connect(self.training_stopped)

        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.log_box.append("🛑 사용자 중지 요청...")
            self.worker.stop()

    def training_stopped(self):
        self.log_box.append("🛑 학습이 중단되었습니다.")
        self.training_done()

    def training_done(self):
        if self.overlay:
            self.overlay.hide_overlay()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_box.append("=== 학습 종료 ===")

    def on_model_saved(self, path: str):
        self.model_saved_signal.emit(path)
        self.log_box.append(f"✔ 모델 저장완료! : {path}")
