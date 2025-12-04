import os
import shutil
import json
import datetime
import sys
import io
import time

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QComboBox, QLineEdit, QFileDialog, QTextEdit
)

from ultralytics import YOLO


# ============================
#   학습 Worker Thread
# ============================
class TrainWorker(QThread):
    log_signal = Signal(str)
    finished_ok = Signal(str)

    def __init__(self, model_name, data_yaml, epochs, patience, paths: dict):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.patience = patience
        self.paths = paths

    def run(self):
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

        runs_dir = self.paths["runs_dir"]
        models_dir = self.paths["models_dir"]
        history_dir = self.paths["history_dir"]

        os.makedirs(runs_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)

        # 시작 로그
        self.log_signal.emit(f"🧪 학습 시작 ({timestamp})")
        self.log_signal.emit(f"data.yaml: {self.data_yaml}")

        # stdout redirect
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

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Redirect(self.log_signal.emit)
        sys.stderr = Redirect(self.log_signal.emit)

        # Device check
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"Device: {device}")
        except:
            device = "cpu"
            self.log_signal.emit("CUDA 체크 실패 → CPU 사용")

        start_time = time.time()

        model = YOLO(self.model_name)

        # -------------------------------
        # 반드시 save=True 해야 YOLO8 CSV 생성됨
        # -------------------------------
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
                save=True,                 # 🔥 핵심
                exist_ok=True
            )
        except Exception as e:
            self.log_signal.emit(f"❌ 학습 실패: {e}")
            sys.stdout, sys.stderr = old_stdout, old_stderr
            return
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        # -------------------------------
        # mAP50 가져오기 (YOLO8 + YOLO11)
        # -------------------------------
        def get_map50(res):
            # YOLO11 구조
            try:
                if hasattr(res.metrics, "map50"):
                    return float(res.metrics.map50)
            except:
                pass

            # YOLO8 구조
            try:
                if hasattr(res.metrics, "box"):
                    return float(res.metrics.box.map50)
            except:
                pass

            # results_dict 구조
            try:
                d = res.results_dict
                # YOLO8 CSV에서의 key
                if "metrics/mAP50(B)" in d:
                    return float(d["metrics/mAP50(B)"])
            except:
                pass

            return None

        map50 = get_map50(results)
        if map50:
            self.log_signal.emit(f"✔ mAP50: {map50:.4f}")
        else:
            self.log_signal.emit("⚠ mAP50 찾지 못함 (YOLO 버전 차이 가능)")

        # -------------------------------
        # 학습 시간
        # -------------------------------
        train_time_sec = time.time() - start_time

        # -------------------------------
        # 파일 저장
        # -------------------------------
        run_dir = os.path.join(runs_dir, f"train_{timestamp}")
        best_src = os.path.join(run_dir, "weights", "best.pt")

        best_name = f"best_{timestamp}.pt"
        best_dst = os.path.join(models_dir, best_name)

        shutil.copy(best_src, best_dst)

        # history 저장
        hist_dir = os.path.join(history_dir, timestamp)
        os.makedirs(hist_dir, exist_ok=True)
        shutil.copy(best_src, os.path.join(hist_dir, "best.pt"))

        # -------------------------------
        # metadata.json 저장
        # -------------------------------
        meta = {
            "timestamp": timestamp,
            "data_yaml": self.data_yaml,
            "base_model": self.model_name,
            "epochs": self.epochs,
            "patience": self.patience,
            "models_file": best_dst,
            "run_dir": run_dir,
            "train_time_sec": train_time_sec,
            "map50": map50
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
        self.data_yaml = None
        self.update_paths(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🧪 Train Model")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # 데이터셋 표시
        self.dataset_label = QLabel("📂 data.yaml 선택되지 않음")
        layout.addWidget(self.dataset_label)

        btn_sel = QPushButton("📂 data.yaml 불러오기")
        btn_sel.clicked.connect(self.select_dataset)
        layout.addWidget(btn_sel)

        # 모델 선택
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("YOLO 모델 선택하기 :"))

        self.model_combo = QComboBox()
        models = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"
        ]
        for m in models:
            self.model_combo.addItem(m)

        row1.addWidget(self.model_combo)
        layout.addLayout(row1)

        # epochs
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Epochs:"))
        self.epoch_input = QLineEdit("30")
        row2.addWidget(self.epoch_input)
        layout.addLayout(row2)

        # patience
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Patience:"))
        self.patience_input = QLineEdit("10")
        row3.addWidget(self.patience_input)
        layout.addLayout(row3)

        # start button
        self.btn_start = QPushButton("🚀 학습 시작")
        self.btn_start.clicked.connect(self.start_training)
        layout.addWidget(self.btn_start)

        # log 출력창
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
        patience = int(self.patience_input.text())
        model_name = self.model_combo.currentText()

        self.btn_start.setEnabled(False)

        if self.overlay:
            self.overlay.show_overlay("🧪 모델 학습 중...")

        self.worker = TrainWorker(model_name, self.data_yaml, epochs, patience, self.paths)
        self.worker.log_signal.connect(self.log_box.append)
        self.worker.finished_ok.connect(self.on_model_saved)
        self.worker.finished.connect(self.training_done)

        self.worker.start()

    def training_done(self):
        if self.overlay:
            self.overlay.hide_overlay()
        self.btn_start.setEnabled(True)
        self.log_box.append("=== 학습 종료 ===")

    def on_model_saved(self, path: str):
        self.model_saved_signal.emit(path)
        self.log_box.append(f"✔ 모델 저장완료! : {path}")
