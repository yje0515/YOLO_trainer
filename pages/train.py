# pages/train.py

import os
import sys
import shutil
import json
import datetime
import csv

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QComboBox, QLineEdit, QFileDialog
)

from ultralytics import YOLO


class TrainWorker(QThread):
    log_signal = Signal(str)
    finished_ok = Signal(str)   # best 모델 경로

    def __init__(self, model_name, data_yaml, epochs, patience,
                 runs_dir, models_dir, history_dir, use_gpu=True):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.patience = patience
        self.runs_dir = runs_dir
        self.models_dir = models_dir
        self.history_dir = history_dir
        self.use_gpu = use_gpu

    def run(self):
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

        try:
            self.log_signal.emit(f"🧪 학습 시작 → {timestamp}")
            self.log_signal.emit(f"📂 data.yaml: {self.data_yaml}")
            self.log_signal.emit(f"🧠 base model: {self.model_name}")

            # YOLO 모델 로드
            model = YOLO(self.model_name)

            # 디바이스 결정
            device = "cpu"
            try:
                import torch
                if self.use_gpu and torch.cuda.is_available():
                    device = "0"
                    self.log_signal.emit("⚡ GPU(CUDA) 사용: device=0")
                else:
                    self.log_signal.emit("💻 CPU 모드로 학습합니다.")
            except Exception:
                self.log_signal.emit("💻 torch 확인 실패 → CPU 모드로 학습합니다.")

            # runs 디렉토리 준비
            os.makedirs(self.runs_dir, exist_ok=True)

            # 학습 실행
            model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                patience=self.patience,
                imgsz=640,
                batch=8,
                device=device,
                project=self.runs_dir,
                name=f"train_{timestamp}",
                exist_ok=True
            )

            # 최신 run 디렉토리 찾기
            run_dir = self.get_latest_run_dir()
            if not run_dir:
                self.log_signal.emit("⚠ run 디렉토리를 찾지 못했습니다.")
                return

            self.log_signal.emit(f"📂 run 디렉토리: {run_dir}")

            best_src = os.path.join(run_dir, "weights", "best.pt")
            if not os.path.exists(best_src):
                self.log_signal.emit("⚠ best.pt 파일이 없습니다.")
                return

            os.makedirs(self.models_dir, exist_ok=True)
            best_name = f"best_{timestamp}.pt"
            best_dst = os.path.join(self.models_dir, best_name)
            shutil.copy(best_src, best_dst)

            self.log_signal.emit(f"✔ best 모델 저장됨 → {best_dst}")

            # history/<timestamp>/ 아래에 결과 저장
            hist_dir = os.path.join(self.history_dir, timestamp)
            os.makedirs(hist_dir, exist_ok=True)

            # best.pt 복사
            hist_best = os.path.join(hist_dir, "best.pt")
            shutil.copy(best_src, hist_best)

            # results.csv 복사 + 메트릭 추출
            metrics = {}
            csv_src = os.path.join(run_dir, "results.csv")
            if os.path.exists(csv_src):
                csv_dst = os.path.join(hist_dir, "results.csv")
                shutil.copy(csv_src, csv_dst)

                with open(csv_src, "r", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if len(rows) >= 2:
                    header, last = rows[0], rows[-1]
                    metrics = {header[i]: last[i] for i in range(len(header))}

            meta_path = os.path.join(hist_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": timestamp,
                        "data_yaml": self.data_yaml,
                        "base_model": self.model_name,
                        "models_file": best_dst,
                        "history_best": hist_best,
                        "run_dir": run_dir,
                        "epochs": self.epochs,
                        "patience": self.patience,
                        "metrics": metrics,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            self.log_signal.emit(f"📚 history 저장됨 → {hist_dir}")
            self.log_signal.emit("✅ 학습 완료")
            self.finished_ok.emit(best_dst)

        except Exception as e:
            self.log_signal.emit(f"❌ 학습 오류: {e}")

    def get_latest_run_dir(self):
        if not os.path.isdir(self.runs_dir):
            return None

        dirs = []
        for name in os.listdir(self.runs_dir):
            full = os.path.join(self.runs_dir, name)
            if os.path.isdir(full):
                mtime = os.path.getmtime(full)
                dirs.append((mtime, full))

        if not dirs:
            return None
        dirs.sort(reverse=True)
        return dirs[0][1]


class TrainPage(QWidget):
    train_log_signal = Signal(str)
    model_saved_signal = Signal(str)  # best 모델 저장 시

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None
        self.dataset_path = None
        self.worker = None

        self.update_paths(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🧪 Train YOLO Model")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        info = QLabel(
            "Dataset 탭에서 데이터셋을 다운로드하거나,\n"
            "아래에서 data.yaml을 직접 선택한 뒤 학습을 시작하세요."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # 현재 data.yaml 표시
        self.dataset_label = QLabel("📂 선택된 data.yaml: (없음)")
        layout.addWidget(self.dataset_label)

        # data.yaml 선택 버튼
        btn_sel = QPushButton("📂 data.yaml 선택")
        btn_sel.clicked.connect(self.select_dataset)
        layout.addWidget(btn_sel)

        # 모델 선택
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Base 모델:"))
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

        # 상태 표시
        self.status_label = QLabel("상태: 대기")
        layout.addWidget(self.status_label)

        # 학습 버튼
        self.btn_train = QPushButton("🚀 학습 시작")
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

        layout.addStretch()

    # MainWindow에서 overlay 주입
    def set_overlay(self, overlay):
        self.overlay = overlay

    # Settings 변경 시 호출
    def update_paths(self, settings: dict):
        self.runs_dir = settings.get("runs_dir", "./runs")
        self.models_dir = settings.get("models_dir", "./models")
        self.history_dir = settings.get("history_dir", "./history")
        self.dataset_dir = settings.get("dataset_dir", "./datasets")

        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)

    # DatasetPage → TrainPage로 data.yaml 세팅
    def set_dataset_path(self, path: str):
        self.dataset_path = path
        self.dataset_label.setText(f"📂 선택된 data.yaml: {path}")
        self.train_log_signal.emit(f"✔ data.yaml 설정됨: {path}")

    # 직접 data.yaml 선택
    def select_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data.yaml",
            self.dataset_dir,
            "YAML Files (*.yaml)"
        )
        if not path:
            return
        self.set_dataset_path(path)

    # 학습 시작
    def start_training(self):
        if not self.dataset_path:
            self.train_log_signal.emit("❌ data.yaml이 선택되지 않았습니다.")
            return

        model_name = self.model_combo.currentText()
        try:
            epochs = int(self.epoch_input.text())
            patience = int(self.pat_input.text())
        except ValueError:
            self.train_log_signal.emit("❌ Epoch/Patience는 정수로 입력하세요.")
            return

        self.status_label.setText("상태: 학습 중...")
        self.train_log_signal.emit(
            f"🚀 학습 시작 (model={model_name}, epochs={epochs}, patience={patience})"
        )

        if self.overlay:
            self.overlay.show_overlay("🧪 YOLO 모델 학습 중...")

        self.worker = TrainWorker(
            model_name=model_name,
            data_yaml=self.dataset_path,
            epochs=epochs,
            patience=patience,
            runs_dir=self.runs_dir,
            models_dir=self.models_dir,
            history_dir=self.history_dir,
            use_gpu=True
        )
        self.worker.log_signal.connect(self.train_log_signal.emit)
        self.worker.finished_ok.connect(self.on_train_finished_ok)
        self.worker.finished.connect(self.on_train_finished_anyway)
        self.worker.start()

    def on_train_finished_ok(self, best_path: str):
        self.model_saved_signal.emit(best_path)
        self.train_log_signal.emit(f"🎉 새 모델 저장: {best_path}")

    def on_train_finished_anyway(self):
        self.status_label.setText("상태: 대기")
        if self.overlay:
            self.overlay.hide_overlay()
