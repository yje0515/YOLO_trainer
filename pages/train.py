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


# ======================================================
# 🔥 데이터셋 자동 판별 (fire / human)
#   - 지금은 "기본값 제안" 용도로만 사용
# ======================================================
def detect_dataset_from_yaml(yaml_path: str) -> str:
    """
    data.yaml 내부의 폴더명을 기준으로 데이터셋을 자동 판별
      - fire → train/val 경로에 fire 문자열 포함
      - human → human 문자열 포함
    기본값: unknown
    """
    if not os.path.isfile(yaml_path):
        return "unknown"

    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        train_path = str(data.get("train", "")).lower()
        val_path = str(data.get("val", "")).lower()
        text = train_path + " " + val_path

        if "fire" in text:
            return "fire"
        if "human" in text:
            return "human"

    except Exception:
        pass

    return "unknown"


# ======================================================
# 학습 Worker Thread
# ======================================================
class TrainWorker(QThread):
    log_signal = Signal(str)
    finished_ok = Signal(str)

    def __init__(self, model_name, data_yaml, epochs, patience, paths: dict, dataset_name="unknown"):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.patience = patience
        self.paths = paths
        self.dataset_name = dataset_name   # fire / human / etc / unknown

    def run(self):
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

        runs_dir = self.paths["runs_dir"]
        models_dir = self.paths["models_dir"]
        history_dir = self.paths["history_dir"]

        os.makedirs(runs_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)

        # -------------------------
        # 로그 시작
        # -------------------------
        self.log_signal.emit(f"🧪 학습 시작 ({timestamp})")
        self.log_signal.emit(f"data.yaml: {self.data_yaml}")
        self.log_signal.emit(f"선택한 dataset 카테고리: {self.dataset_name}")

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

        # -------------------------
        # Device
        # -------------------------
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"Device: {device}")
        except:
            device = "cpu"
            self.log_signal.emit("CUDA 체크 실패 → CPU 사용")

        # -------------------------
        # Train 실행
        # -------------------------
        start_time = time.time()

        model = YOLO(self.model_name)

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
                save=True,
                exist_ok=True
            )
        except Exception as e:
            self.log_signal.emit(f"❌ 학습 실패: {e}")
            sys.stdout, sys.stderr = old_stdout, old_stderr
            return
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        # -------------------------
        # mAP50 계산
        # -------------------------
        def get_map50(res):
            try:
                if hasattr(res.metrics, "map50"):
                    return float(res.metrics.map50)
            except:
                pass
            try:
                if hasattr(res.metrics, "box"):
                    return float(res.metrics.box.map50)
            except:
                pass
            try:
                d = res.results_dict
                if "metrics/mAP50(B)" in d:
                    return float(d["metrics/mAP50(B)"])
            except:
                pass
            return None

        map50 = get_map50(results)
        if map50:
            self.log_signal.emit(f"✔ mAP50: {map50:.4f}")
        else:
            self.log_signal.emit("⚠ mAP50 찾지 못함")

        # -------------------------
        # 시간 계산
        # -------------------------
        train_time_sec = time.time() - start_time

        # -------------------------
        # Best 모델 저장
        # -------------------------
        run_dir = os.path.join(runs_dir, f"train_{timestamp}")
        best_src = os.path.join(run_dir, "weights", "best.pt")

        best_name = f"best_{timestamp}.pt"
        best_dst = os.path.join(models_dir, best_name)

        shutil.copy(best_src, best_dst)

        # -------------------------
        # history/{timestamp}/ 저장
        # -------------------------
        hist_dir = os.path.join(history_dir, timestamp)
        os.makedirs(hist_dir, exist_ok=True)
        shutil.copy(best_src, os.path.join(hist_dir, "best.pt"))

        # -------------------------
        # metadata.json 저장
        # -------------------------
        meta = {
            "timestamp": timestamp,
            "data_yaml": self.data_yaml,
            "dataset": self.dataset_name,     # 🔥 사용자가 선택한 카테고리
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


# ======================================================
# Train Page UI
# ======================================================
class TrainPage(QWidget):
    model_saved_signal = Signal(str)

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None
        self.data_yaml = None
        self.dataset_name = "unknown"
        self.update_paths(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🧪 Train Model")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # data.yaml 표시
        self.dataset_label = QLabel("📂 data.yaml 선택되지 않음")
        layout.addWidget(self.dataset_label)

        btn_sel = QPushButton("📂 data.yaml 불러오기")
        btn_sel.clicked.connect(self.select_dataset)
        layout.addWidget(btn_sel)

        # -------------------------
        # Dataset Category 선택 (필수)
        # -------------------------
        row_ds = QHBoxLayout()
        row_ds.addWidget(QLabel("Dataset Category:"))

        self.dataset_combo = QComboBox()
        # 첫 항목은 '선택하세요' → 이 상태면 학습 불가
        self.dataset_combo.addItem("카테고리 선택")
        self.dataset_combo.addItem("fire")
        self.dataset_combo.addItem("human")
        self.dataset_combo.addItem("etc")
        self.dataset_combo.addItem("unknown")
        row_ds.addWidget(self.dataset_combo)

        layout.addLayout(row_ds)

        # 모델 선택
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("YOLO 모델 선택하기 :"))
        self.model_combo = QComboBox()
        for m in [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"
        ]:
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

        # 로그
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-family:Consolas; font-size:12px;")
        layout.addWidget(self.log_box)

        layout.addStretch()

    def set_overlay(self, overlay):
        self.overlay = overlay

    def update_paths(self, settings: dict):
        self.paths = settings

    # Dataset 선택 (DatasetPage에서 signal로도 들어오고, 직접 선택도 가능)
    def set_dataset_path(self, path: str):
        self.data_yaml = path
        self.dataset_label.setText(f"📂 선택된 data.yaml: {path}")

        # 자동 감지 결과를 기본 선택값으로 제안만 해줌
        auto_ds = detect_dataset_from_yaml(path)
        self.dataset_name = auto_ds

        # 콤보박스 쪽에도 추천값 반영 (있으면 변경)
        if hasattr(self, "dataset_combo"):
            idx = self.dataset_combo.findText(auto_ds)
            if idx >= 0:
                self.dataset_combo.setCurrentIndex(idx)
            else:
                # fire/human/etc/unknown 중 없는 값이면 unknown으로
                idx2 = self.dataset_combo.findText("unknown")
                if idx2 >= 0:
                    self.dataset_combo.setCurrentIndex(idx2)

    def select_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", ".", "YAML (*.yaml)"
        )
        if path:
            self.set_dataset_path(path)

    def start_training(self):
        if not self.data_yaml:
            self.log_box.append("❌ data.yaml 선택 후 학습이 가능합니다.")
            return

        # 🔥 Dataset 카테고리 필수 선택
        current_ds = self.dataset_combo.currentText()
        if current_ds == "카테고리 선택":
            self.log_box.append("❌ Dataset 카테고리를 먼저 선택해주세요. (fire / human / etc / unknown)")
            return

        # 최종 선택값 반영
        self.dataset_name = current_ds

        try:
            epochs = int(self.epoch_input.text())
            patience = int(self.patience_input.text())
        except ValueError:
            self.log_box.append("❌ Epochs / Patience는 정수로 입력해주세요.")
            return

        model_name = self.model_combo.currentText()

        self.btn_start.setEnabled(False)

        if self.overlay:
            self.overlay.show_overlay("🧪 모델 학습 중...")

        self.worker = TrainWorker(
            model_name,
            self.data_yaml,
            epochs,
            patience,
            self.paths,
            dataset_name=self.dataset_name
        )
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
