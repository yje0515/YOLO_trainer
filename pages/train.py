import os
import shutil
import json
import datetime
import sys
import io
import time
import re  # 🔥 Epoch 로그 파싱용

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QComboBox, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar
)

from ultralytics import YOLO


# ======================================================
# 🔧 시간 포맷 헬퍼 (mm:ss / hh:mm:ss)
# ======================================================
def format_time(sec: float | int | None) -> str:
    if sec is None:
        return "-"
    try:
        sec = float(sec)
    except Exception:
        return "-"
    if sec < 0:
        sec = 0
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


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

    # 🔥 진행 상황 실시간 전파
    # elapsed_sec, expected_total_sec, current_epoch, total_epochs
    progress_signal = Signal(float, float, int, int)

    def __init__(self, model_name, data_yaml, epochs, patience, paths: dict, dataset_name="unknown"):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.patience = patience
        self.paths = paths
        self.dataset_name = dataset_name   # fire / human / etc / unknown

        # ---- 진행률/ETA 계산용 내부 상태 ----
        self._start_time: float | None = None        # 학습 전체 시작 시각(초)
        self._prepare_end_time: float | None = None  # 이미지 스캔/준비 끝난 시각
        self._epoch1_end_time: float | None = None   # Epoch 1 종료 시각(= Epoch2 등장 시점)
        self._expected_total_time: float | None = None  # 예상 총 학습시간 (초)
        self._first_epoch_seen_time: float | None = None
        self.current_epoch: int = 0
        self.total_epochs: int = epochs

    # --------------------------------------------------
    # 🔥 로그 한 줄이 들어올 때마다 호출되는 헬퍼
    #   - Epoch 파싱
    #   - 준비/1에포크 시간 측정
    #   - 예상 총 학습시간 계산
    #   - 진행률 시그널 전송
    # --------------------------------------------------
    def _handle_log_line(self, line: str):
        now = time.time()

        # 최초 로그 시각 = 전체 학습 시작 시각으로 사용
        if self._start_time is None:
            self._start_time = now

        # "  1/30 " 이런 형식의 Epoch 로그 파싱
        m = re.search(r"^\s*(\d+)/(\d+)\s", line)
        if m:
            ep = int(m.group(1))
            total = int(m.group(2))
            # 총 Epoch 정보 업데이트
            if total > 0:
                self.total_epochs = total

            # 현재 Epoch 갱신
            if ep > self.current_epoch:
                self.current_epoch = ep

            # Epoch 1이 처음 보이는 시점 = 준비 끝/학습 시작 지점으로 간주
            if ep == 1 and self._first_epoch_seen_time is None:
                self._first_epoch_seen_time = now
                if self._prepare_end_time is None:
                    self._prepare_end_time = now

            # Epoch 2 이상이 처음 보이는 시점 = Epoch 1 종료 시점으로 간주
            if ep >= 2 and self._epoch1_end_time is None:
                self._epoch1_end_time = now
                # 혹시라도 prepare_end_time이 비어있다면 첫 epoch 등장 시각 기준으로 보정
                if self._prepare_end_time is None:
                    self._prepare_end_time = self._first_epoch_seen_time or self._start_time or now

                # 🔥 예상 총 학습시간 계산 (보수적으로)
                if self._start_time is not None and self._prepare_end_time is not None:
                    t_prepare = self._prepare_end_time - self._start_time
                else:
                    t_prepare = 0.0
                t_epoch1 = self._epoch1_end_time - (self._prepare_end_time or self._start_time or self._epoch1_end_time)

                # 기본 공식: 준비시간 + (1 Epoch 순수 학습시간 × 총 Epoch 수)
                expected_total = t_prepare + max(t_epoch1, 0.1) * self.total_epochs
                self._expected_total_time = expected_total

                # 로그에 한 번 안내
                self.log_signal.emit(
                    f"⏳ 1에포크 기준 예상 총 학습시간: 약 {format_time(expected_total)}"
                )

        # 매 로그마다 진행률 업데이트
        self._emit_progress(now)

    # --------------------------------------------------
    # 🔥 진행률/ETA 시그널 발행
    # --------------------------------------------------
    def _emit_progress(self, now: float | None = None, force_done: bool = False):
        if now is None:
            now = time.time()
        if self._start_time is None:
            return

        elapsed = now - self._start_time
        expected = self._expected_total_time

        # 학습이 모두 끝난 뒤 post-processing 중일 때 강제로 100% 근처로 맞춰주기
        if force_done:
            if expected is None or expected < elapsed:
                expected = elapsed
            self._expected_total_time = expected

        # 예상 시간이 아직 없으면, Epoch 비율로만 대략 진행률 표시
        if not expected or expected <= 0:
            if self.total_epochs > 0:
                frac = min(1.0, self.current_epoch / float(self.total_epochs))
                progress = int(frac * 100)
            else:
                progress = 0
            expected = 0.0
        else:
            progress = int(min(100, (elapsed / expected) * 100))

        # UI 쪽에서 퍼센트는 다시 계산할 수 있도록, 여기선 시간/epoch 정보만 보냄
        self.progress_signal.emit(elapsed, expected, self.current_epoch, self.total_epochs)

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
            def __init__(self, callback, owner: "TrainWorker"):
                self.callback = callback
                self.buffer = ""
                self.owner = owner

            def write(self, text):
                self.buffer += text
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    if line.strip():
                        # 1) 로그 출력
                        self.callback(line.strip())
                        # 2) ETA/진행률 갱신
                        self.owner._handle_log_line(line.strip())
                return len(text)

            def flush(self):
                if self.buffer.strip():
                    self.callback(self.buffer.strip())
                    self.owner._handle_log_line(self.buffer.strip())
                    self.buffer = ""

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Redirect(self.log_signal.emit, self)
        sys.stderr = Redirect(self.log_signal.emit, self)

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
        self._start_time = start_time  # 진행률 계산에 사용

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

        # 🔥 학습 루프는 끝났지만, 아직 파일 복사/메타 저장 작업이 남아있으므로
        # 여기서 한 번 더 "100% 근처"로 진행률 보정
        self._emit_progress(now=time.time(), force_done=True)

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
        end_time = time.time()
        train_time_sec = end_time - start_time

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
        self.log_signal.emit(f"⏱ 실제 학습 시간: {format_time(train_time_sec)}")
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

        # 🔥 ETA(종료시각) 계산용: 학습 시작 시각
        self.train_start_wall: datetime.datetime | None = None
        self.eta_logged = False  # 로그창에 ETA 한 번만 찍기 위한 플래그

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
            # ── Object Detection ──
            "yolov8n.pt",
            "yolov8s.pt",
            "yolo11n.pt",
            "yolo11s.pt",

            # ── Pose 모델 ──
            "yolov8n-pose.pt",
            "yolov8s-pose.pt",
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
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

        # 🔥 진행률 ProgressBar + 상태 라벨
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel(
            "진행률: 0%  |  경과 00:00 / 예상 -  (Epoch 0/0)  |  종료 예정시각: -"
        )
        self.progress_label.setStyleSheet("color:#555; font-size:12px;")
        layout.addWidget(self.progress_label)

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

    # --------------------------------------------------
    # 🔥 진행률/ETA 업데이트 슬롯
    # --------------------------------------------------
    def on_progress_update(self, elapsed_sec: float, expected_sec: float, current_epoch: int, total_epochs: int):
        # 퍼센트 계산
        if expected_sec and expected_sec > 0:
            progress = int(min(100, (elapsed_sec / expected_sec) * 100))
        else:
            if total_epochs > 0:
                progress = int(min(100, (current_epoch / float(total_epochs)) * 100))
            else:
                progress = 0

        self.progress_bar.setValue(progress)

        # 🔥 예상 종료 시각 계산 (start_wall + expected_total_sec)
        eta_str = "-"
        if expected_sec and expected_sec > 0 and self.train_start_wall is not None:
            try:
                eta_dt = self.train_start_wall + datetime.timedelta(seconds=expected_sec)
                eta_str = eta_dt.strftime("%H:%M")
            except Exception:
                eta_str = "-"

            # 로그에 한 번만 찍기
            if not self.eta_logged:
                self.log_box.append(f"🕒 예상 종료 시각: {eta_str} 기준 (보수적 추정)")
                self.eta_logged = True

        self.progress_label.setText(
            f"진행률: {progress}%  |  경과 {format_time(elapsed_sec)} / "
            f"예상 {format_time(expected_sec if expected_sec > 0 else None)}  "
            f"(Epoch {current_epoch}/{total_epochs})  |  종료 예정시각: {eta_str}"
        )

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

        # 진행률/ETA 초기화
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            "진행률: 0%  |  경과 00:00 / 예상 -  (Epoch 0/0)  |  종료 예정시각: -"
        )
        self.train_start_wall = datetime.datetime.now()
        self.eta_logged = False

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
        # 🔥 진행률 연결
        self.worker.progress_signal.connect(self.on_progress_update)

        self.worker.start()

    def training_done(self):
        if self.overlay:
            self.overlay.hide_overlay()
        self.btn_start.setEnabled(True)
        self.log_box.append("=== 학습 종료 ===")

        # 혹시 100%가 아니라면, 종료 시점에서 100%로 마무리
        if self.progress_bar.value() < 100:
            self.progress_bar.setValue(100)

    def on_model_saved(self, path: str):
        self.model_saved_signal.emit(path)
        self.log_box.append(f"✔ 모델 저장완료! : {path}")
