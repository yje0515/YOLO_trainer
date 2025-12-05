import os
import datetime
import cv2
import numpy as np
import json

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QTextEdit, QHBoxLayout, QSlider
)

from ultralytics import YOLO


# ====================================
#   실시간 YOLO Predict Worker
# ====================================
class PredictWorker(QThread):
    frame_ready = Signal(np.ndarray)
    finished_ok = Signal(str)
    log_signal = Signal(str)

    def __init__(self, model_path, source_path, save_dir, conf: float = 0.5):
        super().__init__()
        self.model_path = model_path
        self.source_path = source_path
        self.save_dir = save_dir
        self.conf = conf

    def run(self):
        model = YOLO(self.model_path)

        results = model.predict(
            source=self.source_path,
            save=True,
            project=self.save_dir,
            name="media",
            conf=self.conf,
            stream=True,
            exist_ok=True,
            verbose=False
        )

        for r in results:
            annotated = r.plot()
            self.frame_ready.emit(annotated)

        final_dir = os.path.join(self.save_dir, "media")

        # metadata 저장
        metadata = {
            "model_path": self.model_path,
            "source_path": self.source_path,
            "save_dir": final_dir,
            "conf": self.conf
        }
        try:
            with open(os.path.join(self.save_dir, "predict_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log_signal.emit(f"❌ metadata 저장 실패: {e}")

        self.finished_ok.emit(final_dir)


# ====================================
#   Predict Page
# ====================================
class PredictPage(QWidget):

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None
        self.paths = settings

        self.latest_train_timestamp = None
        self.selected_path = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🔍 Predict")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # -------------------------------------------------
        # 1) 모델 선택
        # -------------------------------------------------
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("모델 선택:"))
        self.model_combo = QComboBox()
        row1.addWidget(self.model_combo)
        layout.addLayout(row1)

        # -------------------------------------------------
        # 2) Confidence 슬라이더
        # -------------------------------------------------
        row_conf = QHBoxLayout()
        row_conf.addWidget(QLabel("Confidence:"))

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickInterval(5)
        self.conf_slider.setSingleStep(1)
        row_conf.addWidget(self.conf_slider)

        self.conf_label = QLabel("50% 이상만 표시")
        row_conf.addWidget(self.conf_label)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)

        layout.addLayout(row_conf)

        # -------------------------------------------------
        # 3) 파일 선택
        # -------------------------------------------------
        btn_file = QPushButton("📂 이미지/영상 선택")
        btn_file.clicked.connect(self.select_file)
        layout.addWidget(btn_file)

        # -------------------------------------------------
        # 4) 미리보기
        # -------------------------------------------------
        self.previewLabel = QLabel("미리보기 없음")
        self.previewLabel.setFixedHeight(300)
        self.previewLabel.setAlignment(Qt.AlignCenter)
        self.previewLabel.setStyleSheet(
            "border:1px solid #444; background:#111; color:#888;"
        )
        layout.addWidget(self.previewLabel)

        # -------------------------------------------------
        # 5) 실행 버튼
        # -------------------------------------------------
        btn_run = QPushButton("🚀 추론 실행")
        btn_run.clicked.connect(self.run_predict)
        layout.addWidget(btn_run)

        # -------------------------------------------------
        # 6) 로그
        # -------------------------------------------------
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        layout.addStretch()

        # 모델 목록 로딩
        self.refresh_model_list()

    # =======================================================
    def set_overlay(self, overlay):
        self.overlay = overlay

    # =======================================================
    def update_paths(self, settings: dict):
        self.paths = settings
        self.refresh_model_list()

    # =======================================================
    # Dataset별 모델 분류 + 최신순 정렬 + 최신 모델 강조
    # =======================================================
    def refresh_model_list(self, _=None):
        self.model_combo.clear()
        models_dir = self.paths.get("models_dir", "")
        history_dir = self.paths.get("history_dir", "")

        if not os.path.isdir(models_dir) or not os.path.isdir(history_dir):
            return

        # metadata 기반 모델 목록 구성
        grouped = {"fire": [], "human": [], "etc": [], "unknown": []}
        metadata_map = {}
        timestamps = []

        for folder in os.listdir(history_dir):
            meta_path = os.path.join(history_dir, folder, "metadata.json")
            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except:
                continue

            dataset = meta.get("dataset", "unknown")
            model_file = os.path.basename(meta.get("models_file"))
            timestamp = meta.get("timestamp")

            metadata_map[model_file] = {
                "dataset": dataset,
                "timestamp": timestamp
            }
            timestamps.append(timestamp)

            if dataset not in grouped:
                grouped["etc"].append(model_file)
            else:
                grouped[dataset].append(model_file)

        # 최신 timestamp 찾기
        if timestamps:
            timestamps.sort(reverse=True)
            self.latest_train_timestamp = timestamps[0]
        else:
            self.latest_train_timestamp = None

        # ---------------------------------------------------
        # QComboBox 구성
        # ---------------------------------------------------
        def add_header(text):
            self.model_combo.addItem(text)
            idx = self.model_combo.count() - 1
            item = self.model_combo.model().item(idx)
            item.setEnabled(False)
            item.setForeground(Qt.gray)

        dataset_labels = {
            "fire": "🔥 Fire Models",
            "human": "🧍 Human Models",
            "etc": "📦 ETC Models",
            "unknown": "❓ Unknown Models"
        }

        # dataset별 최신순 정렬
        for ds, label in dataset_labels.items():
            models = grouped[ds]
            if not models:
                continue

            # 최신순 (metadata timestamp 기준)
            models.sort(key=lambda m: metadata_map[m]["timestamp"], reverse=True)

            add_header(f"--- {label} ---")

            for model_file in models:
                display_text = f"{metadata_map[model_file]['timestamp']} | {model_file}"
                self.model_combo.addItem(display_text)

                # 최신 모델 강조
                if metadata_map[model_file]["timestamp"] == self.latest_train_timestamp:
                    idx = self.model_combo.count() - 1
                    item = self.model_combo.model().item(idx)
                    item.setBackground(Qt.cyan)

    # =======================================================
    def on_conf_changed(self, value: int):
        self.conf_label.setText(f"{value}% 이상만 표시")

    # =======================================================
    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image or Video", ".", "Files (*.jpg *.png *.mp4 *.avi)"
        )
        if path:
            self.selected_path = path
            self.log_box.append(f"📂 선택됨: {path}")
            self.show_preview(path)

    def show_preview(self, path: str):
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self.previewLabel.setPixmap(
                    pixmap.scaled(
                        self.previewLabel.width(),
                        self.previewLabel.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            return

        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()

        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            self.previewLabel.setPixmap(
                pixmap.scaled(
                    self.previewLabel.width(),
                    self.previewLabel.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    # =======================================================
    def run_predict(self):
        if not self.selected_path:
            self.log_box.append("❌ 먼저 파일을 선택해주세요.")
            return

        display_text = self.model_combo.currentText()
        if "---" in display_text:
            self.log_box.append("❌ 모델을 선택해주세요.")
            return

        model_file = display_text.split("|")[1].strip()
        model_path = os.path.join(self.paths["models_dir"], model_file)

        if not self.latest_train_timestamp:
            self.log_box.append("❌ train 기록이 없습니다.")
            return

        predict_root = os.path.join(
            self.paths["history_dir"],
            self.latest_train_timestamp,
            "predict_log"
        )
        os.makedirs(predict_root, exist_ok=True)

        now_dir_name = datetime.datetime.now().strftime("predict_%y%m%d_%H%M")
        save_dir = os.path.join(predict_root, now_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        conf = self.conf_slider.value() / 100.0

        if self.overlay:
            self.overlay.show_overlay("🔍 추론 중...")

        self.worker = PredictWorker(
            model_path=model_path,
            source_path=self.selected_path,
            save_dir=save_dir,
            conf=conf,
        )
        self.worker.frame_ready.connect(self.update_preview)
        self.worker.finished_ok.connect(self.predict_finished)
        self.worker.start()

    # =======================================================
    def update_preview(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape

        qimg = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.previewLabel.setPixmap(
            pixmap.scaled(
                self.previewLabel.width(),
                self.previewLabel.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # =======================================================
    def predict_finished(self, final_dir):
        self.log_box.append(f"✔ 결과 저장 완료: {final_dir}")

        if self.overlay:
            self.overlay.hide_overlay()
