import os
import datetime
import cv2
import numpy as np

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
        self.save_dir = save_dir     # 최종 저장 디렉토리 (predict_log/predict_xxxx)
        self.conf = conf

    def run(self):
        model = YOLO(self.model_path)

        # 실시간 + 저장
        results = model.predict(
            source=self.source_path,
            save=True,
            project=self.save_dir,   # predict_log/predict_xxxx
            name="media",            # predict_xxxx/media 안에 저장됨
            conf=self.conf,
            stream=True,
            exist_ok=True,
            verbose=False
        )

        for r in results:
            annotated = r.plot()     # YOLO가 그린 BGR frame
            self.frame_ready.emit(annotated)

        final_dir = os.path.join(self.save_dir, "media")
        # 🔥 predict_metadata.json 저장 (추가)
        metadata = {
            "model_path": self.model_path,
            "source_path": self.source_path,
            "save_dir": final_dir,
            "conf": self.conf
        }

        meta_path = os.path.join(self.save_dir, "predict_metadata.json")
        try:
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log_signal.emit(f"❌ metadata 저장 실패: {e}")
        self.finished_ok.emit(final_dir)


# ====================================
#   Predict Page UI
# ====================================
class PredictPage(QWidget):

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None
        self.paths = settings

        # 최신 train timestamp 가져오기 위해 저장
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

        # 초기 모델 목록 로딩
        self.refresh_model_list()

    # =======================================================
    # main.py에서 overlay를 받기 위한 함수
    # =======================================================
    def set_overlay(self, overlay):
        self.overlay = overlay

    # =======================================================
    # settings 변경 시 반영
    # =======================================================
    def update_paths(self, settings: dict):
        self.paths = settings
        self.refresh_model_list()

    # =======================================================
    # 모델 목록 리프레시
    # =======================================================
    def refresh_model_list(self, _=None):
        self.model_combo.clear()
        models_dir = self.paths.get("models_dir", "")

        if not os.path.exists(models_dir):
            return

        for f in os.listdir(models_dir):
            if f.endswith(".pt"):
                self.model_combo.addItem(f)

        # 최신 train timestamp 찾아서 저장
        history_dir = self.paths.get("history_dir", "")
        self.latest_train_timestamp = self._get_latest_train_timestamp(history_dir)

    # train 기록 중 최신 폴더명(timestamp) 가져오기
    def _get_latest_train_timestamp(self, history_dir):
        if not os.path.isdir(history_dir):
            return None

        timestamps = []
        for name in os.listdir(history_dir):
            sub = os.path.join(history_dir, name)
            if os.path.isdir(sub):
                timestamps.append(name)

        if not timestamps:
            return None

        # timestamp 내림차순 정렬
        try:
            timestamps.sort(reverse=True)
            return timestamps[0]
        except:
            return None

    # =======================================================
    # Confidence 슬라이더
    # =======================================================
    def on_conf_changed(self, value: int):
        self.conf_label.setText(f"{value}% 이상만 표시")

    # =======================================================
    # 파일 선택 → 미리보기 표시
    # =======================================================
    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image or Video", ".", "Files (*.jpg *.png *.mp4 *.avi)"
        )
        if path:
            self.selected_path = path
            self.log_box.append(f"📂 선택됨: {path}")
            self.show_preview(path)

    # -------------------------------------------------------
    # 미리보기 표시 (이미지/영상 첫 프레임)
    # -------------------------------------------------------
    def show_preview(self, path: str):
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self.previewLabel.setPixmap(
                    pixmap.scaled(
                        self.previewLabel.width(),
                        self.previewLabel.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
            return

        # 영상
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
                    Qt.SmoothTransformation
                )
            )

    # =======================================================
    # 🔥 Predict 실행
    # =======================================================
    def run_predict(self):
        if not self.selected_path:
            self.log_box.append("❌ 먼저 파일을 선택해주세요.")
            return

        model_file = self.model_combo.currentText()
        if not model_file:
            self.log_box.append("❌ 사용할 모델이 없습니다.")
            return

        if not self.latest_train_timestamp:
            self.log_box.append("❌ train 기록을 찾지 못했습니다.")
            return

        # --------------------------------------------------
        # 경로 구성 (A안)
        #   history/{timestamp}/predict_log/predict_YYMMDD_HHMM/
        # --------------------------------------------------
        predict_root = os.path.join(
            self.paths["history_dir"],
            self.latest_train_timestamp,
            "predict_log"
        )
        os.makedirs(predict_root, exist_ok=True)

        now_dir_name = datetime.datetime.now().strftime("predict_%y%m%d_%H%M")
        save_dir = os.path.join(predict_root, now_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------------------------------
        # 모델 파일 경로
        # --------------------------------------------------
        model_path = os.path.join(self.paths["models_dir"], model_file)

        # conf
        conf_percent = self.conf_slider.value()
        conf = conf_percent / 100.0

        self.log_box.append(f"⚙ Confidence: {conf_percent}%")
        self.log_box.append(f"📁 저장 경로: {save_dir}")

        if self.overlay:
            self.overlay.show_overlay("🔍 추론 중...")

        # Worker 실행
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
    # 실시간 프레임 업데이트
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
                Qt.SmoothTransformation
            )
        )

    # =======================================================
    # predict 완료 콜백
    # =======================================================
    def predict_finished(self, final_dir):
        self.log_box.append(f"✔ 결과 저장 완료: {final_dir}")

        if self.overlay:
            self.overlay.hide_overlay()
