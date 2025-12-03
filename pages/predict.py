# pages/predict.py

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QFileDialog, QComboBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Signal, QUrl

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from ultralytics import YOLO


class PredictPage(QWidget):
    predict_log_signal = Signal(str)

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None
        self.media_path = None

        self.update_paths(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🔮 Predict (이미지 / 영상)")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # 모델 선택
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("모델 선택:"))
        self.model_combo = QComboBox()
        r1.addWidget(self.model_combo)
        layout.addLayout(r1)

        # 모델 리스트 채우기
        self.refresh_model_list()

        # 미디어 선택 버튼
        btn_media = QPushButton("📂 이미지 / 영상 선택")
        btn_media.clicked.connect(self.select_media)
        layout.addWidget(btn_media)

        # 예측 버튼
        btn_predict = QPushButton("🚀 Predict 실행")
        btn_predict.clicked.connect(self.run_predict)
        layout.addWidget(btn_predict)

        # 이미지 프리뷰
        self.image_preview = QLabel()
        self.image_preview.setFixedHeight(320)
        self.image_preview.setStyleSheet("border:1px solid gray; background-color:#111;")
        self.image_preview.setScaledContents(True)
        layout.addWidget(self.image_preview)

        # 영상 위젯
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(320)
        layout.addWidget(self.video_widget)

        # 미디어 플레이어
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        self.video_widget.hide()  # 기본은 숨김

        layout.addStretch()

    # overlay 주입
    def set_overlay(self, overlay):
        self.overlay = overlay

    # settings 변경 시
    def update_paths(self, settings: dict):
        self.models_dir = settings.get("models_dir", "./models")
        self.predict_output_dir = settings.get("predict_output_dir", "./predict_output")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.predict_output_dir, exist_ok=True)

    # models 폴더에서 .pt 리스트 가져오기
    def refresh_model_list(self):
        self.model_combo.clear()
        if not os.path.isdir(self.models_dir):
            self.model_combo.addItem("(모델 없음)")
            return

        files = sorted(f for f in os.listdir(self.models_dir)
                       if f.lower().endswith(".pt"))
        if not files:
            self.model_combo.addItem("(모델 없음)")
        else:
            self.model_combo.addItems(files)

    # 이미지/영상 선택
    def select_media(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image/Video",
            ".",
            "Media Files (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv)"
        )
        if not file_path:
            return

        self.media_path = file_path
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.video_widget.hide()
            self.media_player.stop()
            self.image_preview.show()
            self.image_preview.setPixmap(QPixmap(file_path))
            self.predict_log_signal.emit(f"✔ 이미지 선택: {file_path}")
        else:
            self.image_preview.hide()
            self.video_widget.show()
            self.predict_log_signal.emit(f"✔ 영상 선택: {file_path}")

    # Predict 실행
    def run_predict(self):
        if not self.media_path:
            self.predict_log_signal.emit("❌ 먼저 이미지 또는 영상을 선택하세요.")
            return

        self.refresh_model_list()  # 혹시 새 모델이 추가되었을 수도 있으니

        model_name = self.model_combo.currentText()
        if model_name == "(모델 없음)":
            self.predict_log_signal.emit("❌ models 폴더에 모델이 없습니다.")
            return

        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            self.predict_log_signal.emit(f"❌ 모델 파일을 찾을 수 없음: {model_path}")
            return

        self.predict_log_signal.emit(f"🔍 모델 로드: {model_path}")
        model = YOLO(model_path)

        ext = os.path.splitext(self.media_path)[1].lower()

        if self.overlay:
            self.overlay.show_overlay("🔮 예측 중...")

        try:
            # 이미지
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                out_dir = os.path.join(self.predict_output_dir, "image")
                os.makedirs(out_dir, exist_ok=True)

                model.predict(
                    self.media_path,
                    save=True,
                    project=self.predict_output_dir,
                    name="image",
                    exist_ok=True
                )

                out_img = self.get_latest_file(out_dir,
                                               (".jpg", ".jpeg", ".png", ".bmp"))
                if out_img:
                    self.video_widget.hide()
                    self.media_player.stop()
                    self.image_preview.show()
                    self.image_preview.setPixmap(QPixmap(out_img))
                    self.predict_log_signal.emit(f"✔ 결과 이미지: {out_img}")
                else:
                    self.predict_log_signal.emit("⚠ 결과 이미지를 찾지 못했습니다.")

            # 영상
            else:
                out_dir = os.path.join(self.predict_output_dir, "video")
                os.makedirs(out_dir, exist_ok=True)

                model.predict(
                    self.media_path,
                    save=True,
                    project=self.predict_output_dir,
                    name="video",
                    exist_ok=True
                )

                out_vid = self.get_latest_file(out_dir,
                                               (".mp4", ".avi", ".mov", ".mkv"))
                if out_vid:
                    self.image_preview.hide()
                    self.video_widget.show()

                    url = QUrl.fromLocalFile(os.path.abspath(out_vid))
                    self.media_player.setSource(url)
                    self.media_player.play()

                    self.predict_log_signal.emit(f"✔ 결과 영상: {out_vid}")
                    self.predict_log_signal.emit("▶ 영상 재생 중...")
                else:
                    self.predict_log_signal.emit("⚠ 결과 영상을 찾지 못했습니다.")
        finally:
            if self.overlay:
                self.overlay.hide_overlay()

    def get_latest_file(self, folder, exts):
        if not os.path.isdir(folder):
            return None
        candidates = []
        for name in os.listdir(folder):
            if name.lower().endswith(exts):
                full = os.path.join(folder, name)
                mtime = os.path.getmtime(full)
                candidates.append((mtime, full))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
