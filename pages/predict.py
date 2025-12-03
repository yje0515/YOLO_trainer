import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QFileDialog, QComboBox
)
from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QPixmap

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from ultralytics import YOLO


class PredictPage(QWidget):
    def __init__(self, settings: dict):
        super().__init__()
        self.update_paths(settings)
        self.overlay = None
        self.media_path = None

        # ===============================
        #  --- 메인 레이아웃 (상단 정렬!!)
        # ===============================
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)   # ★ 상단 정렬로 고정

        # ---------------------------
        # 제목
        # ---------------------------
        title = QLabel("🔍 Predict Model")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # ---------------------------
        # 모델 선택
        # ---------------------------
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("모델 선택:"))
        self.model_combo = QComboBox()
        row1.addWidget(self.model_combo)
        layout.addLayout(row1)

        btn_refresh = QPushButton("🔄 모델 새로고침(최신순)")
        btn_refresh.clicked.connect(self.refresh_model_list)
        layout.addWidget(btn_refresh)

        btn_file = QPushButton("📂 모델 파일 불러오기")
        btn_file.clicked.connect(self.load_model_file)
        layout.addWidget(btn_file)

        # ---------------------------
        # 미디어 선택
        # ---------------------------
        btn_media = QPushButton("📂 이미지 / 영상 선택")
        btn_media.clicked.connect(self.select_media)
        layout.addWidget(btn_media)

        # 선택된 파일 표시
        self.path_label = QLabel("📂 선택된 파일: 없음")
        layout.addWidget(self.path_label)

        # ---------------------------
        # Predict 실행
        # ---------------------------
        btn_predict = QPushButton("🚀 Predict 실행")
        btn_predict.clicked.connect(self.run_predict)
        layout.addWidget(btn_predict)

        # ---------------------------
        # 이미지 미리보기
        # ---------------------------
        self.image_preview = QLabel()
        self.image_preview.setMinimumHeight(320)
        self.image_preview.setStyleSheet("border:1px solid gray; background:black;")
        self.image_preview.setAlignment(Qt.AlignCenter)    # ★ 비율 유지
        self.image_preview.setScaledContents(False)        # ★ 찌그러짐 방지
        layout.addWidget(self.image_preview)

        # ---------------------------
        # 영상 미리보기
        # ---------------------------
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(320)
        layout.addWidget(self.video_widget)
        self.video_widget.hide()

        # Video Player
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setAudioOutput(self.audio_output)

        # 모델 리스트 로드
        self.refresh_model_list()

    # ---------------------------
    # overlay 연결
    # ---------------------------
    def set_overlay(self, overlay):
        self.overlay = overlay

    # ---------------------------
    # 경로 업데이트
    # ---------------------------
    def update_paths(self, settings):
        self.models_dir = settings["models_dir"]
        self.predict_output = settings["predict_output_dir"]

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.predict_output, exist_ok=True)

    # ---------------------------
    # 모델 새로고침 (최신순)
    # ---------------------------
    def refresh_model_list(self):
        self.model_combo.clear()

        if not os.path.isdir(self.models_dir):
            self.model_combo.addItem("(모델 없음)")
            return

        files = [
            f for f in os.listdir(self.models_dir)
            if f.lower().endswith(".pt")
        ]

        files = sorted(
            files,
            key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)),
            reverse=True
        )

        if not files:
            self.model_combo.addItem("(모델 없음)")
        else:
            self.model_combo.addItems(files)

    # ---------------------------
    # 모델 파일 직접 불러오기
    # ---------------------------
    def load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "모델 파일 선택", ".", "PyTorch Model (*.pt)"
        )
        if not path:
            return

        name = os.path.basename(path)
        dst = os.path.join(self.models_dir, name)

        if not os.path.exists(dst):
            shutil.copy(path, dst)

        self.refresh_model_list()
        self.model_combo.setCurrentText(name)

    # ---------------------------
    # 이미지 / 영상 선택
    # ---------------------------
    def select_media(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image or Video",
            ".",
            "Media (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv)"
        )

        if not file_path:
            return

        self.media_path = file_path
        self.path_label.setText(f"📂 선택된 파일: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.media_player.stop()
            self.video_widget.hide()
            self.image_preview.show()

            pix = QPixmap(file_path)
            self.image_preview.setPixmap(pix)

        else:
            # 영상
            self.image_preview.hide()
            self.video_widget.show()

            self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(file_path)))
            self.media_player.play()

    # ---------------------------
    # 최신 생성 파일 찾기
    # ---------------------------
    def get_latest(self, folder, extensions):
        if not os.path.isdir(folder):
            return None

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ]
        if not files:
            return None

        return max(files, key=os.path.getmtime)

    # ---------------------------
    # Predict 실행
    # ---------------------------
    def run_predict(self):
        if not self.media_path:
            return

        model_name = self.model_combo.currentText()
        if model_name == "(모델 없음)":
            return

        model_path = os.path.join(self.models_dir, model_name)
        model = YOLO(model_path)

        if self.overlay:
            self.overlay.show_overlay("🔮 추론 중...")

        ext = os.path.splitext(self.media_path)[1].lower()

        try:
            # ---------------------------
            # 이미지 예측
            # ---------------------------
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                out_dir = os.path.join(self.predict_output, "img_out")
                os.makedirs(out_dir, exist_ok=True)

                model.predict(
                    self.media_path,
                    save=True,
                    project=out_dir,
                    name="result",
                    exist_ok=True
                )

                result_dir = os.path.join(out_dir, "result")
                latest = self.get_latest(result_dir, (".jpg", ".png"))

                if latest:
                    self.video_widget.hide()
                    self.image_preview.show()

                    pix = QPixmap(latest)
                    self.image_preview.setPixmap(pix)

            # ---------------------------
            # 영상 예측
            # ---------------------------
            else:
                out_dir = os.path.join(self.predict_output, "video_out")
                os.makedirs(out_dir, exist_ok=True)

                model.predict(
                    self.media_path,
                    save=True,
                    project=out_dir,
                    name="result",
                    exist_ok=True
                )

                result_dir = os.path.join(out_dir, "result")
                latest = self.get_latest(result_dir, (".mp4", ".avi", ".mov"))

                if latest:
                    self.image_preview.hide()
                    self.video_widget.show()

                    self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(latest)))
                    self.media_player.play()

        finally:
            if self.overlay:
                self.overlay.hide_overlay()
