import os
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtGui import QMovie
from PySide6.QtCore import Qt, QSize


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background-color: rgba(0, 0, 0, 140);")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # 햄토리 GIF (작게 중앙)
        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignCenter)

        gif_path = os.path.abspath("resources/hamtory.gif")
        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(QSize(120, 120))  # 크기 줄이기
            self.gif_label.setMovie(self.movie)
        else:
            self.gif_label.setText("hamtory.gif 없음")

        # 텍스트
        self.text_label = QLabel("작업 중...")
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("color:white; font-size:18px; font-weight:bold;")

        layout.addWidget(self.gif_label)
        layout.addWidget(self.text_label)

    def show_overlay(self, message: str = "작업 중..."):
        self.text_label.setText(message)
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.show()
        if hasattr(self, "movie"):
            self.movie.start()

    def hide_overlay(self):
        self.hide()
        if hasattr(self, "movie"):
            self.movie.stop()
