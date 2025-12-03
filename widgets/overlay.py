import os
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtGui import QMovie
from PySide6.QtCore import Qt


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background-color: rgba(0,0,0,120);")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # GIF (더 작게)
        self.gif = QLabel()
        self.gif.setAlignment(Qt.AlignCenter)

        gif_path = os.path.abspath("resources/hamtory.gif")
        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(Qt.QSize(120,120))  # ← 크기 줄이기
            self.gif.setMovie(self.movie)
        else:
            self.gif.setText("GIF Missing!")

        # Text
        self.text = QLabel("작업 중...")
        self.text.setStyleSheet("color:white; font-size:18px; font-weight:bold;")
        self.text.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.gif)
        layout.addWidget(self.text)

    def show_overlay(self, msg="작업 중..."):
        self.text.setText(msg)
        self.setGeometry(self.parent().rect())
        self.show()
        if hasattr(self, "movie"):
            self.movie.start()

    def hide_overlay(self):
        self.hide()
        if hasattr(self, "movie"):
            self.movie.stop()
