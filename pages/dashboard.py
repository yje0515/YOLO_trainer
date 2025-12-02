# pages/dashboard.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)

        title = QLabel("🏠 YOLO Trainer Dashboard")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        desc = QLabel(
            "이 프로그램은 Roboflow에서 데이터셋을 받아와서\n"
            "YOLO 모델을 학습하고, 학습 이력을 관리하고, 예측까지 할 수 있는\n"
            "지은님 개인용 데스크톱 트레이너입니다."
        )
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch()
