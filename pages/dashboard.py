from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📊 Dashboard")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)

        desc = QLabel("최근 학습 기록, 상태 요약 등이 들어갈 대시보드 화면입니다.")
        desc.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch()
