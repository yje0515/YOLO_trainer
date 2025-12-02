# log_panel.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtCore import QDateTime


class LogPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        title = QLabel("📜 Logs")
        title.setStyleSheet("font-weight: bold;")

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(120)
        self.text.setStyleSheet("font-family: Consolas; font-size: 12px;")

        layout.addWidget(title)
        layout.addWidget(self.text)

    def log(self, message: str):
        now = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.text.append(f"[{now}] {message}")
