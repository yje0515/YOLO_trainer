from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit


class LogPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 10)
        layout.setSpacing(4)

        self.label = QLabel("📜 Log")
        self.edit = QTextEdit()
        self.edit.setReadOnly(True)
        self.edit.setFixedHeight(150)

        layout.addWidget(self.label)
        layout.addWidget(self.edit)

    def log(self, text: str):
        self.edit.append(text)
