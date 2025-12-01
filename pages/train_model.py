from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class TrainModelPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🧪 Train Model")
        desc = QLabel("YOLO 모델을 학습시키는 페이지입니다. (추후 연동)")
        desc.setWordWrap(True)

        self.btn_train = QPushButton("🚀 학습 시작 (나중에 연결)")

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addSpacing(20)
        layout.addWidget(self.btn_train)
        layout.addStretch()
