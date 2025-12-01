from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog


class PredictPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🔍 Predict (이미지/영상)")

        desc = QLabel("학습된 모델로 이미지/영상을 Predict하는 페이지입니다.")
        desc.setWordWrap(True)

        self.btn_model = QPushButton("📂 모델 선택")
        self.btn_input = QPushButton("🖼 입력 파일 선택")
        self.btn_predict = QPushButton("🔍 Predict 실행")

        # 파일 저장
        self.btn_model.clicked.connect(self.select_model)
        self.btn_input.clicked.connect(self.select_input)
        self.btn_predict.clicked.connect(lambda: print("[TODO] Predict 실행"))

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addSpacing(10)
        layout.addWidget(self.btn_model)
        layout.addWidget(self.btn_input)
        layout.addWidget(self.btn_predict)
        layout.addStretch()

    def select_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "Model (*.pt)")
        if file:
            print("모델:", file)

    def select_input(self):
        file, _ = QFileDialog.getOpenFileName(self, "입력 선택", "", "Images/Video (*.jpg *.png *.mp4)")
        if file:
            print("입력파일:", file)
