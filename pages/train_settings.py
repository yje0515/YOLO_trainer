from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QSlider, QHBoxLayout
from PySide6.QtCore import Qt


class TrainSettingsPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🛠 Training Settings")

        # 모델 선택
        model_label = QLabel("YOLO 모델 선택")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])

        # Epoch 슬라이더
        epoch_label = QLabel("Epoch")
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setRange(1, 300)
        self.epoch_slider.setValue(50)

        # Patience 슬라이더
        pat_label = QLabel("Early Stopping Patience")
        self.pat_slider = QSlider(Qt.Horizontal)
        self.pat_slider.setRange(1, 100)
        self.pat_slider.setValue(20)

        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(model_label)
        layout.addWidget(self.model_combo)
        layout.addSpacing(20)
        layout.addWidget(epoch_label)
        layout.addWidget(self.epoch_slider)
        layout.addSpacing(20)
        layout.addWidget(pat_label)
        layout.addWidget(self.pat_slider)
        layout.addStretch()
