import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget
)

from pages.dataset import DatasetPage
from pages.train import TrainPage
from pages.predict import PredictPage
from pages.dashboard import DashboardPage
from pages.history import HistoryPage
from pages.settings import SettingsPage, load_settings

from widgets.overlay import LoadingOverlay


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Trainer By YJE")
        self.resize(1300, 850)

        # SETTINGS 로드
        self.settings = load_settings()

        # 전체 레이아웃
        layout = QHBoxLayout(self)

        # -------------------------------
        # 좌측 메뉴
        # -------------------------------
        left = QVBoxLayout()
        btn_dashboard = QPushButton("📊 Dashboard")
        btn_dataset = QPushButton("📁 Dataset")
        btn_train = QPushButton("🧪 Train")
        btn_predict = QPushButton("🔍 Predict")
        btn_history = QPushButton("📜 History")
        btn_settings = QPushButton("⚙ Settings")

        for b in [btn_dashboard, btn_dataset, btn_train,
                  btn_predict, btn_history, btn_settings]:
            b.setMinimumHeight(45)
            left.addWidget(b)

        left.addStretch()
        layout.addLayout(left, 1)

        # -------------------------------
        # 오른쪽 스택 페이지
        # -------------------------------
        self.stack = QStackedWidget()

        self.page_dashboard = DashboardPage()
        self.page_dataset = DatasetPage(self.settings)
        self.page_train = TrainPage(self.settings)
        self.page_predict = PredictPage(self.settings)
        self.page_history = HistoryPage(self.settings)
        self.page_settings = SettingsPage()

        self.stack.addWidget(self.page_dashboard)  # 0
        self.stack.addWidget(self.page_dataset)    # 1
        self.stack.addWidget(self.page_train)      # 2
        self.stack.addWidget(self.page_predict)    # 3
        self.stack.addWidget(self.page_history)    # 4
        self.stack.addWidget(self.page_settings)   # 5

        layout.addWidget(self.stack, 5)

        # -------------------------------
        # 메뉴 클릭 연결
        # -------------------------------
        btn_dashboard.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_dataset.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_train.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        btn_predict.clicked.connect(lambda: self.stack.setCurrentIndex(3))
        btn_history.clicked.connect(lambda: self.stack.setCurrentIndex(4))
        btn_settings.clicked.connect(lambda: self.stack.setCurrentIndex(5))

        # -------------------------------
        # 오버레이 생성 (전 페이지 공통)
        # -------------------------------
        self.overlay = LoadingOverlay(self)

        # 각 페이지에 공유
        self.page_train.set_overlay(self.overlay)
        self.page_dataset.set_overlay(self.overlay)
        self.page_predict.set_overlay(self.overlay)

        # 설정 변경 → 전체 반영
        self.page_settings.settings_changed.connect(self.update_settings)

        # 데이터셋 선택 → TrainPage 전달
        self.page_dataset.dataset_ready.connect(self.page_train.set_dataset_path)

    # 설정 변경 반영
    def update_settings(self, new_settings):
        self.settings = new_settings
        self.page_train.update_paths(new_settings)
        self.page_dataset.update_paths(new_settings)
        self.page_predict.update_paths(new_settings)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
