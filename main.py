import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget
)

from pages.dashboard import DashboardPage
    # 경로 반영하기 전체 페이지 업데이트
from pages.dataset import DatasetPage
from pages.train import TrainPage
from pages.predict import PredictPage
from pages.history import HistoryPage
from pages.settings import SettingsPage, load_settings

from widgets.overlay import LoadingOverlay


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Trainer - By YJE")
        self.resize(1300, 850)

        # settings.json 로드
        self.settings = load_settings()

        # 메인 레이아웃
        layout = QHBoxLayout(self)

        # -----------------------------
        # 좌측 메뉴 (사이드바)
        # -----------------------------
        sidebar = QVBoxLayout()
        self.stack = QStackedWidget()

        self.btn_dashboard = QPushButton("📊 Dashboard")
        self.btn_dataset = QPushButton("📁 Dataset")
        self.btn_train = QPushButton("🧪 Train")
        self.btn_predict = QPushButton("🔍 Predict")
        self.btn_history = QPushButton("📜 History")
        self.btn_settings = QPushButton("⚙ Settings")

        buttons = [
            self.btn_dashboard, self.btn_dataset, self.btn_train,
            self.btn_predict, self.btn_history, self.btn_settings
        ]

        for idx, btn in enumerate(buttons):
            btn.setMinimumHeight(45)
            btn.clicked.connect(lambda _, i=idx: self.stack.setCurrentIndex(i))
            sidebar.addWidget(btn)

        sidebar.addStretch()
        layout.addLayout(sidebar, 1)

        # -----------------------------
        # 페이지 스택 (우측 화면)
        # -----------------------------
        self.page_dashboard = DashboardPage(self.settings)
        self.page_dataset = DatasetPage(self.settings)
        self.page_train = TrainPage(self.settings)
        self.page_predict = PredictPage(self.settings)
        self.page_history = HistoryPage(self.settings)
        self.page_settings = SettingsPage()

        self.stack.addWidget(self.page_dashboard)  # index: 0
        self.stack.addWidget(self.page_dataset)    # index: 1
        self.stack.addWidget(self.page_train)      # index: 2
        self.stack.addWidget(self.page_predict)    # index: 3
        self.stack.addWidget(self.page_history)    # index: 4
        self.stack.addWidget(self.page_settings)   # index: 5

        layout.addWidget(self.stack, 4)

        # -----------------------------
        # 공통 로딩 오버레이(햄토리)
        # -----------------------------
        self.overlay = LoadingOverlay(self)

        # overlay 전달 (Dataset, Train, Predict 페이지 필요)
        if hasattr(self.page_dataset, "set_overlay"):
            self.page_dataset.set_overlay(self.overlay)

        if hasattr(self.page_train, "set_overlay"):
            self.page_train.set_overlay(self.overlay)

        if hasattr(self.page_predict, "set_overlay"):
            self.page_predict.set_overlay(self.overlay)

        # -----------------------------
        # 시그널 연결
        # -----------------------------

        # Dataset → Train : data.yaml 경로 전달
        if hasattr(self.page_dataset, "dataset_ready") and hasattr(self.page_train, "set_dataset_path"):
            self.page_dataset.dataset_ready.connect(self.page_train.set_dataset_path)

        # Settings 변경 → 모든 페이지 업데이트
        self.page_settings.settings_changed.connect(self.update_settings)

        # Train에서 모델 저장 시 Predict/History 갱신
        if hasattr(self.page_train, "model_saved_signal"):
            self.page_train.model_saved_signal.connect(self.page_predict.refresh_model_list)
            self.page_train.model_saved_signal.connect(self.page_history.reload_history)

        # 기본 페이지
        self.stack.setCurrentIndex(0)

    # ---------------------------------------------------------
    # settings.json 이 변경되면 전체 페이지에 반영
    # ---------------------------------------------------------
    def update_settings(self, new_settings: dict):
        self.settings = new_settings

        pages = [
            self.page_dashboard,
            self.page_dataset,
            self.page_train,
            self.page_predict,
            self.page_history
        ]

        for page in pages:
            if hasattr(page, "update_paths"):
                page.update_paths(new_settings)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
