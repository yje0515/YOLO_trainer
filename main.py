import os
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget
)

from pages.dashboard import DashboardPage
from pages.dataset import DatasetPage
from pages.train import TrainPage
from pages.predict import PredictPage
from pages.history import HistoryPage
from pages.settings import SettingsPage, load_settings

# ⭐ 새로 추가된 페이지
from pages.model_comparison import ModelComparisonPage

from widgets.overlay import LoadingOverlay

# ======================
# matplotlib 한글 폰트 설정
# ======================
import matplotlib
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/malgun.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rc("font", family="Malgun Gothic")
else:
    matplotlib.rc("font", family="DejaVu Sans")

matplotlib.rcParams['axes.unicode_minus'] = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Trainer - By YJE")
        self.resize(1400, 850)

        # settings.json 로드
        self.settings = load_settings()

        # 메인 레이아웃
        layout = QHBoxLayout(self)

        # -----------------------------
        # 좌측 메뉴 (사이드바)
        # -----------------------------
        sidebar = QVBoxLayout()
        self.stack = QStackedWidget()

        # ⭐ 여기서 버튼 순서 + ModelList 추가
        self.btn_dashboard = QPushButton("🏠 Dashboard")
        self.btn_history = QPushButton("📚 History")
        self.btn_model_comparison = QPushButton("📈 Model Graph")
        self.btn_dataset = QPushButton("📁 Dataset")
        self.btn_train = QPushButton("🧪 Train")
        self.btn_predict = QPushButton("🔍 Predict")
        self.btn_settings = QPushButton("⚙ Settings")

        buttons = [
            self.btn_dashboard,     # index 0
            self.btn_history,       # index 1
            self.btn_model_comparison,    # index 2
            self.btn_dataset,       # index 3
            self.btn_train,         # index 4
            self.btn_predict,       # index 5
            self.btn_settings       # index 6
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
        self.page_dashboard = DashboardPage(self.settings)      # 0
        self.page_history = HistoryPage(self.settings)          # 1
        self.page_model_comparison = ModelComparisonPage(self.settings)     # 2 ⭐ 추가된 페이지
        self.page_dataset = DatasetPage(self.settings)          # 3
        self.page_train = TrainPage(self.settings)              # 4
        self.page_predict = PredictPage(self.settings)          # 5
        self.page_settings = SettingsPage()                     # 6

        # 페이지 스택에 추가
        self.stack.addWidget(self.page_dashboard)
        self.stack.addWidget(self.page_history)
        self.stack.addWidget(self.page_model_comparison)   # ⭐ 새로운 페이지
        self.stack.addWidget(self.page_dataset)
        self.stack.addWidget(self.page_train)
        self.stack.addWidget(self.page_predict)
        self.stack.addWidget(self.page_settings)

        layout.addWidget(self.stack, 4)

        # -----------------------------
        # 공통 로딩 오버레이
        # -----------------------------
        self.overlay = LoadingOverlay(self)

        # overlay 전달
        if hasattr(self.page_dataset, "set_overlay"):
            self.page_dataset.set_overlay(self.overlay)

        if hasattr(self.page_train, "set_overlay"):
            self.page_train.set_overlay(self.overlay)

        if hasattr(self.page_predict, "set_overlay"):
            self.page_predict.set_overlay(self.overlay)

        # -----------------------------
        # 시그널 연결
        # -----------------------------
        if hasattr(self.page_dataset, "dataset_ready") and hasattr(self.page_train, "set_dataset_path"):
            self.page_dataset.dataset_ready.connect(self.page_train.set_dataset_path)

        self.page_settings.settings_changed.connect(self.update_settings)

        if hasattr(self.page_train, "model_saved_signal"):
            self.page_train.model_saved_signal.connect(self.page_predict.refresh_model_list)
            self.page_train.model_saved_signal.connect(self.page_history.reload_history)
            self.page_train.model_saved_signal.connect(self.page_model_comparison.reload_models)  # ⭐ 모델리스트 갱신 추가
            self.page_train.model_saved_signal.connect(self.page_dashboard.reload_data)
            self.page_train.model_saved_signal.connect(self.page_dashboard.rebuild_ui)

        # 기본 페이지: Dashboard
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
            self.page_history,
            self.page_model_comparison   # ⭐ Model List 페이지도 반영해야 함
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
