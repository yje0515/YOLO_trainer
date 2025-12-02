# pip install pyside6
# pip install matplotlib
# pip install ultralytics
# pip install roboflow

# exe 실행 파일 얻기 (프로젝트 최종 완성 후)
# pip install pyinstaller
# pyinstaller --noconsole --onefile main.py

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget

from sidebar import Sidebar
from log_panel import LogPanel

from pages.dashboard import DashboardPage
from pages.dataset import DatasetPage
from pages.train_settings import TrainSettingsPage
from pages.train_model import TrainModelPage
from pages.predict import PredictPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Trainer (Python Desktop)")
        self.resize(1300, 800)

        # 메인 컨테이너
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # 상단 영역 (사이드바 + 페이지 스택)
        top_area = QWidget()
        top_layout = QHBoxLayout(top_area)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # 사이드바
        self.sidebar = Sidebar()
        self.sidebar.menu_clicked.connect(self.change_page)

        # 스택 페이지
        self.stack = QStackedWidget()
        self.page_dashboard = DashboardPage()
        self.page_dataset = DatasetPage()
        self.page_train_settings = TrainSettingsPage()
        self.page_train_model = TrainModelPage()
        self.page_predict = PredictPage()

        self.stack.addWidget(self.page_dashboard)       # 0
        self.stack.addWidget(self.page_dataset)         # 1
        self.stack.addWidget(self.page_train_settings)  # 2
        self.stack.addWidget(self.page_train_model)     # 3
        self.stack.addWidget(self.page_predict)         # 4

        top_layout.addWidget(self.sidebar)
        top_layout.addWidget(self.stack, stretch=1)

        # 로그 패널
        self.log_panel = LogPanel()

        # 전체 구성
        container_layout.addWidget(top_area, stretch=1)
        container_layout.addWidget(self.log_panel, stretch=0)

        self.setCentralWidget(container)

        # ⬇⬇⬇ 신호 연결 추가 (중요)
        self.page_dataset.run_code_signal.connect(self.log_panel.log)
        self.page_train_model.train_log_signal.connect(self.log_panel.log)
        # ⬆⬆⬆ 이거 있어야 로그 다 뜸!

        # 기본 페이지 = Dashboard
        self.change_page(0)

    def change_page(self, index):
        self.stack.setCurrentIndex(index)
        self.log_panel.log(f"페이지 전환 → {index}번 페이지")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
