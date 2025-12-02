import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget

from sidebar import Sidebar
from log_panel import LogPanel

from pages.dashboard import DashboardPage
from pages.dataset import DatasetPage
from pages.train import TrainPage
from pages.predict import PredictPage
from pages.history import HistoryPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Trainer (Python Desktop)")
        self.resize(1300, 800)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        top_area = QWidget()
        top_layout = QHBoxLayout(top_area)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.menu_clicked.connect(self.change_page)

        self.stack = QStackedWidget()

        self.page_dashboard = DashboardPage()
        self.page_dataset = DatasetPage()
        self.page_train = TrainPage()
        self.page_predict = PredictPage()
        self.page_history = HistoryPage()

        self.stack.addWidget(self.page_dashboard)  # 0
        self.stack.addWidget(self.page_dataset)    # 1
        self.stack.addWidget(self.page_train)      # 2
        self.stack.addWidget(self.page_predict)    # 3
        self.stack.addWidget(self.page_history)    # 4

        top_layout.addWidget(self.sidebar)
        top_layout.addWidget(self.stack, stretch=1)

        self.log_panel = LogPanel()

        container_layout.addWidget(top_area, stretch=1)
        container_layout.addWidget(self.log_panel, stretch=0)

        self.setCentralWidget(container)

        # 📌 DatasetPage → TrainPage 데이터셋 전달
        self.page_dataset.dataset_ready.connect(self.page_train.set_dataset_path)

        # 📌 로그 연결
        self.page_train.train_log_signal.connect(self.log_panel.log)
        self.page_predict.predict_log_signal.connect(self.log_panel.log)

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
