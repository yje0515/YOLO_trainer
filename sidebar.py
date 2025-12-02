# sidebar.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal


class Sidebar(QWidget):
    menu_clicked = Signal(int)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        self.buttons = []

        menu_list = [
            "🏠 Dashboard",
            "📁 Dataset",
            "🧪 Train",
            "📚 History",
            "🔍 Predict",
        ]

        for index, text in enumerate(menu_list):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E9E9E9;
                    border: 1px solid #CCCCCC;
                    padding: 8px;
                    font-size: 14px;
                    text-align: left;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #F5F5F5;
                }
            """)
            btn.clicked.connect(lambda checked, i=index: self.on_button_clicked(i))
            self.buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()
        self.current_index = None

    def on_button_clicked(self, index: int):
        self.set_active(index)
        self.menu_clicked.emit(index)

    def set_active(self, index: int):
        for i, btn in enumerate(self.buttons):
            if i == index:
                btn.setChecked(True)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #A7D8FF;
                        border: 1px solid #6BB6FF;
                        padding: 8px;
                        font-size: 14px;
                        text-align: left;
                        border-radius: 6px;
                    }
                    QPushButton:hover {
                        background-color: #9CD0FF;
                    }
                """)
            else:
                btn.setChecked(False)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #E9E9E9;
                        border: 1px solid #CCCCCC;
                        padding: 8px;
                        font-size: 14px;
                        text-align: left;
                        border-radius: 6px;
                    }
                    QPushButton:hover {
                        background-color: #F5F5F5;
                    }
                """)
        self.current_index = index
