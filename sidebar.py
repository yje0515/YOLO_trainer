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
            "📊 Dashboard",
            "📁 Dataset",
            "🛠 Training Settings",
            "🧪 Train Model",
            "🔍 Predict"
        ]

        for index, text in enumerate(menu_list):
            btn = QPushButton(text)
            btn.setCheckable(True)

            # 기본 버튼 스타일 (라이트 그레이톤)
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

    def on_button_clicked(self, index):
        self.set_active(index)
        self.menu_clicked.emit(index)

    def set_active(self, index):
        """활성 버튼만 하늘색 + 눌린 느낌으로 표시"""
        for i, btn in enumerate(self.buttons):
            if i == index:
                # 눌린버튼처럼 보이게 inset-style 효과 적용
                btn.setChecked(True)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #A7D8FF;          /* 밝은 하늘색 */
                        border: 1px solid #6BB6FF;         /* 조금 더 진한 파랑 */
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
                # 디폴트 디자인으로 복원
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
