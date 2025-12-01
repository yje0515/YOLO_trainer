from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PySide6.QtCore import Signal, Qt


class DatasetPage(QWidget):
    run_code_signal = Signal(str)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(12)

        # 제목
        title = QLabel("📁 Dataset (Roboflow 코드 실행)")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        # 설명 라벨 (윗부분)
        desc = QLabel("Roboflow 데이터셋 다운로드 코드를 아래 박스에 그대로 넣고 실행하세요.\n"
                      "예시)")
        desc.setWordWrap(True)

        # 예시 코드 박스
        example_code_html = """
        <div style="
            background-color:#f8f8f8;
            border:2px solid #333;
            border-radius: 6px;
            padding:12px;
            font-family:Consolas;
            font-size:13px;
            white-space: pre;
            color:#000;
        ">
        from roboflow import Roboflow
        rf = Roboflow(api_key="XXX")
        project = rf.workspace("workspace").project("project")
        version = project.version(3)
        dataset = version.download("yolov8")
        </div>
        """

        example_label = QLabel()
        example_label.setText(example_code_html)
        example_label.setTextFormat(Qt.RichText)
        example_label.setWordWrap(True)

        # 코드 입력 박스
        self.code_edit = QTextEdit()
        self.code_edit.setPlaceholderText("여기에 Roboflow Python 코드를 붙여넣으세요.")
        self.code_edit.setMinimumHeight(250)

        # 실행 버튼
        run_btn = QPushButton("🚀 코드 실행")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self.run_code)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addSpacing(8)
        layout.addWidget(example_label)   # 라운드 코드 박스
        layout.addSpacing(15)
        layout.addWidget(self.code_edit)
        layout.addWidget(run_btn)
        layout.addStretch()

    def run_code(self):
        code = self.code_edit.toPlainText().strip()
        if not code:
            self.run_code_signal.emit("[Dataset] 실행할 코드가 없습니다.")
            return

        self.run_code_signal.emit("=== Roboflow 코드 실행 시작 ===")

        try:
            exec_globals = {}
            exec(code, exec_globals)
            self.run_code_signal.emit("✔ 실행 성공!")
            self.run_code_signal.emit("=== 데이터셋 다운로드 완료 ===")
        except Exception as e:
            self.run_code_signal.emit(f"❌ 오류 발생: {e}")
