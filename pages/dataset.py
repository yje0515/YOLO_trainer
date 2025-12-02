# pages/dataset.py
import os
import re
import subprocess
import tempfile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QFileDialog, QPlainTextEdit
)
from PySide6.QtCore import Qt, Signal


class DatasetPage(QWidget):
    dataset_ready = Signal(str)  # data.yaml 경로를 TrainPage로 전달

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📁 Dataset")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "1) Roboflow에서 받은 Python 코드를 복붙 후 '실행'하면 자동 다운로드 후 해당 데이터셋이 적용됩니다.\n"
            "2) 또는 '📂 data.yaml 직접 선택' 버튼을 눌러 기존 데이터셋을 불러올 수 있습니다."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # 입력 박스
        self.code_edit = QPlainTextEdit()
        self.code_edit.setPlaceholderText(
            "예시)\n"
            "from roboflow import Roboflow\n"
            "rf = Roboflow(api_key=\"XXX\")\n"
            "project = rf.workspace(\"workspace\").project(\"project\")\n"
            "version = project.version(3)\n"
            "dataset = version.download(\"yolov8\")"
        )
        layout.addWidget(self.code_edit)

        # Roboflow 실행 버튼
        self.btn_run = QPushButton("▶ 데이터셋 다운로드 실행 (Roboflow)")
        self.btn_run.clicked.connect(self.run_script)
        layout.addWidget(self.btn_run)

        # 🔥 data.yaml 직접 선택 버튼
        self.btn_select_yaml = QPushButton("📂 data.yaml 직접 선택")
        self.btn_select_yaml.clicked.connect(self.select_yaml_file)
        layout.addWidget(self.btn_select_yaml)

        # 출력 영역
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family: Consolas; font-size: 12px;")
        layout.addWidget(self.output)

    ############################################################
    # 📂 1) data.yaml 직접 선택 기능
    ############################################################
    def select_yaml_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", ".", "YAML Files (*.yaml)"
        )
        if not file_path:
            return

        self.output.append(f"✔ data.yaml 선택됨: {file_path}")
        self.dataset_ready.emit(file_path)

    ############################################################
    # ▶ 2) Roboflow 코드 실행 기능
    ############################################################
    # 사용자로부터 입력된 데이터셋코드를 받아와 데이터셋 다운로드
    def run_script(self):
        raw_code = self.code_edit.toPlainText().strip()
        if not raw_code:
            self.output.append("❌ 오류: Roboflow 코드를 입력하세요.")
            return

        self.output.append("\n⏳ Roboflow 스크립트 실행 중...\n")

        # 임시 파이썬 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp:
            tmp.write(raw_code)
            tmp_path = tmp.name

        try:
            python_exe = os.sys.executable

            proc = subprocess.Popen(
                [python_exe, tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            out, err = proc.communicate()

            if out:
                self.output.append(out)
            if err:
                self.output.append("❗ 오류:\n" + err)
                self.output.append("❗ 데이터셋 다운로드 실패 \n")
                self.output.append("❗ 코드를 확인 후 다시 다운로드해 주세요.")
            else: # 오류가 나지 않으면
                # data.yaml 찾기
                yaml_path = self.find_yaml()
                if yaml_path:
                    self.output.append(f"\n✔ 데이터셋 다운로드 완료!\n✔ data.yaml: {yaml_path}")
                    self.dataset_ready.emit(yaml_path)
                else:
                    self.output.append("\n⚠ data.yaml을 찾을 수 없습니다.")


        finally:
            os.remove(tmp_path)

    ############################################################
    # data.yaml 자동 탐색
    ############################################################
    def find_yaml(self):
        for root, dirs, files in os.walk(".", topdown=True):
            if "data.yaml" in files:
                return os.path.join(root, "data.yaml")
        return None
