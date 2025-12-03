# pages/dataset.py

import os
import subprocess
import tempfile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QFileDialog, QTextEdit
)
from PySide6.QtCore import Qt, Signal


class DatasetPage(QWidget):
    # data.yaml 경로를 밖으로 알려주는 시그널 (TrainPage에서 받음)
    dataset_ready = Signal(str)

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None

        # 경로 설정
        self.update_paths(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)

        title = QLabel("📁 Dataset (Roboflow 코드 실행 & data.yaml 선택)")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        desc = QLabel(
            "1) Roboflow에서 받은 Python 코드를 아래에 붙여넣고 실행하면,\n"
            "   지정한 데이터셋 폴더에 자동으로 다운로드됩니다.\n"
            "2) 이미 다운로드된 data.yaml이 있다면 직접 선택할 수도 있습니다."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # 코드 입력 박스
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

        # 실행 버튼
        self.btn_run = QPushButton("▶ Roboflow 코드 실행 (데이터셋 다운로드)")
        self.btn_run.clicked.connect(self.run_script)
        layout.addWidget(self.btn_run)

        # data.yaml 직접 선택
        self.btn_select_yaml = QPushButton("📂 data.yaml 직접 선택")
        self.btn_select_yaml.clicked.connect(self.select_yaml_file)
        layout.addWidget(self.btn_select_yaml)

        # 출력 로그
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family:Consolas; font-size:12px;")
        layout.addWidget(self.output)

        layout.addStretch()

    # MainWindow에서 햄토리 오버레이를 넘겨줄 때 호출
    def set_overlay(self, overlay):
        self.overlay = overlay

    # Settings 변경 시 호출
    def update_paths(self, settings: dict):
        self.dataset_dir = settings.get("dataset_dir", "./datasets")
        self.temp_dir = settings.get("temp_dir", "./temp")
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    # data.yaml을 직접 선택
    def select_yaml_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data.yaml",
            self.dataset_dir,
            "YAML Files (*.yaml)"
        )
        if not file_path:
            return

        self.output.append(f"✔ data.yaml 선택됨: {file_path}")
        self.dataset_ready.emit(file_path)

    # Roboflow 코드 실행
    def run_script(self):
        raw_code = self.code_edit.toPlainText().strip()
        if not raw_code:
            self.output.append("❌ 오류: Roboflow 코드를 입력하세요.")
            return

        # 오버레이 표시
        if self.overlay:
            self.overlay.show_overlay("📥 데이터셋 다운로드 중...")

        try:
            self.output.append("\n⏳ 스크립트 실행 중...\n")

            # 임시 파이썬 파일 생성 (temp_dir 안에)
            os.makedirs(self.temp_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                suffix=".py",
                dir=self.temp_dir,
                text=True
            )
            os.close(fd)

            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(raw_code)

            python_exe = os.sys.executable

            # cwd = dataset_dir 로 지정해서, Roboflow가 이 폴더 밑에 다운로드하게 함
            proc = subprocess.Popen(
                [python_exe, tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=self.dataset_dir
            )

            out, err = proc.communicate()

            if out:
                self.output.append(out)
            if err:
                self.output.append("❗ 오류 로그:\n" + err)

            if proc.returncode != 0:
                self.output.append("❌ 데이터셋 다운로드 실패 (스크립트 오류)")
                return

            # data.yaml 탐색 (dataset_dir 기준)
            yaml_path = self.find_yaml_in_dataset_dir()
            if yaml_path:
                self.output.append(f"\n✔ 데이터셋 다운로드 완료!\n✔ data.yaml: {yaml_path}")
                self.dataset_ready.emit(yaml_path)
            else:
                self.output.append("\n⚠ data.yaml을 찾지 못했습니다. 폴더 구조를 확인하세요.")

        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            if self.overlay:
                self.overlay.hide_overlay()

    def find_yaml_in_dataset_dir(self):
        """dataset_dir 아래에서 data.yaml을 찾아서 가장 최근 파일을 반환."""
        candidates = []
        for root, dirs, files in os.walk(self.dataset_dir):
            if "data.yaml" in files:
                full = os.path.join(root, "data.yaml")
                mtime = os.path.getmtime(full)
                candidates.append((mtime, full))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
