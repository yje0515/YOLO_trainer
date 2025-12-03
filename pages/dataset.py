import os
import subprocess
import tempfile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QFileDialog, QTextEdit
)
from PySide6.QtCore import Qt, Signal


class DatasetPage(QWidget):
    dataset_ready = Signal(str)  # data.yaml 경로를 TrainPage로 전달

    def __init__(self, settings: dict):
        super().__init__()
        self.overlay = None
        self.update_paths(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("📁 Dataset (Roboflow 코드 실행 & data.yaml 선택)")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        desc = QLabel(
            "1) Roboflow에서 받은 Python 코드를 아래에 붙여넣고 '▶ 데이터셋 다운로드' 버튼을 눌러주세요.\n"
            "2) 다운로드 후 자동으로 해당 데이터셋이 선택됩니다. \n"
            "3) 준비된 data.yaml이 있다면 '📂 data.yaml 불러오기'로 선택할 수 있습니다.\n"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.code_edit = QPlainTextEdit()
        self.code_edit.setPlaceholderText(
            "예시)\n"
            "* pip은 제외됩니다.\n"
            "from roboflow import Roboflow\n"
            "rf = Roboflow(api_key=\"XXX\")\n"
            "project = rf.workspace(\"workspace\").project(\"project\")\n"
            "version = project.version(3)\n"
            "dataset = version.download(\"yolov8\") * YOLO 버전을 확인해 주세요."
        )
        layout.addWidget(self.code_edit)

        self.btn_run = QPushButton("▶ 데이터셋 다운로드")
        self.btn_run.clicked.connect(self.run_script)
        layout.addWidget(self.btn_run)

        self.btn_select_yaml = QPushButton("📂 data.yaml 불러오기")
        self.btn_select_yaml.clicked.connect(self.select_yaml_file)
        layout.addWidget(self.btn_select_yaml)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family:Consolas; font-size:11px;")
        layout.addWidget(self.output)

        layout.addStretch()

    def set_overlay(self, overlay):
        self.overlay = overlay

    def update_paths(self, settings: dict):
        self.settings = settings
        self.dataset_dir = settings.get("dataset_dir", "./datasets")
        self.temp_dir = settings.get("temp_dir", "./temp")
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

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

    def run_script(self):
        raw_code = self.code_edit.toPlainText().strip()
        if not raw_code:
            self.output.append("❌ Roboflow 코드를 확인해 주세요.")
            return

        if self.overlay:
            self.overlay.show_overlay("📥 데이터셋 다운로드 중...")

        tmp_path = None
        try:
            self.output.append("\n⏳ 스크립트 실행 중...\n")

            fd, tmp_path = tempfile.mkstemp(
                suffix=".py",
                dir=self.temp_dir,
                text=True
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(raw_code)

            python_exe = os.sys.executable

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

            yaml_path = self.find_latest_yaml()
            if yaml_path:
                self.output.append(f"\n✔ 데이터셋 다운로드 완료! 학습을 진행하세요.\n✔ data.yaml: {yaml_path}")
                self.dataset_ready.emit(yaml_path)
            else:
                self.output.append("\n⚠ data.yaml을 찾지 못했습니다.")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if self.overlay:
                self.overlay.hide_overlay()

    def find_latest_yaml(self):
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
