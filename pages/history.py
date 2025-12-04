import os
import json
import shutil
import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QMessageBox,
    QAbstractItemView, QListWidget, QListWidgetItem, QSplitter, QComboBox
)


def format_seconds(sec: float | int | None) -> str:
    if sec is None:
        return "-"
    try:
        sec = float(sec)
    except Exception:
        return "-"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


class HistoryPage(QWidget):

    def __init__(self, settings: dict):
        super().__init__()
        self.paths = settings

        self.all_entries: list[dict] = []
        self.filtered_entries: list[dict] = []
        self.page_size = 10
        self.current_page = 0

        # ===========================
        # 메인 Splitter (좌/우)
        # ===========================
        main_split = QSplitter(self)
        main_split.setOrientation(Qt.Horizontal)

        # ===========================
        # LEFT — Train History Panel
        # ===========================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(30, 30, 30, 30)
        left_layout.setSpacing(10)

        title = QLabel("📜 Train History")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        left_layout.addWidget(title)

        # Dataset + 검색 필터
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Dataset:"))

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["전체", "fire", "human"])
        self.dataset_combo.currentTextChanged.connect(self.apply_filter)
        filter_row.addWidget(self.dataset_combo)

        filter_row.addWidget(QLabel("검색:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("timestamp / model")
        self.search_edit.textChanged.connect(self.apply_filter)
        filter_row.addWidget(self.search_edit)

        left_layout.addLayout(filter_row)

        # ====================================
        # Train History Table
        # ====================================
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Dataset", "Base Model",
            "Epochs", "Patience", "mAP50", "Train Time"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self.on_row_selected)

        left_layout.addWidget(self.table, stretch=2)

        # ====================================
        # 페이징
        # ====================================
        paging = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 이전")
        self.btn_next = QPushButton("다음 ▶")
        self.page_label = QLabel("0 / 0 페이지")

        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)

        paging.addWidget(self.btn_prev)
        paging.addWidget(self.btn_next)
        paging.addStretch()
        paging.addWidget(self.page_label)
        left_layout.addLayout(paging)

        # ====================================
        # 상세 metadata
        # ====================================
        detail_title = QLabel("📄 상세 메타데이터")
        detail_title.setStyleSheet("font-weight:bold;")
        left_layout.addWidget(detail_title)

        self.detail_edit = QTextEdit()
        self.detail_edit.setReadOnly(True)
        self.detail_edit.setStyleSheet("font-family:Consolas; font-size:12px;")
        left_layout.addWidget(self.detail_edit, stretch=1)

        # 삭제 버튼
        del_box = QHBoxLayout()
        del_box.addStretch()
        self.btn_delete = QPushButton("🗑 선택한 모델 삭제")
        self.btn_delete.clicked.connect(self.delete_selected)
        del_box.addWidget(self.btn_delete)
        left_layout.addLayout(del_box)

        # ===========================
        # RIGHT — Predict Viewer
        # ===========================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(10)

        right_title = QLabel("🔍 Predict Results")
        right_title.setStyleSheet("font-size:16px; font-weight:bold;")
        right_layout.addWidget(right_title)

        # 파일 리스트
        self.media_list = QListWidget()
        self.media_list.itemClicked.connect(self.on_media_selected)
        right_layout.addWidget(self.media_list, stretch=1)

        # 미리보기
        self.preview_label = QLabel("미리보기 없음")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(300)
        self.preview_label.setStyleSheet(
            "border:1px solid #444; background:#111; color:#999;"
        )
        right_layout.addWidget(self.preview_label, stretch=1)

        # conf 표시
        self.conf_label = QLabel("conf: -")
        self.conf_label.setStyleSheet("color:#bbb;")
        right_layout.addWidget(self.conf_label)

        # ===========================
        # Splitter 배치
        # ===========================
        main_split.addWidget(left_widget)
        main_split.addWidget(right_widget)
        main_split.setSizes([900, 400])

        m = QVBoxLayout(self)
        m.addWidget(main_split)

        # 초기 로딩
        self.reload_history()

    # =========================================================
    # settings 변경 반영
    # =========================================================
    def update_paths(self, settings: dict):
        self.paths = settings
        self.reload_history()

    # =========================================================
    # Train History 로딩
    # =========================================================
    def reload_history(self):
        history_dir = self.paths.get("history_dir", "history")

        self.all_entries.clear()

        if not os.path.isdir(history_dir):
            self.filtered_entries = []
            self.refresh_table()
            return

        for name in os.listdir(history_dir):
            subdir = os.path.join(history_dir, name)
            if not os.path.isdir(subdir):
                continue

            meta_path = os.path.join(subdir, "metadata.json")
            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # 🔥 map50 cast to float
                if "map50" in meta and meta["map50"] is not None:
                    try:
                        meta["map50"] = float(meta["map50"])
                    except:
                        meta["map50"] = None

            except Exception:
                continue

            entry = {
                "timestamp": meta.get("timestamp", name),
                "dataset": meta.get("dataset", "unknown"),
                "base_model": meta.get("base_model", "-"),
                "epochs": meta.get("epochs", "-"),
                "patience": meta.get("patience", "-"),
                "map50": meta.get("map50"),
                "train_time_sec": meta.get("train_time_sec"),
                "meta_path": meta_path,
                "meta": meta,
                "history_dir": subdir,
                "models_file": meta.get("models_file"),
                "run_dir": meta.get("run_dir"),
                "predict_root": os.path.join(subdir, "predict_log")
            }

            self.all_entries.append(entry)

        # 최신순 정렬
        self.all_entries.sort(key=lambda x: x["timestamp"], reverse=True)

        self.apply_filter()

    # =========================================================
    # 검색 + dataset 필터
    # =========================================================
    def apply_filter(self):
        q = self.search_edit.text().strip().lower()
        ds = self.dataset_combo.currentText()

        temp = []
        for e in self.all_entries:
            ok_dataset = (ds == "전체") or (e["dataset"] == ds)
            if not ok_dataset:
                continue

            if q:
                text = f"{e['timestamp']} {e['base_model']}".lower()
                if q not in text:
                    continue

            temp.append(e)

        self.filtered_entries = temp
        self.current_page = 0
        self.refresh_table()

    # =========================================================
    # 테이블 리프레시
    # =========================================================
    def refresh_table(self):
        total = len(self.filtered_entries)
        if total == 0:
            self.table.setRowCount(0)
            self.page_label.setText("0 / 0 페이지")
            return

        page_cnt = (total + self.page_size - 1) // self.page_size
        self.current_page = min(self.current_page, page_cnt - 1)

        start = self.current_page * self.page_size
        end = min(start + self.page_size, total)
        page_entries = self.filtered_entries[start:end]

        self.table.setRowCount(len(page_entries))

        for row, e in enumerate(page_entries):
            cols = [
                e["timestamp"],
                e["dataset"],
                e["base_model"],
                str(e["epochs"]),
                str(e["patience"]),
                f"{e['map50']:.4f}" if e["map50"] is not None else "-",
                format_seconds(e["train_time_sec"])
            ]
            for c, v in enumerate(cols):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, c, item)

        self.page_label.setText(f"{self.current_page + 1} / {page_cnt} 페이지")

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh_table()

    def next_page(self):
        total = len(self.filtered_entries)
        if total == 0:
            return
        page_cnt = (total + self.page_size - 1) // self.page_size
        if self.current_page < page_cnt - 1:
            self.current_page += 1
            self.refresh_table()

    # =========================================================
    # 상세 메타데이터 + Predict 결과 로딩
    # =========================================================
    def _get_selected_entry(self):
        row = self.table.currentRow()
        if row < 0:
            return None

        idx = self.current_page * self.page_size + row
        if 0 <= idx < len(self.filtered_entries):
            return self.filtered_entries[idx]
        return None

    def on_row_selected(self, row: int, _col: int):
        entry = self._get_selected_entry()
        if not entry:
            self.detail_edit.clear()
            return

        # metadata 표시
        meta = entry.get("meta", {})
        try:
            pretty = json.dumps(meta, indent=4, ensure_ascii=False)
        except:
            pretty = str(meta)
        self.detail_edit.setPlainText(pretty)

        # 오른쪽 Predict Viewer 갱신
        self.load_predict_media(entry)

    # =========================================================
    # Predict 결과 로드
    # =========================================================
    def load_predict_media(self, entry: dict):
        self.media_list.clear()
        self.preview_label.setText("미리보기 없음")
        self.conf_label.setText("conf: -")

        predict_root = entry.get("predict_root")
        if not predict_root or not os.path.isdir(predict_root):
            return

        # predict_log 내부 predict_* 폴더 탐색
        folders = sorted([
            f for f in os.listdir(predict_root)
            if f.startswith("predict_")
        ], reverse=True)

        for fold in folders:
            full_dir = os.path.join(predict_root, fold)

            # 1) predict_metadata.json 읽기
            meta_file = os.path.join(full_dir, "predict_metadata.json")
            conf_str = "-"

            if os.path.isfile(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        pmeta = json.load(f)
                        conf = pmeta.get("conf")
                        if conf is not None:
                            conf_str = f"{conf * 100:.1f}%"
                except:
                    pass

            # 2) media 폴더 탐색
            media_dir = os.path.join(full_dir, "media")
            if not os.path.isdir(media_dir):
                continue

            for name in sorted(os.listdir(media_dir)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv")):
                    item = QListWidgetItem(f"{fold} | {name}")
                    item.setData(Qt.UserRole, os.path.join(media_dir, name))
                    item.setData(Qt.UserRole + 1, conf_str)
                    self.media_list.addItem(item)

    # =========================================================
    # 미디어 미리보기
    # =========================================================
    def on_media_selected(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        conf_info = item.data(Qt.UserRole + 1)

        self.conf_label.setText(f"conf: {conf_info}")

        if not path or not os.path.exists(path):
            return

        self.show_media_preview(path)

    def show_media_preview(self, path: str):
        # 이미지
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            pixmap = QPixmap(path)
            if pixmap.isNull():
                return
            self.preview_label.setPixmap(
                pixmap.scaled(
                    self.preview_label.width(),
                    self.preview_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
            return

        # 영상 → 첫 프레임
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # =========================================================
    # 삭제 기능
    # =========================================================
    def delete_selected(self):
        entry = self._get_selected_entry()
        if not entry:
            QMessageBox.information(self, "알림", "삭제할 기록을 먼저 선택하세요.")
            return

        ts = entry["timestamp"]
        models_file = entry.get("models_file")
        run_dir = entry.get("run_dir")
        history_dir = entry.get("history_dir")

        msg = QMessageBox.question(
            self,
            "삭제 확인",
            f"선택한 학습 기록({ts}) 및 관련 파일 전체를 삭제하시겠습니까?\n(되돌릴 수 없습니다.)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if msg != QMessageBox.Yes:
            return

        # 모델 파일 삭제
        try:
            if models_file and os.path.isfile(models_file):
                os.remove(models_file)
        except:
            pass

        # runs 삭제
        try:
            if run_dir and os.path.isdir(run_dir):
                shutil.rmtree(run_dir)
        except:
            pass

        # history 삭제
        try:
            if history_dir and os.path.isdir(history_dir):
                shutil.rmtree(history_dir)
        except:
            pass

        QMessageBox.information(self, "완료", "기록 삭제 완료")
        self.reload_history()
        self.media_list.clear()
        self.preview_label.setText("미리보기 없음")
        self.conf_label.setText("conf: -")
