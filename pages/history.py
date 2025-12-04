# pages/history.py
import os
import json
import shutil

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QMessageBox,
    QAbstractItemView
)


def format_seconds(sec: float | int | None) -> str:
    """초 단위를 'HH:MM:SS' 형식 문자열로 변환."""
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
    """
    학습 기록 페이지
    - 상단: 검색 + 표(타임스탬프, 베이스모델, Epoch, Patience, mAP50, 소요시간)
    - 하단: 상세 메타데이터 보기 + 선택한 모델 삭제 버튼
    """

    def __init__(self, settings: dict):
        super().__init__()
        self.paths = settings

        self.all_entries: list[dict] = []      # 전체 히스토리
        self.filtered_entries: list[dict] = [] # 검색 적용된 히스토리
        self.page_size = 10
        self.current_page = 0

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(10)

        title = QLabel("📜 Train History")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        main_layout.addWidget(title)

        # ===== 검색 영역 =====
        search_layout = QHBoxLayout()
        search_label = QLabel("검색 (timestamp / base_model 포함):")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("예: yolo11n, 251204, floating, ...")
        self.search_edit.textChanged.connect(self.on_search_changed)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        main_layout.addLayout(search_layout)

        # ===== 표 영역 =====
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Base Model", "Epochs", "Patience", "mAP50", "Train Time"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self.on_row_selected)
        main_layout.addWidget(self.table, stretch=2)

        # ===== 페이징 영역 =====
        paging_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 이전")
        self.btn_next = QPushButton("다음 ▶")
        self.page_label = QLabel("0 / 0 페이지")

        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)

        paging_layout.addWidget(self.btn_prev)
        paging_layout.addWidget(self.btn_next)
        paging_layout.addStretch()
        paging_layout.addWidget(self.page_label)
        main_layout.addLayout(paging_layout)

        # ===== 상세 정보 + 삭제 영역 =====
        detail_title = QLabel("📄 상세 메타데이터")
        detail_title.setStyleSheet("font-weight:bold;")
        main_layout.addWidget(detail_title)

        self.detail_edit = QTextEdit()
        self.detail_edit.setReadOnly(True)
        self.detail_edit.setStyleSheet("font-family:Consolas; font-size:12px;")
        main_layout.addWidget(self.detail_edit, stretch=1)

        delete_layout = QHBoxLayout()
        delete_layout.addStretch()
        self.btn_delete = QPushButton("🗑 선택한 모델 삭제")
        self.btn_delete.clicked.connect(self.delete_selected)
        delete_layout.addWidget(self.btn_delete)
        main_layout.addLayout(delete_layout)

        main_layout.addStretch()

        # 초기 데이터 로딩
        self.reload_history()

    # =================== 공통 메서드 ===================

    def update_paths(self, settings: dict):
        """settings.json 변경 시 경로 갱신 + 히스토리 다시 로딩"""
        self.paths = settings
        self.reload_history()

    # =================== 히스토리 로딩 ===================

    def reload_history(self):
        """history_dir 아래의 metadata.json들을 모두 읽어와 테이블 갱신"""
        history_dir = self.paths.get("history_dir", "history")
        self.all_entries.clear()

        if not os.path.isdir(history_dir):
            self.filtered_entries = []
            self.refresh_table()
            return

        # history_dir 아래의 하위 폴더들 탐색
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
            except Exception:
                continue

            timestamp = meta.get("timestamp", name)
            base_model = meta.get("base_model", "-")
            epochs = meta.get("epochs", "-")
            patience = meta.get("patience", "-")

            # 🔹 여기서 중요한 포인트:
            #    - mAP50 → meta["map50"]
            #    - 학습 시간 → meta["train_time_sec"]
            map50 = meta.get("map50", None)
            train_time_sec = meta.get("train_time_sec", None)

            entry = {
                "timestamp": timestamp,
                "base_model": base_model,
                "epochs": epochs,
                "patience": patience,
                "map50": map50,
                "train_time_sec": train_time_sec,
                "meta_path": meta_path,
                "meta": meta,
                "history_dir": subdir,
                "models_file": meta.get("models_file", None),
                "run_dir": meta.get("run_dir", None),
            }
            self.all_entries.append(entry)

        # timestamp 기준 내림차순 정렬 (최신 학습이 위로)
        self.all_entries.sort(key=lambda x: x["timestamp"], reverse=True)

        # 검색 필터 초기화
        self.apply_filter()

    def apply_filter(self):
        """검색어를 반영해 filtered_entries 재구성"""
        q = self.search_edit.text().strip().lower()
        if not q:
            self.filtered_entries = list(self.all_entries)
        else:
            tmp = []
            for e in self.all_entries:
                text = f"{e['timestamp']} {e['base_model']}".lower()
                if q in text:
                    tmp.append(e)
            self.filtered_entries = tmp

        # 페이지 초기화 후 테이블 다시 그림
        self.current_page = 0
        self.refresh_table()

    # =================== 테이블 / 페이징 ===================

    def refresh_table(self):
        """현재 filtered_entries와 current_page 기준으로 표 갱신"""
        total = len(self.filtered_entries)
        if total == 0:
            self.table.setRowCount(0)
            self.page_label.setText("0 / 0 페이지")
            return

        page_count = (total + self.page_size - 1) // self.page_size
        if self.current_page >= page_count:
            self.current_page = page_count - 1

        start = self.current_page * self.page_size
        end = min(start + self.page_size, total)
        page_entries = self.filtered_entries[start:end]

        self.table.setRowCount(len(page_entries))

        for row, e in enumerate(page_entries):
            # Timestamp
            item_ts = QTableWidgetItem(e["timestamp"])
            item_ts.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, item_ts)

            # Base Model
            item_model = QTableWidgetItem(e["base_model"])
            item_model.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, item_model)

            # Epochs
            item_epoch = QTableWidgetItem(str(e["epochs"]))
            item_epoch.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, item_epoch)

            # Patience
            item_pat = QTableWidgetItem(str(e["patience"]))
            item_pat.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 3, item_pat)

            # mAP50
            map50_val = e.get("map50", None)
            if map50_val is None:
                map50_str = "-"
            else:
                try:
                    map50_str = f"{float(map50_val):.4f}"
                except Exception:
                    map50_str = str(map50_val)
            item_map = QTableWidgetItem(map50_str)
            item_map.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 4, item_map)

            # Train Time
            tt_str = format_seconds(e.get("train_time_sec", None))
            item_tt = QTableWidgetItem(tt_str)
            item_tt.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 5, item_tt)

        self.page_label.setText(f"{self.current_page + 1} / {page_count} 페이지")

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh_table()

    def next_page(self):
        total = len(self.filtered_entries)
        if total == 0:
            return
        page_count = (total + self.page_size - 1) // self.page_size
        if self.current_page < page_count - 1:
            self.current_page += 1
            self.refresh_table()

    # =================== 이벤트 핸들러 ===================

    def on_search_changed(self, _text: str):
        self.apply_filter()

    def _get_selected_entry(self):
        """현재 테이블에서 선택된 행에 대응하는 entry를 반환"""
        row = self.table.currentRow()
        if row < 0:
            return None

        # 현재 페이지 기준으로 실제 index 계산
        idx = self.current_page * self.page_size + row
        if 0 <= idx < len(self.filtered_entries):
            return self.filtered_entries[idx]
        return None

    def on_row_selected(self, row: int, _column: int):
        """행 선택 시 상세 메타데이터 표시"""
        entry = self._get_selected_entry()
        if not entry:
            self.detail_edit.clear()
            return

        meta = entry.get("meta", {})
        try:
            pretty = json.dumps(meta, indent=4, ensure_ascii=False)
        except Exception:
            pretty = str(meta)

        self.detail_edit.setPlainText(pretty)

    def delete_selected(self):
        """선택된 기록 및 관련 파일들 삭제"""
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
            f"선택한 학습 기록({ts})과 연결된 모델/로그 폴더를 모두 삭제할까요?\n"
            f"이 작업은 되돌릴 수 없습니다.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if msg != QMessageBox.Yes:
            return

        # 모델 파일 삭제
        if models_file and os.path.isfile(models_file):
            try:
                os.remove(models_file)
            except Exception:
                pass

        # runs/train_xxxx 폴더 삭제
        if run_dir and os.path.isdir(run_dir):
            try:
                shutil.rmtree(run_dir)
            except Exception:
                pass

        # history/timestamp 폴더 삭제
        if history_dir and os.path.isdir(history_dir):
            try:
                shutil.rmtree(history_dir)
            except Exception:
                pass

        QMessageBox.information(self, "완료", "선택한 학습 기록과 관련 파일이 삭제되었습니다.")
        self.reload_history()
