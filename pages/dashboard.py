import os
import json
import csv

from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def format_seconds(sec: Optional[float]) -> str:
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


class DashboardPage(QWidget):
    """
    YOLO Trainer 대시보드 페이지

    - history_dir 아래 metadata.json들을 읽어서 모델 성능 요약
    - 섹션 구성:
        1) 최고 성능 모델 (mAP50 기준 Top 1)
        2) 최근 학습 모델 3개 (timestamp 기준)
        3) Dataset별 Top3 (fire / human / unknown)
        4) (하단) 전체 mAP50 비교 그래프
    """

    def __init__(self, settings: dict):
        super().__init__()

        self.paths = settings
        self.entries: List[Dict] = []

        # ------------------------
        # 메인: 스크롤 가능 레이아웃
        # ------------------------
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(18)

        scroll.setWidget(container)

        # ------------------------
        # 타이틀
        # ------------------------
        title = QLabel("🏠 Dashboard")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.main_layout.addWidget(title)

        # 데이터 로드 + UI 빌드
        self.reload_data()
        self.build_ui()

    # =========================================================
    # 경로 설정 변경 대응
    # =========================================================
    def update_paths(self, settings: dict):
        self.paths = settings
        self.reload_data()
        self.rebuild_ui()

    # =========================================================
    # History 메타데이터 로딩
    # =========================================================
    def reload_data(self):
        """history_dir 아래 metadata.json들을 읽어 self.entries 구성"""
        history_dir = self.paths.get("history_dir", "history")

        self.entries = []
        if not os.path.isdir(history_dir):
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
            except Exception:
                continue

            # mAP50 숫자로 정규화
            raw_map = meta.get("map50", None)
            map50_val: Optional[float] = None
            if raw_map is not None:
                try:
                    map50_val = float(raw_map)
                except Exception:
                    map50_val = None

            entry = {
                "timestamp": meta.get("timestamp", name),
                "dataset": meta.get("dataset", "unknown"),
                "base_model": meta.get("base_model", "-"),
                "epochs": meta.get("epochs", "-"),
                "patience": meta.get("patience", "-"),
                "map50": map50_val,
                "train_time_sec": meta.get("train_time_sec", None),
                "models_file": meta.get("models_file"),
                "run_dir": meta.get("run_dir"),
                "meta": meta,
            }

            self.entries.append(entry)

        # timestamp 기준 최신순 정렬 (YYMMDD_HHMM이므로 문자열 정렬로도 동작)
        self.entries.sort(key=lambda e: e["timestamp"], reverse=True)

    # =========================================================
    # UI 전체 다시 그리기
    # =========================================================
    def rebuild_ui(self):
        # main_layout 내 기존 위젯 제거
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        # 다시 타이틀 + 섹션들 구성
        title = QLabel("🏠 Dashboard")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.main_layout.addWidget(title)

        self.build_ui()

    # =========================================================
    # UI 구성
    # =========================================================
    def build_ui(self):
        if not self.entries:
            info = QLabel("📭 아직 학습 기록이 없습니다.\nTrain 탭에서 모델을 학습하면 이곳에 요약 정보가 표시됩니다.")
            info.setAlignment(Qt.AlignCenter)
            info.setStyleSheet("color:#666; font-size:14px; margin-top:40px;")
            self.main_layout.addWidget(info)
            return

        # ------------------------
        # 1) 최고 성능 모델 섹션
        # ------------------------
        best_entry = self._get_best_model()
        if best_entry:
            sec_title = QLabel("🏆 최고 성능 모델")
            sec_title.setStyleSheet("font-size: 16px; font-weight: bold;")
            self.main_layout.addWidget(sec_title)

            self.main_layout.addWidget(self._create_model_card(best_entry))

        # ------------------------
        # 2) 최근 학습 모델 3개 섹션
        # ------------------------
        recent_entries = self._get_recent_models(3)
        if recent_entries:
            sec_title = QLabel("🕒 최근 학습 모델 (최신 3개)")
            sec_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
            self.main_layout.addWidget(sec_title)

            row = QHBoxLayout()
            row.setSpacing(10)
            for e in recent_entries:
                row.addWidget(self._create_model_small_card(e))
            row.addStretch()
            self.main_layout.addLayout(row)

        # ------------------------
        # 3) Dataset별 Top3 섹션
        # ------------------------
        for ds_name, icon in [("fire", "🔥"), ("human", "👤"), ("unknown", "❓")]:
            ds_entries = self._get_top_by_dataset(ds_name, 3)
            if not ds_entries:
                continue
            sec_title = QLabel(f"{icon} Dataset: {ds_name} (Top 3)")
            sec_title.setStyleSheet("font-size: 15px; font-weight: bold; margin-top: 16px;")
            self.main_layout.addWidget(sec_title)

            row = QHBoxLayout()
            row.setSpacing(10)
            for e in ds_entries:
                row.addWidget(self._create_model_small_card(e))
            row.addStretch()
            self.main_layout.addLayout(row)

        # ------------------------
        # 4) 전체 mAP50 비교 그래프
        # ------------------------
        if any(e.get("map50") is not None for e in self.entries):
            sec_title = QLabel("📈 mAP50 비교 (최신 10개)")
            sec_title.setStyleSheet("font-size: 15px; font-weight: bold; margin-top: 20px;")
            self.main_layout.addWidget(sec_title)

            chart = Map50BarChart(self.entries)
            chart.setMinimumHeight(260)
            self.main_layout.addWidget(chart)

        self.main_layout.addStretch()

    # =========================================================
    # 데이터 가공 헬퍼들
    # =========================================================
    def _get_best_model(self) -> Optional[Dict]:
        """mAP50 기준 최고 성능 모델 1개"""
        candidates = [e for e in self.entries if e.get("map50") is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["map50"])

    def _get_recent_models(self, n: int) -> List[Dict]:
        """timestamp 기준 최신 N개"""
        return self.entries[:n]

    def _get_top_by_dataset(self, dataset_name: str, n: int) -> List[Dict]:
        """특정 dataset에 대한 mAP50 기준 Top N"""
        filtered = [
            e for e in self.entries
            if (e.get("dataset") or "unknown") == dataset_name
            and e.get("map50") is not None
        ]
        filtered.sort(key=lambda e: e["map50"], reverse=True)
        return filtered[:n]

    # =========================================================
    # 카드 생성 UI 헬퍼들
    # =========================================================
    def _create_model_card(self, entry: Dict) -> QWidget:
        """
        큰 카드 (최고 성능 모델용)
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #DDDDDD;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # 첫 줄: Dataset + 모델명
        title = QLabel(f"Dataset: {entry.get('dataset', 'unknown')}   |   Base: {entry.get('base_model', '-')}")
        title.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(title)

        # 두 번째 줄: mAP50
        map50 = entry.get("map50")
        if map50 is not None:
            map_text = f"mAP50: {map50:.4f}"
        else:
            map_text = "mAP50: -"
        lbl_map = QLabel(map_text)
        lbl_map.setStyleSheet("font-size: 13px;")
        layout.addWidget(lbl_map)

        # 세 번째 줄: Epochs / Time
        epochs = entry.get("epochs", "-")
        time_str = format_seconds(entry.get("train_time_sec"))
        lbl_info = QLabel(f"Epochs: {epochs}    |    Train Time: {time_str}")
        lbl_info.setStyleSheet("font-size: 12px; color:#555;")
        layout.addWidget(lbl_info)

        # 네 번째 줄: Timestamp
        ts = entry.get("timestamp", "-")
        lbl_ts = QLabel(f"Timestamp: {ts}")
        lbl_ts.setStyleSheet("font-size: 12px; color:#777;")
        layout.addWidget(lbl_ts)

        return frame

    def _create_model_small_card(self, entry: Dict) -> QWidget:
        """
        작은 카드 (최근 모델 / Dataset별 Top3 용)
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
            }
        """)
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # 상단: Dataset + BaseModel
        title = QLabel(f"{entry.get('dataset', 'unknown')} / {entry.get('base_model', '-')}")
        title.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(title)

        # mAP50
        map50 = entry.get("map50")
        if map50 is not None:
            map_text = f"mAP50: {map50:.4f}"
        else:
            map_text = "mAP50: -"
        lbl_map = QLabel(map_text)
        lbl_map.setStyleSheet("font-size: 12px;")
        layout.addWidget(lbl_map)

        # Epochs + Time
        epochs = entry.get("epochs", "-")
        time_str = format_seconds(entry.get("train_time_sec"))
        lbl_info = QLabel(f"Ep: {epochs}  |  Time: {time_str}")
        lbl_info.setStyleSheet("font-size: 11px; color:#555;")
        layout.addWidget(lbl_info)

        # Timestamp
        ts = entry.get("timestamp", "-")
        lbl_ts = QLabel(f"TS: {ts}")
        lbl_ts.setStyleSheet("font-size: 11px; color:#777;")
        layout.addWidget(lbl_ts)

        return frame


# =============================================================
# mAP50 막대 그래프 위젯
# =============================================================
class Map50BarChart(FigureCanvas):
    """
    최신 10개 모델의 mAP50을 Bar Chart로 표시
    """

    def __init__(self, entries: List[Dict], parent=None):
        self.fig = Figure(figsize=(5, 3))
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.25, left=0.08, right=0.98, top=0.9)

        self.draw_chart(entries)

    def draw_chart(self, entries: List[Dict]):
        # mAP50 있는 것만 추출
        valid = [e for e in entries if e.get("map50") is not None]
        if not valid:
            self.ax.clear()
            self.ax.text(
                0.5, 0.5,
                "mAP50 데이터가 없습니다.",
                ha="center", va="center",
                fontsize=11
            )
            self.draw()
            return

        # 최신 10개만 사용
        recent = valid[:10]
        # 최신순 → 그래프는 오래된 것 왼쪽, 최신 오른쪽 정렬
        recent = list(reversed(recent))

        labels = [e["timestamp"] for e in recent]
        values = [e["map50"] for e in recent]

        self.ax.clear()
        self.ax.bar(range(len(values)), values)
        self.ax.set_ylim(0, max(values) * 1.1 if values else 1.0)
        self.ax.set_xticks(range(len(labels)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        self.ax.set_ylabel("mAP50")
        self.ax.set_title("최근 모델 mAP50")

        self.fig.tight_layout()
        self.draw()
