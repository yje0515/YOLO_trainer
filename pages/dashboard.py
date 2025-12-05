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

# =========================================================
# 공통 포맷 헬퍼
# =========================================================
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


# =========================================================
# Loss + Accuracy 통합 그래프 (Dashboard용)
# =========================================================
class LossAccChart(FigureCanvas):
    def __init__(self, run_dir: str, parent=None):
        self.fig = Figure(figsize=(5.2, 3.2))   # 요청대로 "조금 더 크게"
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.draw_chart(run_dir)

    def draw_chart(self, run_dir: str):
        csv_path = os.path.join(run_dir, "results.csv")
        if not os.path.isfile(csv_path):
            self.ax.text(0.5, 0.5, "결과 CSV가 없습니다", ha="center", va="center")
            self.draw()
            return

        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
        except:
            self.ax.text(0.5, 0.5, "CSV 로드 실패", ha="center", va="center")
            self.draw()
            return

        # Accuracy 찾기 (mAP50)
        acc_col = None
        for c in df.columns:
            if "mAP50" in c:
                acc_col = c
                break

        # Loss 찾기
        loss_col = None
        for c in df.columns:
            if "train/box_loss" in c or "box_loss" in c:
                loss_col = c
                break

        self.ax.clear()

        # LOSS
        if loss_col:
            self.ax.plot(
                df[loss_col],
                color="#FF4444",
                linewidth=2,
                marker="o",
                label="Loss"
            )
            self.ax.set_ylabel("Loss", color="#FF4444")
            self.ax.tick_params(axis='y', labelcolor="#FF4444")

        # Accuracy
        if acc_col:
            ax2 = self.ax.twinx()
            ax2.plot(
                df[acc_col],
                color="#0066FF",
                linewidth=2,
                marker="o",
                label="Accuracy"
            )
            ax2.set_ylabel("Accuracy", color="#0066FF")
            ax2.tick_params(axis='y', labelcolor="#0066FF")

        self.ax.set_xlabel("Epoch")
        self.ax.set_title("Loss / Accuracy Curve")
        self.ax.grid(True, alpha=0.25)

        self.fig.tight_layout()
        self.draw()


# =========================================================
# mAP50 Line Chart (dataset별 색상 적용)
# =========================================================
class Map50LineChart(FigureCanvas):
    def __init__(self, entries: List[Dict], parent=None):
        self.fig = Figure(figsize=(6, 3.2))
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.28, left=0.08, right=0.98, top=0.88)

        self.draw_chart(entries)

    def draw_chart(self, entries: List[Dict]):
        # mAP50 있는 데이터만 가져오기
        valid = [e for e in entries if e.get("map50") is not None]
        if not valid:
            self.ax.text(0.5, 0.5, "mAP50 데이터 없음", ha="center", va="center")
            self.draw()
            return

        # 최신순 10개 → 오래된 것이 왼쪽
        recent = valid[:10]
        recent = list(reversed(recent))

        timestamps = [e["timestamp"] for e in recent]
        values = [e["map50"] for e in recent]
        datasets = [e.get("dataset", "unknown") for e in recent]

        # 색 매핑
        color_map = {
            "fire": "#FF4444",
            "human": "#0066FF",
            "unknown": "#888888",
            "etc": "#00AA44"
        }

        self.ax.clear()

        # dataset마다 선 1개씩 그리도록
        for ds in ["fire", "human", "etc", "unknown"]:
            xs = []
            ys = []
            for i, d in enumerate(datasets):
                if d == ds:
                    xs.append(i)
                    ys.append(values[i])
            if xs:
                self.ax.plot(
                    xs, ys,
                    linestyle="--",
                    marker="o",
                    linewidth=2,
                    color=color_map.get(ds, "#888888"),
                    label=ds
                )

        self.ax.set_xticks(range(len(timestamps)))
        self.ax.set_xticklabels(timestamps, rotation=45, ha="right", fontsize=8)
        self.ax.set_ylabel("mAP50")
        self.ax.set_title("최근 모델 mAP50 변화 (Line Chart)")
        self.ax.legend()

        self.fig.tight_layout()
        self.draw()


# =========================================================
# Dashboard Page (전체)
# =========================================================
class DashboardPage(QWidget):
    """
    YOLO Trainer Dashboard Page
    """

    def __init__(self, settings: dict):
        super().__init__()

        self.paths = settings
        self.entries: List[Dict] = []

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

        title = QLabel("🏠 Dashboard")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.main_layout.addWidget(title)

        self.reload_data()
        self.build_ui()

    # =========================================================
    def update_paths(self, settings: dict):
        self.paths = settings
        self.reload_data()
        self.rebuild_ui()

    # =========================================================
    def reload_data(self):
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
            except:
                continue

            # mAP50 정규화
            raw_map = meta.get("map50")
            try:
                map50_val = float(raw_map) if raw_map is not None else None
            except:
                map50_val = None

            entry = {
                "timestamp": meta.get("timestamp", name),
                "dataset": meta.get("dataset", "unknown"),
                "base_model": meta.get("base_model", "-"),
                "epochs": meta.get("epochs", "-"),
                "patience": meta.get("patience", "-"),
                "map50": map50_val,
                "train_time_sec": meta.get("train_time_sec"),
                "models_file": meta.get("models_file"),
                "run_dir": meta.get("run_dir"),
                "meta": meta,
            }
            self.entries.append(entry)

        self.entries.sort(key=lambda e: e["timestamp"], reverse=True)

    # =========================================================
    def rebuild_ui(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        title = QLabel("🏠 Dashboard")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.main_layout.addWidget(title)

        self.build_ui()

    # =========================================================
    def build_ui(self):
        if not self.entries:
            info = QLabel("📭 아직 학습이 없습니다.")
            info.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(info)
            return

        # ------------------------
        # 1) 최고 성능 모델
        # ------------------------
        best = self._get_best_model()
        if best:
            sec = QLabel("🏆 최고 성능 모델")
            sec.setStyleSheet("font-size: 16px; font-weight: bold;")
            self.main_layout.addWidget(sec)

            # 카드
            self.main_layout.addWidget(self._create_model_card(best))

            # 🔥 그래프 추가 (요청 사항)
            run_dir = best.get("run_dir")
            if run_dir:
                chart = LossAccChart(run_dir)
                chart.setMinimumHeight(300)
                self.main_layout.addWidget(chart)

        # ------------------------
        # 2) 최근 모델 3개
        # ------------------------
        recent = self._get_recent_models(3)
        if recent:
            sec = QLabel("🕒 최근 학습 모델 (최신 3개)")
            sec.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
            self.main_layout.addWidget(sec)

            row = QHBoxLayout()
            for e in recent:
                row.addWidget(self._create_model_small_card(e))
            row.addStretch()
            self.main_layout.addLayout(row)

        # ------------------------
        # 3) Dataset별 Top3
        # ------------------------
        for ds_name, icon in [("fire", "🔥"), ("human", "👤"), ("unknown", "❓"), ("etc", "🌱")]:
            ds_entries = self._get_top_by_dataset(ds_name, 3)
            if ds_entries:
                sec = QLabel(f"{icon} Dataset: {ds_name} (Top 3)")
                sec.setStyleSheet("font-size: 15px; font-weight: bold; margin-top: 16px;")
                self.main_layout.addWidget(sec)

                row = QHBoxLayout()
                for e in ds_entries:
                    row.addWidget(self._create_model_small_card(e))
                row.addStretch()
                self.main_layout.addLayout(row)

        # ------------------------
        # 4) 전체 Line Chart
        # ------------------------
        if any(e.get("map50") is not None for e in self.entries):
            sec = QLabel("📈 mAP50 비교 (Line Chart)")
            sec.setStyleSheet("font-size: 15px; font-weight: bold; margin-top: 20px;")
            self.main_layout.addWidget(sec)

            chart = Map50LineChart(self.entries)
            chart.setMinimumHeight(280)
            self.main_layout.addWidget(chart)

        self.main_layout.addStretch()

    # =========================================================
    # 데이터 헬퍼
    # =========================================================
    def _get_best_model(self) -> Optional[Dict]:
        vals = [e for e in self.entries if e.get("map50") is not None]
        if not vals:
            return None
        return max(vals, key=lambda e: e["map50"])

    def _get_recent_models(self, n: int) -> List[Dict]:
        return self.entries[:n]

    def _get_top_by_dataset(self, ds: str, n: int) -> List[Dict]:
        filt = [e for e in self.entries if (e.get("dataset") or "unknown") == ds and e.get("map50") is not None]
        filt.sort(key=lambda e: e["map50"], reverse=True)
        return filt[:n]

    # =========================================================
    # 카드 UI
    # =========================================================
    def _create_model_card(self, e: Dict) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #FFF;
                border: 1px solid #DDD;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel(f"Dataset: {e['dataset']}   |   Base: {e['base_model']}")
        title.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(title)

        map50 = e.get("map50")
        lbl_map = QLabel(f"mAP50: {map50:.4f}" if map50 is not None else "mAP50: -")
        layout.addWidget(lbl_map)

        lbl_info = QLabel(f"Epochs: {e['epochs']}  |  Time: {format_seconds(e['train_time_sec'])}")
        lbl_info.setStyleSheet("color:#555; font-size:12px;")
        layout.addWidget(lbl_info)

        lbl_ts = QLabel(f"Timestamp: {e['timestamp']}")
        lbl_ts.setStyleSheet("color:#777; font-size:12px;")
        layout.addWidget(lbl_ts)

        return frame

    def _create_model_small_card(self, e: Dict) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background:#FFF;
                border:1px solid #E0E0E0;
                border-radius:6px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel(f"{e['dataset']} / {e['base_model']}")
        title.setStyleSheet("font-size:12px; font-weight:bold;")
        layout.addWidget(title)

        map50 = e.get("map50")
        lbl_map = QLabel(f"mAP50: {map50:.4f}" if map50 is not None else "mAP50: -")
        layout.addWidget(lbl_map)

        lbl_info = QLabel(
            f"Ep:{e['epochs']} | Time:{format_seconds(e['train_time_sec'])}"
        )
        lbl_info.setStyleSheet("font-size:11px; color:#555;")
        layout.addWidget(lbl_info)

        lbl_ts = QLabel(f"TS: {e['timestamp']}")
        lbl_ts.setStyleSheet("font-size:11px; color:#777;")
        layout.addWidget(lbl_ts)

        return frame
