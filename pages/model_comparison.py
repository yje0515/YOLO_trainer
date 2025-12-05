import os
import json
import matplotlib
matplotlib.use("Agg")  # PySide6 충돌 방지
import matplotlib.pyplot as plt

from io import BytesIO
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget,
    QScrollArea, QFrame, QGridLayout
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


# ============================================================
# Helper — Matplotlib → QPixmap 변환
# ============================================================
def fig_to_pixmap(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    pix = QPixmap()
    pix.loadFromData(buf.getvalue())
    buf.close()
    return pix


# ============================================================
# Dataset별 mAP50 최근 10개 그래프
# ============================================================
class DatasetMapChart(QWidget):
    def __init__(self, entries):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)

        fig = self.build_figure(entries)
        pix = fig_to_pixmap(fig)
        plt.close(fig)

        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(
            pix.scaled(620, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        layout.addWidget(lbl)

    def build_figure(self, entries):
        fig, ax = plt.subplots(figsize=(6.2, 2.6))

        if not entries:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            return fig

        # 최신순 정렬 → 오래된 순으로 재정렬
        entries = sorted(entries, key=lambda x: x["timestamp"], reverse=True)
        entries = list(reversed(entries[:10]))

        timestamps = [e["timestamp"] for e in entries]
        values = [e["map50"] for e in entries]

        ax.plot(
            range(len(values)), values,
            linestyle="--",
            marker="o",
            linewidth=2,
            color="#0074FF"
        )

        ax.set_xticks(range(len(timestamps)))
        ax.set_xticklabels(timestamps, rotation=45, ha="right", fontsize=8)
        ax.set_title("최근 10개 모델 성능 (mAP50)", fontsize=11)
        ax.set_ylabel("mAP50")
        ax.grid(True, alpha=0.25)

        return fig


# ============================================================
# Model Card + Graph (Small version)
# ============================================================
class ModelCard(QWidget):
    def __init__(self, meta: dict, run_dir: str):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # ---------------------------
        # 카드 영역
        # ---------------------------
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: #F7F9FC;
                border: 1px solid #DDE3EC;
                border-radius: 6px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(3)
        card_layout.setContentsMargins(8, 8, 8, 8)

        # 제목
        title = QLabel(f"{meta.get('timestamp')} — {meta.get('base_model')}")
        title.setStyleSheet("font-size: 12px; font-weight: bold;")
        card_layout.addWidget(title)

        # Sub info
        sub = QLabel(
            f"DS: {meta.get('dataset','?')} | "
            f"mAP50: {meta.get('map50')} | "
            f"Ep: {meta.get('epochs')}"
        )
        sub.setStyleSheet("color:#555; font-size:11px;")
        card_layout.addWidget(sub)

        layout.addWidget(card)

        # ---------------------------
        # 그래프 추가
        # ---------------------------
        graph = self.create_graph_combined(run_dir)
        layout.addWidget(self.graph_label(graph))

    # ============================================================
    # Loss / Accuracy 통합 그래프
    # ============================================================
    def create_graph_combined(self, run_dir):
        if not run_dir:
            return QPixmap()

        csv_path = os.path.join(run_dir, "results.csv")
        if not os.path.isfile(csv_path):
            return QPixmap()

        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]

            acc_col = next((c for c in df.columns if "mAP50" in c), None)
            loss_col = next((c for c in df.columns if "box_loss" in c), None)

            fig, ax1 = plt.subplots(figsize=(3.0, 1.4))

            if loss_col:
                ax1.plot(
                    df[loss_col], color="#FF4444",
                    marker="o", markersize=3, linewidth=1.3
                )
                ax1.set_ylabel("Loss", color="#FF4444", fontsize=8)

            if acc_col:
                ax2 = ax1.twinx()
                ax2.plot(
                    df[acc_col], color="#0074FF",
                    marker="o", markersize=3, linewidth=1.3
                )
                ax2.set_ylabel("Acc", color="#0074FF", fontsize=8)

            ax1.set_xlabel("Epoch", fontsize=8)
            ax1.set_title("Loss / Accuracy", fontsize=9)
            ax1.grid(True, alpha=0.25)

            pix = fig_to_pixmap(fig)
            plt.close(fig)
            return pix

        except Exception:
            return QPixmap()

    def graph_label(self, pix):
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(
            pix.scaled(
                300, 160,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        return lbl


# ============================================================
# Model Comparison Page — 3-per-row grid + dataset tabs
# ============================================================
class ModelComparisonPage(QWidget):
    def __init__(self, settings: dict):
        super().__init__()
        self.paths = settings

        main = QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(10)

        title = QLabel("📈 Model Graph")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main.addWidget(title)

        self.tabs_widget = QTabWidget()
        main.addWidget(self.tabs_widget)

        # 🔥 etc 탭 추가됨
        self.tabs = {
            "fire": self._create_tab("🔥 Fire Dataset"),
            "human": self._create_tab("🧍 Human Dataset"),
            "etc": self._create_tab("📦 Etc Dataset"),
            "unknown": self._create_tab("❓ Unknown Dataset"),
        }

        self.reload_models()

    # ============================================================
    # Create scrollable tab with grid
    # ============================================================
    def _create_tab(self, label):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(10, 10, 10, 10)

        grid = QGridLayout()
        grid.setSpacing(15)

        container_layout.addLayout(grid)
        container_layout.addStretch()

        scroll.setWidget(container)
        self.tabs_widget.addTab(scroll, label)

        return {"layout": container_layout, "grid": grid}

    # ============================================================
    # Reload model cards & graphs per dataset
    # ============================================================
    def reload_models(self):
        history_dir = self.paths.get("history_dir", "C:/yolo_data/history")
        if not os.path.isdir(history_dir):
            return

        # 탭 초기화
        for tab in self.tabs.values():
            layout = tab["layout"]
            grid = tab["grid"]

            while grid.count():
                item = grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # 기존 그래프 제거
            if layout.count() > 1:
                old = layout.itemAt(0).widget()
                if old:
                    old.deleteLater()

        folders = sorted(os.listdir(history_dir), reverse=True)

        # 🔥 여기에도 etc 추가
        dataset_entries = {"fire": [], "human": [], "etc": [], "unknown": []}

        # 카드 배치
        for folder in folders:
            sub = os.path.join(history_dir, folder)
            meta_path = os.path.join(sub, "metadata.json")

            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except:
                continue

            dataset = meta.get("dataset", "unknown")

            # etc 처리
            if dataset not in dataset_entries:
                dataset = "etc"

            run_dir = meta.get("run_dir")

            # 그래프용 데이터 저장
            if meta.get("map50") is not None:
                dataset_entries[dataset].append(meta)

            card = ModelCard(meta, run_dir)

            tab = self.tabs.get(dataset, self.tabs["unknown"])
            grid = tab["grid"]

            count = grid.count()
            row = count // 3
            col = count % 3

            grid.addWidget(card, row, col)

        # ---------------------------
        # 각 탭 상단에 그래프 삽입
        # ---------------------------
        for ds, tab in self.tabs.items():
            entries = dataset_entries.get(ds, [])
            chart = DatasetMapChart(entries)
            tab["layout"].insertWidget(0, chart)
