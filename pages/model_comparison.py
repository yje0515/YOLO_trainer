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
        # 그래프 추가 (Combined Loss/Accuracy)
        # ---------------------------
        graph = self.create_graph_combined(run_dir)
        layout.addWidget(self.graph_label(graph))

    # ============================================================
    # Loss / Accuracy 통합 그래프 (소형 버전)
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

            # Accuracy 컬럼 찾기
            acc_col = next((c for c in df.columns if "mAP50" in c), None)
            # Loss 컬럼 찾기
            loss_col = next((c for c in df.columns if "box_loss" in c), None)

            fig, ax1 = plt.subplots(figsize=(3.0, 1.4))

            # LOSS (left axis)
            if loss_col:
                ax1.plot(
                    df[loss_col],
                    color="#FF4444",
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    label="Loss"
                )
                ax1.set_ylabel("Loss", color="#FF4444", fontsize=8)
                ax1.tick_params(axis='y', labelcolor="#FF4444", labelsize=7)

            # ACCURACY (right axis)
            if acc_col:
                ax2 = ax1.twinx()
                ax2.plot(
                    df[acc_col],
                    color="#0074FF",
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    label="Accuracy"
                )
                ax2.set_ylabel("Acc", color="#0074FF", fontsize=8)
                ax2.tick_params(axis='y', labelcolor="#0074FF", labelsize=7)

            ax1.set_xlabel("Epoch", fontsize=8)
            ax1.tick_params(axis='x', labelsize=7)
            ax1.set_title("Loss / Accuracy", fontsize=9)
            ax1.grid(True, alpha=0.25)

            pix = fig_to_pixmap(fig)
            plt.close(fig)
            return pix

        except Exception:
            return QPixmap()

    # ============================================================
    # QLabel wrapping for graph
    # ============================================================
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
# Model Comparison Page (Grid Layout 3 per row)
# ============================================================
class ModelComparisonPage(QWidget):
    def __init__(self, settings: dict):
        super().__init__()
        self.paths = settings

        main = QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(10)

        title = QLabel("📚 Model Comparison")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main.addWidget(title)

        self.tabs_widget = QTabWidget()
        main.addWidget(self.tabs_widget)

        # 탭 3개 생성
        self.tabs = {
            "fire": self._create_tab("🔥 Fire Dataset"),
            "human": self._create_tab("🧍 Human Dataset"),
            "unknown": self._create_tab("❓ Unknown Dataset"),
        }

        self.reload_models()

    # ============================================================
    # Create scrollable tab with GridLayout (3 columns)
    # ============================================================
    def _create_tab(self, label):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(15)
        grid.setContentsMargins(10, 10, 10, 10)

        scroll.setWidget(container)
        self.tabs_widget.addTab(scroll, label)
        return grid

    # ============================================================
    # Reload all model cards into grids
    # ============================================================
    def reload_models(self):
        history_dir = self.paths.get("history_dir", "C:/yolo_data/history")
        if not os.path.isdir(history_dir):
            return

        # Clear present widgets
        for grid in self.tabs.values():
            while grid.count():
                item = grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # 최신순 정렬
        folders = sorted(os.listdir(history_dir), reverse=True)

        # insert models 3 per row
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
            run_dir = meta.get("run_dir")
            card = ModelCard(meta, run_dir)

            # insert in grid
            grid = self.tabs.get(dataset, self.tabs["unknown"])
            row = grid.rowCount()
            col = grid.columnCount()

            # 계산식: 현재 widget 수 기반으로 위치 계산
            count = grid.count()
            row = count // 3
            col = count % 3

            grid.addWidget(card, row, col)
