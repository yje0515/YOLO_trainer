import os
import json
import matplotlib
matplotlib.use("Agg")  # PySide6 충돌 방지
import matplotlib.pyplot as plt

from io import BytesIO
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget,
    QScrollArea, QFrame, QHBoxLayout
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
# Model Row (카드 + 통합 그래프 1개)
# ============================================================
class ModelRow(QWidget):
    def __init__(self, meta: dict, run_dir: str):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ---------------------------
        # 카드형 정보
        # ---------------------------
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: #F7F9FC;
                border: 1px solid #DDE3EC;
                border-radius: 8px;
            }
        """)
        card_layout = QVBoxLayout(card)

        title = QLabel(f"📌 {meta.get('timestamp')} — {meta.get('base_model')}")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        card_layout.addWidget(title)

        sub = QLabel(
            f"Dataset: {meta.get('dataset', 'unknown')}   |   "
            f"mAP50: {meta.get('map50')}   |   "
            f"Epochs: {meta.get('epochs')}   |   "
            f"Train Time: {meta.get('train_time_sec'):.2f}s"
        )
        sub.setStyleSheet("color:#555;")
        card_layout.addWidget(sub)

        layout.addWidget(card)

        # ---------------------------
        # 그래프 1개 (Loss + Accuracy)
        # ---------------------------
        graph_pix = self.create_graph_combined(run_dir)
        layout.addWidget(self.graph_label(graph_pix))

    # ============================================================
    # 통합 그래프 (Loss + Accuracy)
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

            # Accuracy 후보 선택: mAP50 계열
            acc_col = None
            for col in df.columns:
                if "mAP50" in col:
                    acc_col = col
                    break

            # Loss 후보 선택: train box_loss
            loss_col = None
            for col in df.columns:
                if "train/box_loss" in col or "train/box_loss" in col.replace("_", "/"):
                    loss_col = col
                    break

            if not acc_col and not loss_col:
                return QPixmap()

            # ---------- Matplotlib 그래프 ----------
            fig, ax1 = plt.subplots(figsize=(4.0, 2.3))

            # LOSS (왼쪽 축)
            if loss_col:
                ax1.plot(
                    df[loss_col],
                    color="#FF4444",
                    marker="o",
                    linewidth=2,
                    label="Loss"
                )
                ax1.set_ylabel("Loss", color="#FF4444")
                ax1.tick_params(axis='y', labelcolor="#FF4444")

            # ACCURACY (오른쪽 축)
            if acc_col:
                ax2 = ax1.twinx()
                ax2.plot(
                    df[acc_col],
                    color="#0074FF",
                    marker="o",
                    linewidth=2,
                    label="Accuracy"
                )
                ax2.set_ylabel("Accuracy", color="#0074FF")
                ax2.tick_params(axis='y', labelcolor="#0074FF")

            ax1.set_xlabel("Epoch")
            ax1.set_title("Loss / Accuracy Curve")
            ax1.grid(True, alpha=0.25)

            pix = fig_to_pixmap(fig)
            plt.close(fig)
            return pix

        except Exception:
            return QPixmap()

    # ============================================================
    # QLabel 포맷팅
    # ============================================================
    def graph_label(self, pix):
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(
            pix.scaled(
                460, 280,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        return lbl


# ============================================================
# Model Comparison Page
# ============================================================
class ModelComparisonPage(QWidget):
    def __init__(self, settings: dict):
        super().__init__()
        self.paths = settings

        main = QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)

        title = QLabel("📚 Model Comparison")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main.addWidget(title)

        # ---------------------------
        # Tabs (fire / human / unknown)
        # ---------------------------
        self.tabs_widget = QTabWidget()
        main.addWidget(self.tabs_widget)

        self.tabs = {
            "fire": self._create_tab("🔥 Fire Dataset"),
            "human": self._create_tab("🧍 Human Dataset"),
            "unknown": self._create_tab("❓ Unknown Dataset"),
        }

        # 초기 로딩
        self.reload_models()

    # ============================================================
    # 탭 만들기
    # ============================================================
    def _create_tab(self, label):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(25)
        container_layout.addStretch()

        scroll.setWidget(container)
        self.tabs_widget.addTab(scroll, label)

        return container_layout

    # ============================================================
    # 모델 재로딩
    # ============================================================
    def reload_models(self):
        history_dir = self.paths.get("history_dir", "C:/yolo_data/history")
        if not os.path.isdir(history_dir):
            return

        # 기존 row 제거
        for tab in self.tabs.values():
            while tab.count() > 0:
                item = tab.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            tab.addStretch()

        # 최신순으로 로딩
        for folder in sorted(os.listdir(history_dir), reverse=True):
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

            row = ModelRow(meta, run_dir)

            if dataset == "fire":
                self.tabs["fire"].insertWidget(0, row)
            elif dataset == "human":
                self.tabs["human"].insertWidget(0, row)
            else:
                self.tabs["unknown"].insertWidget(0, row)
