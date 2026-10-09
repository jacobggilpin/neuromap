"""
neuromap - Dedicated Pre-Processing & Artifact Reconstruction Workstation
Includes:
- ICTraceThumbnailWidget: Mini-waveform thumbnail preview for independent components
- AnalysisPanelWidget: FastICA component explorer, on-demand decomposition, ASR burst cleaning
"""
import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QTabWidget, QScrollArea, QCheckBox, QSlider
)

try:
    from .engine import EEGDataLoader, IndependentComponentInfo
except ImportError:
    from engine import EEGDataLoader, IndependentComponentInfo

class ICTraceThumbnailWidget(QWidget):
    """Clean mini-waveform thumbnail preview for an Independent Component."""
    def __init__(self, trace: np.ndarray, color: str = "#00FFA3", parent=None):
        super().__init__(parent)
        self.trace = trace
        self.color = QColor(color)
        self.setFixedHeight(34)
        self.setMinimumWidth(160)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        mid_y = h / 2.0

        # Background
        painter.fillRect(0, 0, w, h, QColor("#0A0C11"))
        painter.setPen(QPen(QColor("#1A202C"), 1))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        # Waveform polyline
        n_pts = len(self.trace)
        if n_pts > 1:
            poly = QPolygonF()
            step_x = w / float(n_pts - 1)
            y_scale = (h * 0.42) / 3.0  # normalize ~3 SD
            for i in range(n_pts):
                px = i * step_x
                py = mid_y - (self.trace[i] * y_scale)
                poly.append(QPointF(px, max(2.0, min(h - 2.0, py))))

            painter.setPen(QPen(self.color, 1.2))
            painter.drawPolyline(poly)


class AnalysisPanelWidget(QWidget):
    """
    Dedicated Pre-Processing & Artifact Reconstruction Workstation (Top-Left):
    - Tab 1: FastICA Spatial Decomposition (component explorer, badges, thumbnails, rejection checkboxes).
    - Tab 2: Artifact Subspace Reconstruction (ASR calibration, thresholding, subspace cleaning).
    - Real-time live subtraction feeding directly to 2D/3D topomaps and cascading waveforms.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._init_ui()
        self.data_loader.data_reconstructed.connect(self._on_data_reconstructed)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header Bar
        header = QFrame()
        header.setFixedHeight(34)
        header.setStyleSheet("""
            QFrame {
                background: #0B0C10;
                border-bottom: 1px solid #1A1F2C;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 2, 10, 2)
        h_layout.setSpacing(10)

        title_lbl = QLabel("<span style='color: #00E5FF;'>●</span> PRE-PROCESSING WORKSTATION")
        title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(title_lbl)

        h_layout.addStretch()

        self.reconstruct_status_lbl = QLabel("FastICA: Standby (Active Stream Filter)")
        self.reconstruct_status_lbl.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: bold;")
        h_layout.addWidget(self.reconstruct_status_lbl)

        layout.addWidget(header)

        # Tabs for Pre-Processing Modes
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: #0E0F14;
            }
            QTabBar::tab {
                background: #12151D;
                color: #8C9BAE;
                padding: 5px 12px;
                font-size: 10px;
                font-weight: bold;
                border: 1px solid #1A1F2C;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1B2433;
                color: #00E5FF;
                border-color: #00E5FF;
            }
        """)

        # Tab 1: FastICA Component Explorer
        self.tab_ica = QWidget()
        self._init_ica_tab()
        self.tab_widget.addTab(self.tab_ica, "FastICA Decomposition")

        # Tab 2: Artifact Subspace Reconstruction (ASR)
        self.tab_asr = QWidget()
        self._init_asr_tab()
        self.tab_widget.addTab(self.tab_asr, "Artifact Subspace (ASR)")

        layout.addWidget(self.tab_widget, stretch=1)

    def _init_ica_tab(self):
        tab_layout = QVBoxLayout(self.tab_ica)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(6)

        # Actions Toolbar
        actions_bar = QHBoxLayout()
        actions_bar.setSpacing(8)

        btn_auto_reject = QPushButton("⚡ Auto-Reject Artifacts")
        btn_auto_reject.setToolTip("Automatically flag and reject ocular blinks, saccades, and cardiac ECG pulses")
        btn_auto_reject.setStyleSheet("""
            QPushButton {
                background: #14221D;
                color: #00FFA3;
                border: 1px solid #00FFA3;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1C332A; }
        """)
        btn_auto_reject.clicked.connect(self._on_auto_reject)
        actions_bar.addWidget(btn_auto_reject)

        btn_restore = QPushButton("Restore All")
        btn_restore.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1F2737; color: #FFFFFF; }
        """)
        btn_restore.clicked.connect(self._on_restore_all)
        actions_bar.addWidget(btn_restore)

        btn_redecompose = QPushButton("⚡ Re-Decompose")
        btn_redecompose.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #00E5FF;
                border: 1px solid #1E2B3D;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1E2D42; }
        """)
        btn_redecompose.clicked.connect(self._on_redecompose_ica)
        actions_bar.addWidget(btn_redecompose)

        actions_bar.addStretch()
        tab_layout.addLayout(actions_bar)

        # Scroll Area for Component Cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #0E0F14; }")

        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 4, 0)
        self.cards_layout.setSpacing(4)

        scroll.setWidget(self.cards_container)
        tab_layout.addWidget(scroll, stretch=1)

        self.rebuild_component_cards()

    def _init_asr_tab(self):
        tab_layout = QVBoxLayout(self.tab_asr)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(12)

        asr_header = QLabel("Artifact Subspace Reconstruction (ASR)")
        asr_header.setStyleSheet("color: #00E5FF; font-size: 12px; font-weight: bold;")
        tab_layout.addWidget(asr_header)

        descr = QLabel(
            "ASR cleans transient, high-variance burst artifacts (e.g. gross body motion, muscle tension, "
            "electrode pops) by projecting corrupted temporal subspaces back onto a clean baseline geometric calibration."
        )
        descr.setWordWrap(True)
        descr.setStyleSheet("color: #8C9BAE; font-size: 10px; line-height: 15px;")
        tab_layout.addWidget(descr)

        self.chk_enable_asr = QCheckBox("Enable ASR Subspace Filtering")
        self.chk_enable_asr.setChecked(False)
        self.chk_enable_asr.toggled.connect(self._on_toggle_asr)
        tab_layout.addWidget(self.chk_enable_asr)

        # Cutoff threshold slider
        thresh_box = QVBoxLayout()
        self.lbl_cutoff = QLabel("Burst Cutoff Threshold (k): 5.0 SD")
        self.lbl_cutoff.setStyleSheet("color: #DDE2EB; font-size: 10px; font-weight: bold;")
        self.slider_cutoff = QSlider(Qt.Horizontal)
        self.slider_cutoff.setRange(3, 12)
        self.slider_cutoff.setValue(5)
        self.slider_cutoff.valueChanged.connect(self._on_cutoff_changed)
        thresh_box.addWidget(self.lbl_cutoff)
        thresh_box.addWidget(self.slider_cutoff)
        tab_layout.addLayout(thresh_box)

        # Action button
        btn_run_asr = QPushButton("⚡ Run ASR Calibration & Cleaning")
        btn_run_asr.setStyleSheet("""
            QPushButton {
                background: #14221D;
                color: #00FFA3;
                border: 1px solid #00FFA3;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1C332A; }
        """)
        btn_run_asr.clicked.connect(self._on_run_asr)
        tab_layout.addWidget(btn_run_asr)

        # Diagnostic Stats
        diag_frame = QFrame()
        diag_frame.setStyleSheet("background: #0A0C11; border: 1px solid #1A202C; border-radius: 4px; padding: 8px;")
        diag_layout = QVBoxLayout(diag_frame)
        diag_layout.setSpacing(4)

        self.lbl_asr_cal = QLabel("Calibration Covariance: Median Cov (64x64) • Rank 64")
        self.lbl_asr_cal.setStyleSheet("color: #8C9BAE; font-size: 10px;")
        self.lbl_asr_cleaned = QLabel("Artifact Subspaces Cleaned: Standby")
        self.lbl_asr_cleaned.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: bold;")

        diag_layout.addWidget(self.lbl_asr_cal)
        diag_layout.addWidget(self.lbl_asr_cleaned)
        tab_layout.addWidget(diag_frame)

        tab_layout.addStretch()

    def rebuild_component_cards(self):
        """Builds interactive cards for all decomposed FastICA components."""
        # Clear existing
        while self.cards_layout.count() > 0:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if len(self.data_loader.ic_components) == 0:
            standby_box = QFrame()
            standby_box.setStyleSheet("""
                QFrame {
                    background: #10141D;
                    border: 1px dashed #202D42;
                    border-radius: 6px;
                }
            """)
            s_layout = QVBoxLayout(standby_box)
            s_layout.setContentsMargins(16, 24, 16, 24)
            s_layout.setAlignment(Qt.AlignCenter)
            s_layout.setSpacing(10)

            icon_lbl = QLabel("⚡")
            icon_lbl.setStyleSheet("font-size: 26px; color: #00E5FF;")
            icon_lbl.setAlignment(Qt.AlignCenter)
            s_layout.addWidget(icon_lbl)

            title = QLabel("CLOSED-LOOP SPATIAL FILTERING ENGINE")
            title.setStyleSheet("color: #FFFFFF; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;")
            title.setAlignment(Qt.AlignCenter)
            s_layout.addWidget(title)

            desc = QLabel(
                "Streaming engine initialized on standby. FastICA spatial decomposition operates\n"
                "on-demand from the active closed-loop stream without blocking application startup."
            )
            desc.setStyleSheet("color: #8C9BAE; font-size: 10px; line-height: 1.5;")
            desc.setAlignment(Qt.AlignCenter)
            s_layout.addWidget(desc)

            btn_decompose_now = QPushButton("⚡ Decompose Active Stream (16 ICs)")
            btn_decompose_now.setCursor(Qt.PointingHandCursor)
            btn_decompose_now.setStyleSheet("""
                QPushButton {
                    background: #14221D;
                    color: #00FFA3;
                    border: 1px solid #00FFA3;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover { background: #1C332A; }
            """)
            btn_decompose_now.clicked.connect(self._on_redecompose_ica)
            s_layout.addWidget(btn_decompose_now)

            self.cards_layout.addWidget(standby_box)
            return

        for comp in self.data_loader.ic_components:
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background: #12151D;
                    border: 1px solid #1A212E;
                    border-radius: 3px;
                }
            """)
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(6, 4, 6, 4)
            card_layout.setSpacing(8)

            # Name and Checkbox
            chk = QCheckBox(comp.name)
            chk.setChecked(comp.is_rejected)
            chk.setToolTip(f"Check to subtract {comp.name} from reconstructed EEG")
            chk.toggled.connect(lambda checked, idx=comp.idx: self.data_loader.toggle_ic_component(idx, checked))
            card_layout.addWidget(chk)

            # Classification Badge
            tag_lbl = QLabel(comp.tag)
            tag_lbl.setStyleSheet(f"""
                color: {comp.tag_color};
                background: #0A0D14;
                border: 1px solid {comp.tag_color};
                border-radius: 2px;
                font-size: 8px;
                font-weight: bold;
                padding: 1px 4px;
            """)
            card_layout.addWidget(tag_lbl)

            # Metrics
            stats_lbl = QLabel(f"Kurt: {comp.kurtosis:.1f} | Var: {comp.variance_pct:.1f}%")
            stats_lbl.setStyleSheet("color: #7A889B; font-family: 'Consolas', monospace; font-size: 9px;")
            card_layout.addWidget(stats_lbl)

            # Mini Waveform Thumbnail Preview
            thumb = ICTraceThumbnailWidget(comp.trace_snippet, color=comp.tag_color, parent=card)
            card_layout.addWidget(thumb, stretch=1)

            self.cards_layout.addWidget(card)

    def _on_auto_reject(self):
        # Automatically flag and reject ocular blinks, saccades, and cardiac ECG pulses
        if len(self.data_loader.ic_components) == 0:
            self.data_loader.decompose_fastica()
        for comp in self.data_loader.ic_components:
            if "OCULAR" in comp.tag or "CARDIAC" in comp.tag or "EMG" in comp.tag:
                self.data_loader.rejected_ic_indices.add(comp.idx)
            else:
                self.data_loader.rejected_ic_indices.discard(comp.idx)
        self.data_loader.ica_enabled = True
        self.data_loader._reconstruct_signal()
        self.rebuild_component_cards()

    def _on_restore_all(self):
        self.data_loader.rejected_ic_indices.clear()
        self.data_loader._reconstruct_signal()
        self.rebuild_component_cards()

    def _on_redecompose_ica(self):
        self.data_loader.decompose_fastica()
        self.rebuild_component_cards()

    def _on_toggle_asr(self, checked: bool):
        self.data_loader.asr_enabled = checked
        self.data_loader._reconstruct_signal()

    def _on_cutoff_changed(self, val: int):
        self.data_loader.asr_cutoff_sd = float(val)
        self.lbl_cutoff.setText(f"Burst Cutoff Threshold (k): {val:.1f} SD")

    def _on_run_asr(self):
        self.data_loader.asr_enabled = True
        self.chk_enable_asr.setChecked(True)
        self.data_loader._reconstruct_signal()
        self.lbl_asr_cleaned.setText(f"Artifact Subspaces Cleaned: {self.data_loader.asr_cleaned_windows_pct:.1f}% of windows")

    def _on_data_reconstructed(self):
        n_rej = len(self.data_loader.rejected_ic_indices)
        status_parts = []
        if self.data_loader.ica_enabled and n_rej > 0:
            status_parts.append(f"FastICA ({n_rej} ICs rejected)")
        if self.data_loader.asr_enabled:
            status_parts.append(f"ASR Active ({self.data_loader.asr_cutoff_sd:.1f} SD)")
        
        if status_parts:
            self.reconstruct_status_lbl.setText("Clean Signal: " + " + ".join(status_parts))
            self.reconstruct_status_lbl.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: bold;")
        else:
            self.reconstruct_status_lbl.setText("Raw Signal (No Subtractions)")
            self.reconstruct_status_lbl.setStyleSheet("color: #7A889B; font-size: 10px;")


# ==============================================================================
# PHASE 4.02: 2D SCALP TOPOMAP WITH VOLTAGE COLORBAR & SMOOTH INTERPOLATION
# ==============================================================================

