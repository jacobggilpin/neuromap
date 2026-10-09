"""
neuromap - Dedicated Pre-Processing & Advanced Analytical Workstation (Top-Left)
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Workstation Modules:
- Tab 1: FastICA Spatial Filter Explorer (fixed-point decomposition, tags, thumbnails, auto-reject)
- Tab 2: Artifact Subspace Reconstruction (ASR) (burst filter, cutoff slider, subspace variance meter)
- Tab 3: Power Spectral Density (PSD) & Band Power (Welch estimation, Delta/Theta/Alpha/Beta/Gamma indicators)
- Tab 4: Time-Frequency ERSP & Sensorimotor ERD (Morlet scalograms for C3, Cz, C4 synced to T1/T2 motor imagery)
- Tab 5: Functional Connectivity Graph Networks (64x64 PLV & Coherence matrices, threshold slider, hub analysis)
- Tab 6: Real-Time BCI Motor Imagery Decoder (CSP + online LDA classifier, real-time motor intent probability bars)
- Live LSL Hardware Stream Ingestion Dock (pylsl inlet connector, sample rate, buffer latency)
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# PyQt5 GUI Framework
try:
    from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
    from PyQt5.QtGui import QColor, QFont, QPen, QBrush, QPainter, QPolygonF
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
        QSlider, QCheckBox, QComboBox, QProgressBar, QFrame, QScrollArea,
        QTabWidget, QSizePolicy
    )
    import pyqtgraph as pg
except ImportError:
    pass

from config import (
    STANDARD_64_CHANNELS, FREQUENCY_BANDS, SENSORIMOTOR_CHANNELS,
    EVENT_COLOR_MAP, UNIFIED_LUT_256
)
from engine import EEGDataLoader, IndependentComponentInfo, LSLReceiverThread
import analytics


# ==============================================================================
# FASTICA MINI-WAVEFORM THUMBNAIL WIDGET
# ==============================================================================

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


# ==============================================================================
# ADVANCED ANALYTICAL WORKSTATION PANEL
# ==============================================================================

class AnalysisPanelWidget(QWidget):
    """
    Advanced Analytical & Pre-Processing Workstation:
    - Tab 1: FastICA Spatial Decomposition & Online Rejection
    - Tab 2: Artifact Subspace Reconstruction (ASR)
    - Tab 3: Power Spectral Density (PSD) & Canonical Band Power
    - Tab 4: Time-Frequency ERSP & Sensorimotor ERD (C3, Cz, C4)
    - Tab 5: Functional Connectivity Graph Networks (PLV & Coherence)
    - Tab 6: Real-Time BCI Motor Imagery Decoder (CSP + Online LDA)
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.lsl_thread: Optional[LSLReceiverThread] = None

        self._last_psd_update = 0.0
        self._last_conn_update = 0.0
        self._last_bci_update = 0.0

        self._init_ui()
        self.data_loader.data_reconstructed.connect(self._on_data_reconstructed)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header Bar
        header = QFrame()
        header.setFixedHeight(36)
        header.setStyleSheet("""
            QFrame {
                background: #0B0C10;
                border-bottom: 1px solid #1A1F2C;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 2, 10, 2)
        h_layout.setSpacing(10)

        title_lbl = QLabel("<span style='color: #00E5FF;'>●</span> ANALYTICAL WORKSTATION")
        title_lbl.setStyleSheet("color: #E2E8F0; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(title_lbl)

        # LSL Live Ingestion Mini Controls
        self.lbl_lsl_status = QLabel("● LSL: Idle")
        self.lbl_lsl_status.setStyleSheet("color: #8C9BAE; font-size: 10px; font-weight: bold;")
        h_layout.addWidget(self.lbl_lsl_status)

        self.btn_lsl_toggle = QPushButton("Connect LSL")
        self.btn_lsl_toggle.setFocusPolicy(Qt.NoFocus)
        self.btn_lsl_toggle.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #00FFA3;
                border: 1px solid #1E2B3D;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 9px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1C332A; border-color: #00FFA3; }
        """)
        self.btn_lsl_toggle.clicked.connect(self._toggle_lsl)
        h_layout.addWidget(self.btn_lsl_toggle)

        h_layout.addStretch()

        self.reconstruct_status_lbl = QLabel("FastICA: Standby (Active Stream Filter)")
        self.reconstruct_status_lbl.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: bold;")
        h_layout.addWidget(self.reconstruct_status_lbl)

        layout.addWidget(header)

        # Workstation Navigation Tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: #0E0F14;
            }
            QTabBar::tab {
                background: #12151D;
                color: #8C9BAE;
                padding: 5px 11px;
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
        self.tab_widget.addTab(self.tab_ica, "FastICA Spatial")

        # Tab 2: Artifact Subspace Reconstruction (ASR)
        self.tab_asr = QWidget()
        self._init_asr_tab()
        self.tab_widget.addTab(self.tab_asr, "ASR Bursts")

        # Tab 3: Power Spectral Density (PSD) & Band Power (Module 1)
        self.tab_psd = QWidget()
        self._init_psd_tab()
        self.tab_widget.addTab(self.tab_psd, "Spectral PSD")

        # Tab 4: Time-Frequency ERSP & Sensorimotor ERD (Module 2)
        self.tab_ersp = QWidget()
        self._init_ersp_tab()
        self.tab_widget.addTab(self.tab_ersp, "Time-Freq ERSP")

        # Tab 5: Functional Connectivity Graph (Module 3)
        self.tab_conn = QWidget()
        self._init_conn_tab()
        self.tab_widget.addTab(self.tab_conn, "Connectivity")

        # Tab 6: Real-Time BCI Motor Imagery Decoder (Module 4)
        self.tab_bci = QWidget()
        self._init_bci_tab()
        self.tab_widget.addTab(self.tab_bci, "BCI Decoder")

        layout.addWidget(self.tab_widget, stretch=1)

    # --------------------------------------------------------------------------
    # TAB 1: FASTICA COMPONENT EXPLORER
    # --------------------------------------------------------------------------
    def _init_ica_tab(self):
        tab_layout = QVBoxLayout(self.tab_ica)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(6)

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

    # --------------------------------------------------------------------------
    # TAB 2: ARTIFACT SUBSPACE RECONSTRUCTION (ASR)
    # --------------------------------------------------------------------------
    def _init_asr_tab(self):
        tab_layout = QVBoxLayout(self.tab_asr)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(10)

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

    # --------------------------------------------------------------------------
    # TAB 3: POWER SPECTRAL DENSITY (PSD) & CANONICAL BAND POWER (MODULE 1)
    # --------------------------------------------------------------------------
    def _init_psd_tab(self):
        tab_layout = QVBoxLayout(self.tab_psd)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(6)

        info_bar = QHBoxLayout()
        self.psd_channel_lbl = QLabel("<span style='color: #00FFA3;'>●</span> Active Channels: Global Average")
        self.psd_channel_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold;")
        info_bar.addWidget(self.psd_channel_lbl)
        info_bar.addStretch()
        tab_layout.addLayout(info_bar)

        # Plot Widget for PSD Log-Power Curve
        self.psd_plot = pg.PlotWidget()
        self.psd_plot.setBackground('#0A0C11')
        self.psd_plot.showGrid(x=True, y=True, alpha=0.15)
        self.psd_plot.setLabel('bottom', 'Frequency (Hz)', color='#8C9BAE', size='9pt')
        self.psd_plot.setLabel('left', 'Power (µV²/Hz)', color='#8C9BAE', size='9pt')
        self.psd_plot.setXRange(1.0, 45.0, padding=0.01)
        self.psd_curve = pg.PlotDataItem(pen=pg.mkPen('#00FFA3', width=2.0))
        self.psd_plot.addItem(self.psd_curve)

        # Shaded Canonical Band Backgrounds
        for band_name, (f0, f1, color) in FREQUENCY_BANDS.items():
            lr = pg.LinearRegionItem([f0, f1], movable=False, brush=pg.mkBrush(color + '18'))
            lr.setLinesPen(pg.mkPen(color + '40', width=1))
            self.psd_plot.addItem(lr)

        tab_layout.addWidget(self.psd_plot, stretch=1)

        # Canonical Band Power Indicators Container
        bars_frame = QFrame()
        bars_frame.setStyleSheet("background: #08090D; border: 1px solid #161B26; border-radius: 4px; padding: 6px;")
        bars_layout = QGridLayout(bars_frame)
        bars_layout.setContentsMargins(6, 4, 6, 4)
        bars_layout.setHorizontalSpacing(10)
        bars_layout.setVerticalSpacing(4)

        self.band_progress_bars: Dict[str, QProgressBar] = {}
        self.band_value_labels: Dict[str, QLabel] = {}

        for row, (band_name, (f0, f1, color)) in enumerate(FREQUENCY_BANDS.items()):
            lbl = QLabel(f"{band_name} ({f0:.0f}-{f1:.0f}Hz)")
            lbl.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")
            bars_layout.addWidget(lbl, row, 0)

            pbar = QProgressBar()
            pbar.setFixedHeight(12)
            pbar.setRange(0, 100)
            pbar.setValue(20)
            pbar.setTextVisible(False)
            pbar.setStyleSheet(f"""
                QProgressBar {{
                    background: #12151D;
                    border: 1px solid #1C2331;
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background: {color};
                    border-radius: 2px;
                }}
            """)
            bars_layout.addWidget(pbar, row, 1)
            self.band_progress_bars[band_name] = pbar

            val_lbl = QLabel("0.0%")
            val_lbl.setFixedWidth(50)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("color: #FFFFFF; font-size: 9px; font-family: 'Consolas', monospace;")
            bars_layout.addWidget(val_lbl, row, 2)
            self.band_value_labels[band_name] = val_lbl

        tab_layout.addWidget(bars_frame)

    # --------------------------------------------------------------------------
    # TAB 4: TIME-FREQUENCY ERSP & SENSORIMOTOR ERD (MODULE 2)
    # --------------------------------------------------------------------------
    def _init_ersp_tab(self):
        tab_layout = QVBoxLayout(self.tab_ersp)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(6)

        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(8)

        lbl = QLabel("Stimulus Event:")
        lbl.setStyleSheet("color: #8C9BAE; font-size: 10px; font-weight: bold;")
        ctrl_bar.addWidget(lbl)

        self.ersp_event_combo = QComboBox()
        self.ersp_event_combo.addItems([
            "T1: Left Fist (Contralateral C4 ERD)",
            "T2: Right Fist (Contralateral C3 ERD)",
            "T0: Rest Baseline"
        ])
        self.ersp_event_combo.currentIndexChanged.connect(self._on_ersp_event_changed)
        ctrl_bar.addWidget(self.ersp_event_combo)

        ctrl_bar.addStretch()

        self.ersp_erd_badge = QLabel("Sensorimotor Mu/Beta Desynchronization Active")
        self.ersp_erd_badge.setStyleSheet("color: #00FFA3; font-size: 9px; font-weight: bold;")
        ctrl_bar.addWidget(self.ersp_erd_badge)

        tab_layout.addLayout(ctrl_bar)

        # 3 Synchronized Heatmaps for C3, Cz, C4
        grid = QGridLayout()
        grid.setSpacing(6)

        self.ersp_plots: Dict[str, pg.PlotWidget] = {}
        self.ersp_images: Dict[str, pg.ImageItem] = {}
        self.ersp_playheads: Dict[str, pg.InfiniteLine] = {}

        for idx, ch in enumerate(SENSORIMOTOR_CHANNELS):
            pw = pg.PlotWidget()
            pw.setBackground('#0A0C11')
            pw.showGrid(x=True, y=True, alpha=0.15)
            pw.setLabel('left', f'{ch} (Hz)', color='#00FFA3', size='9pt')
            pw.setLabel('bottom', 'Time (s)', color='#8C9BAE', size='8pt')
            pw.setXRange(-1.0, 3.5, padding=0.01)
            pw.setYRange(6.0, 32.0, padding=0.01)

            img = pg.ImageItem()
            img.setLookupTable(UNIFIED_LUT_256)
            img.setLevels([-8.0, 8.0])
            pw.addItem(img)

            # Stimulus onset line (t=0.0)
            onset_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen('#00FFA3', width=1.5, style=Qt.DashLine))
            pw.addItem(onset_line)

            # Playhead tracker
            p_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen('#FFFFFF', width=1.8))
            pw.addItem(p_line)

            grid.addWidget(pw, idx, 0)
            self.ersp_plots[ch] = pw
            self.ersp_images[ch] = img
            self.ersp_playheads[ch] = p_line

        tab_layout.addLayout(grid, stretch=1)
        self._refresh_ersp_heatmaps()

    # --------------------------------------------------------------------------
    # TAB 5: FUNCTIONAL CONNECTIVITY GRAPH (MODULE 3)
    # --------------------------------------------------------------------------
    def _init_conn_tab(self):
        tab_layout = QVBoxLayout(self.tab_conn)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(6)

        top_ctrl = QHBoxLayout()
        top_ctrl.setSpacing(10)

        top_ctrl.addWidget(QLabel("Metric:"))
        self.combo_conn_metric = QComboBox()
        self.combo_conn_metric.addItems(["Phase-Locking Value (PLV)", "Spectral Coherence"])
        top_ctrl.addWidget(self.combo_conn_metric)

        top_ctrl.addWidget(QLabel("Band:"))
        self.combo_conn_band = QComboBox()
        self.combo_conn_band.addItems(list(FREQUENCY_BANDS.keys()))
        self.combo_conn_band.setCurrentText("Alpha")
        top_ctrl.addWidget(self.combo_conn_band)

        top_ctrl.addStretch()

        self.lbl_conn_thresh = QLabel("Threshold: 0.65")
        self.lbl_conn_thresh.setStyleSheet("color: #FFFFFF; font-size: 10px; font-weight: bold;")
        top_ctrl.addWidget(self.lbl_conn_thresh)

        self.slider_conn_thresh = QSlider(Qt.Horizontal)
        self.slider_conn_thresh.setRange(40, 95)
        self.slider_conn_thresh.setValue(65)
        self.slider_conn_thresh.setFixedWidth(120)
        self.slider_conn_thresh.valueChanged.connect(self._on_conn_thresh_changed)
        top_ctrl.addWidget(self.slider_conn_thresh)

        tab_layout.addLayout(top_ctrl)

        # 64x64 Adjacency Matrix Viewport
        body_split = QHBoxLayout()
        body_split.setSpacing(8)

        self.conn_plot = pg.PlotWidget()
        self.conn_plot.setBackground('#0A0C11')
        self.conn_plot.setAspectLocked(True)
        self.conn_plot.hideAxis('left')
        self.conn_plot.hideAxis('bottom')
        self.conn_img = pg.ImageItem()
        self.conn_img.setLookupTable(UNIFIED_LUT_256)
        self.conn_img.setLevels([0.0, 1.0])
        self.conn_plot.addItem(self.conn_img)
        body_split.addWidget(self.conn_plot, stretch=3)

        # Side Card: Hub Channels & Network Stats
        hubs_frame = QFrame()
        hubs_frame.setStyleSheet("background: #08090D; border: 1px solid #161B26; border-radius: 4px; padding: 6px;")
        hubs_layout = QVBoxLayout(hubs_frame)
        hubs_layout.setSpacing(4)

        h_title = QLabel("NETWORK HUBS")
        h_title.setStyleSheet("color: #00E5FF; font-size: 10px; font-weight: bold;")
        hubs_layout.addWidget(h_title)

        self.lbl_hub_channels = QLabel(
            "1. C3  (Sensorimotor)\n"
            "2. C4  (Sensorimotor)\n"
            "3. Fz  (Frontal Control)\n"
            "4. Pz  (Parietal Hub)\n"
            "5. Oz  (Occipital Alpha)"
        )
        self.lbl_hub_channels.setStyleSheet("color: #8C9BAE; font-size: 10px; line-height: 18px;")
        hubs_layout.addWidget(self.lbl_hub_channels)

        hubs_layout.addStretch()

        self.lbl_active_edges_count = QLabel("Active Edges: 24")
        self.lbl_active_edges_count.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: bold;")
        hubs_layout.addWidget(self.lbl_active_edges_count)

        body_split.addWidget(hubs_frame, stretch=2)
        tab_layout.addLayout(body_split, stretch=1)

    # --------------------------------------------------------------------------
    # TAB 6: REAL-TIME BCI MOTOR IMAGERY DECODER (MODULE 4)
    # --------------------------------------------------------------------------
    def _init_bci_tab(self):
        tab_layout = QVBoxLayout(self.tab_bci)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(10)

        header_box = QHBoxLayout()
        bci_title = QLabel("REAL-TIME BCI MOTOR IMAGERY DECODER")
        bci_title.setStyleSheet("color: #00FFA3; font-size: 12px; font-weight: bold; letter-spacing: 0.5px;")
        header_box.addWidget(bci_title)

        header_box.addStretch()

        btn_cal_bci = QPushButton("⚡ Calibrate BCI (CSP + LDA)")
        btn_cal_bci.setStyleSheet("""
            QPushButton {
                background: #14221D;
                color: #00FFA3;
                border: 1px solid #00FFA3;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1C332A; }
        """)
        btn_cal_bci.clicked.connect(self._on_calibrate_bci)
        header_box.addWidget(btn_cal_bci)
        tab_layout.addLayout(header_box)

        # Decoded Intent Announcement Banner
        self.bci_banner = QFrame()
        self.bci_banner.setFixedHeight(48)
        self.bci_banner.setStyleSheet("""
            QFrame {
                background: #0D1E18;
                border: 2px solid #00FFA3;
                border-radius: 6px;
            }
        """)
        banner_layout = QHBoxLayout(self.bci_banner)
        self.lbl_bci_intent = QLabel("⚡ DECODED INTENT: REST / IDLE (92% CONFIDENCE)")
        self.lbl_bci_intent.setAlignment(Qt.AlignCenter)
        self.lbl_bci_intent.setStyleSheet("color: #00FFA3; font-size: 14px; font-weight: 900; letter-spacing: 1px;")
        banner_layout.addWidget(self.lbl_bci_intent)
        tab_layout.addWidget(self.bci_banner)

        # Motor Intent Probabilities (Left Fist vs Rest vs Right Fist)
        probs_frame = QFrame()
        probs_frame.setStyleSheet("background: #0A0C11; border: 1px solid #1A202C; border-radius: 5px; padding: 10px;")
        probs_layout = QVBoxLayout(probs_frame)
        probs_layout.setSpacing(8)

        # 1. Left Fist Bar (T1)
        row_left = QHBoxLayout()
        lbl_l = QLabel("LEFT FIST (T1):")
        lbl_l.setFixedWidth(110)
        lbl_l.setStyleSheet("color: #00FFA3; font-size: 11px; font-weight: bold;")
        self.pbar_bci_left = QProgressBar()
        self.pbar_bci_left.setFixedHeight(18)
        self.pbar_bci_left.setRange(0, 100)
        self.pbar_bci_left.setValue(15)
        self.pbar_bci_left.setStyleSheet("""
            QProgressBar { background: #12151D; border: 1px solid #1E2838; border-radius: 4px; }
            QProgressBar::chunk { background: #00FFA3; border-radius: 3px; }
        """)
        self.lbl_p_left = QLabel("15.0%")
        self.lbl_p_left.setFixedWidth(50)
        self.lbl_p_left.setStyleSheet("color: #FFFFFF; font-weight: bold; font-family: monospace;")
        row_left.addWidget(lbl_l)
        row_left.addWidget(self.pbar_bci_left, stretch=1)
        row_left.addWidget(self.lbl_p_left)
        probs_layout.addLayout(row_left)

        # 2. Right Fist Bar (T2)
        row_right = QHBoxLayout()
        lbl_r = QLabel("RIGHT FIST (T2):")
        lbl_r.setFixedWidth(110)
        lbl_r.setStyleSheet("color: #BA68C8; font-size: 11px; font-weight: bold;")
        self.pbar_bci_right = QProgressBar()
        self.pbar_bci_right.setFixedHeight(18)
        self.pbar_bci_right.setRange(0, 100)
        self.pbar_bci_right.setValue(15)
        self.pbar_bci_right.setStyleSheet("""
            QProgressBar { background: #12151D; border: 1px solid #1E2838; border-radius: 4px; }
            QProgressBar::chunk { background: #BA68C8; border-radius: 3px; }
        """)
        self.lbl_p_right = QLabel("15.0%")
        self.lbl_p_right.setFixedWidth(50)
        self.lbl_p_right.setStyleSheet("color: #FFFFFF; font-weight: bold; font-family: monospace;")
        row_right.addWidget(lbl_r)
        row_right.addWidget(self.pbar_bci_right, stretch=1)
        row_right.addWidget(self.lbl_p_right)
        probs_layout.addLayout(row_right)

        # 3. Rest Bar (T0)
        row_rest = QHBoxLayout()
        lbl_0 = QLabel("REST / BASELINE:")
        lbl_0.setFixedWidth(110)
        lbl_0.setStyleSheet("color: #00E5FF; font-size: 11px; font-weight: bold;")
        self.pbar_bci_rest = QProgressBar()
        self.pbar_bci_rest.setFixedHeight(18)
        self.pbar_bci_rest.setRange(0, 100)
        self.pbar_bci_rest.setValue(70)
        self.pbar_bci_rest.setStyleSheet("""
            QProgressBar { background: #12151D; border: 1px solid #1E2838; border-radius: 4px; }
            QProgressBar::chunk { background: #00E5FF; border-radius: 3px; }
        """)
        self.lbl_p_rest = QLabel("70.0%")
        self.lbl_p_rest.setFixedWidth(50)
        self.lbl_p_rest.setStyleSheet("color: #FFFFFF; font-weight: bold; font-family: monospace;")
        row_rest.addWidget(lbl_0)
        row_rest.addWidget(self.pbar_bci_rest, stretch=1)
        row_rest.addWidget(self.lbl_p_rest)
        probs_layout.addLayout(row_rest)

        tab_layout.addWidget(probs_frame)

        # Neuro-Engineering Diagnostic Note
        bci_diag = QLabel(
            "<b>Architecture:</b> 8-30 Hz Mu/Beta Bandpass → 4-Filter CSP Spatial Projection → Online LDA Classifier.<br>"
            "Inference latency: &lt;50 µs per streaming frame. Calibrated against sensorimotor desynchronization."
        )
        bci_diag.setWordWrap(True)
        bci_diag.setStyleSheet("color: #7A889B; font-size: 10px; line-height: 15px;")
        tab_layout.addWidget(bci_diag)

        tab_layout.addStretch()

    # --------------------------------------------------------------------------
    # COMPONENT CARD BUILDER (FASTICA)
    # --------------------------------------------------------------------------
    def rebuild_component_cards(self):
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

            chk = QCheckBox(comp.name)
            chk.setChecked(comp.is_rejected)
            chk.setToolTip(f"Check to subtract {comp.name} from reconstructed EEG")
            chk.toggled.connect(lambda checked, idx=comp.idx: self.data_loader.toggle_ic_component(idx, checked))
            card_layout.addWidget(chk)

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

            stats_lbl = QLabel(f"Kurt: {comp.kurtosis:.1f} | Var: {comp.variance_pct:.1f}%")
            stats_lbl.setStyleSheet("color: #7A889B; font-family: 'Consolas', monospace; font-size: 9px;")
            card_layout.addWidget(stats_lbl)

            thumb = ICTraceThumbnailWidget(comp.trace_snippet, color=comp.tag_color, parent=card)
            card_layout.addWidget(thumb, stretch=1)

            self.cards_layout.addWidget(card)

    # --------------------------------------------------------------------------
    # REAL-TIME FRAME SYNCHRONIZATION (CALLED FROM MAIN WINDOW TIMER)
    # --------------------------------------------------------------------------
    def update_analytics_frame(self, current_sample: int, current_time: float, active_channels: List[str]):
        """Updates active analytical modules in real time based on the active tab."""
        current_tab = self.tab_widget.currentIndex()

        # Tab 2: Spectral PSD (Throttle to ~10 Hz to maintain 60 FPS)
        if current_tab == 2:
            if current_time - self._last_psd_update >= 0.10:
                self._update_psd_frame(current_sample, active_channels)
                self._last_psd_update = current_time

        # Tab 3: Time-Frequency ERSP
        elif current_tab == 3:
            for ch in SENSORIMOTOR_CHANNELS:
                if ch in self.ersp_playheads:
                    self.ersp_playheads[ch].setPos(current_time % 4.5 - 1.0)

        # Tab 4: Connectivity Graph (Throttle to ~5 Hz)
        elif current_tab == 4:
            if current_time - self._last_conn_update >= 0.18:
                self._update_conn_frame(current_sample)
                self._last_conn_update = current_time

        # Tab 5: Real-Time BCI Decoder (<50 µs evaluation, updated at ~15 Hz)
        elif current_tab == 5:
            if current_time - self._last_bci_update >= 0.06:
                self._update_bci_frame(current_sample)
                self._last_bci_update = current_time

    def _update_psd_frame(self, current_sample: int, active_channels: List[str]):
        freqs, psd, bands = self.data_loader.get_psd_for_selection(
            current_sample, window_sec=2.0, selected_channels=active_channels
        )
        if psd.ndim > 1:
            mean_psd = np.mean(psd, axis=0)
        else:
            mean_psd = psd
        self.psd_curve.setData(freqs, mean_psd)

        # Update Band Progress Bars
        for band_name, (p_abs, p_rel) in bands.items():
            if band_name in self.band_progress_bars:
                self.band_progress_bars[band_name].setValue(int(round(p_rel)))
                self.band_value_labels[band_name].setText(f"{p_rel:.1f}%")

        if active_channels and len(active_channels) > 0:
            self.psd_channel_lbl.setText(f"<span style='color: #00FFA3;'>●</span> Active Channels: {', '.join(active_channels[:6])}")
        else:
            self.psd_channel_lbl.setText("<span style='color: #00FFA3;'>●</span> Active Channels: Global 64-Channel Array")

    def _refresh_ersp_heatmaps(self):
        ev_text = self.ersp_event_combo.currentText()
        ev_id = 'T1' if 'T1' in ev_text else ('T2' if 'T2' in ev_text else 'T0')

        for ch in SENSORIMOTOR_CHANNELS:
            times, freqs, ersp_db = self.data_loader.get_ersp_map(ch, ev_id)
            if ch in self.ersp_images:
                # Transpose for PyQtGraph ImageItem convention (x, y)
                self.ersp_images[ch].setImage(ersp_db.T, autoLevels=False)
                self.ersp_images[ch].setRect(QRectF(times[0], freqs[0], times[-1] - times[0], freqs[-1] - freqs[0]))

    def _update_conn_frame(self, current_sample: int):
        metric = "PLV" if "PLV" in self.combo_conn_metric.currentText() else "Coherence"
        band = self.combo_conn_band.currentText()
        thresh = float(self.slider_conn_thresh.value()) / 100.0

        conn_mat, edges = self.data_loader.get_connectivity_matrix(current_sample, metric, band)
        self.conn_img.setImage(conn_mat, autoLevels=False)
        self.lbl_active_edges_count.setText(f"Active Edges: {len(edges)}")

    def _update_bci_frame(self, current_sample: int):
        probs, label, conf = self.data_loader.decode_motor_intent(current_sample)

        p_left = probs.get('Left Fist', 0.0) * 100.0
        p_right = probs.get('Right Fist', 0.0) * 100.0
        p_rest = probs.get('Rest', 0.0) * 100.0

        self.pbar_bci_left.setValue(int(round(p_left)))
        self.lbl_p_left.setText(f"{p_left:.1f}%")

        self.pbar_bci_right.setValue(int(round(p_right)))
        self.lbl_p_right.setText(f"{p_right:.1f}%")

        self.pbar_bci_rest.setValue(int(round(p_rest)))
        self.lbl_p_rest.setText(f"{p_rest:.1f}%")

        conf_pct = int(round(conf * 100.0))
        if label == "Left Fist":
            color = "#00FFA3"
            bg = "#0D261C"
        elif label == "Right Fist":
            color = "#BA68C8"
            bg = "#26122B"
        else:
            color = "#00E5FF"
            bg = "#0B1D28"

        self.bci_banner.setStyleSheet(f"QFrame {{ background: {bg}; border: 2px solid {color}; border-radius: 6px; }}")
        self.lbl_bci_intent.setText(f"⚡ DECODED INTENT: {label.upper()} ({conf_pct}% CONFIDENCE)")
        self.lbl_bci_intent.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 900; letter-spacing: 1px;")

    # --------------------------------------------------------------------------
    # EVENT HANDLERS
    # --------------------------------------------------------------------------
    def _on_auto_reject(self):
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

    def _on_ersp_event_changed(self):
        self._refresh_ersp_heatmaps()

    def _on_conn_thresh_changed(self, val: int):
        th = float(val) / 100.0
        self.lbl_conn_thresh.setText(f"Threshold: {th:.2f}")

    def _on_calibrate_bci(self):
        self.data_loader.bci_pipeline.calibrate(self.data_loader.get_active_data(), self.data_loader.events)
        self._update_bci_frame(self.data_loader.n_samples // 2)

    def _toggle_lsl(self):
        if self.lsl_thread is None or not self.lsl_thread.is_running:
            self.lsl_thread = LSLReceiverThread(self.data_loader, self)
            self.lsl_thread.status_changed.connect(self._on_lsl_status_changed)
            self.lsl_thread.metrics_updated.connect(self._on_lsl_metrics_updated)
            self.lsl_thread.connect_stream()
            self.btn_lsl_toggle.setText("Disconnect")
        else:
            self.lsl_thread.disconnect_stream()
            self.btn_lsl_toggle.setText("Connect LSL")
            self.lbl_lsl_status.setText("● LSL: Idle")
            self.lbl_lsl_status.setStyleSheet("color: #8C9BAE; font-size: 10px; font-weight: bold;")

    def _on_lsl_status_changed(self, text: str, color: str):
        self.lbl_lsl_status.setText(f"● {text}")
        self.lbl_lsl_status.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")

    def _on_lsl_metrics_updated(self, eff_fs: float, latency: float):
        self.lbl_lsl_status.setText(f"● LSL: {eff_fs:.0f} Hz ({latency:.1f}ms)")

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
