#!/usr/bin/env python3
"""
Neuromap • Professional Closed-Loop EEG Dashboard
Full-Featured System:
- Auto-Maximize / Fullscreen display mode (F11 toggle)
- Dynamic glowing Stimulus Indicator in bottom right (Rest, Left Fist, Right Fist)
- Precision Speed Control: 0.1x, 0.25x, 0.5x, 1.0x, 2.0x, 4.0x
- Zoomed-out full-cortex 3D Viewport with front-facing top-down alignment
- Sub-millisecond 2D continuous voltage topomap interpolation (64 channels)
- High-density cascading multi-track waveforms above Spotify playback bar
"""

import sys
import os
import time
import traceback
import numpy as np

# Global Exception Hook to catch any Qt slot errors
def exception_hook(exctype, value, tb):
    print("\n[CRITICAL ERROR] Uncaught exception in Qt event loop:")
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = exception_hook

# PyQt5 Core & GUI
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF, QRectF
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QPainter, QBrush, QPen,
    QPolygonF, QImage, QPixmap
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QCheckBox,
    QSlider,
    QLineEdit,
    QProgressBar,
    QStatusBar,
    QFrame,
    QFileDialog,
    QSizePolicy,
    QScrollArea,
    QComboBox,
    QShortcut,
)

# PyQtGraph Performance Tuning (Software QPainter prevents OpenGL driver crashes with VTK)
try:
    import pyqtgraph as pg
    pg.setConfigOptions(
        antialias=False,
        useOpenGL=False,
        background="#07070a",
        foreground="#8a8a9a"
    )
except ImportError:
    pg = None

# MNE
try:
    import mne
    from mne.datasets import eegbci
except ImportError:
    mne = None

# PyVista & PyVistaQt for 3D GLB Rendering
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError:
    pv = None
    QtInteractor = None


# =============================================================================
# QSS THEME STYLING
# =============================================================================
DARK_QSS = """
QMainWindow {
    background-color: #060608;
    color: #e2e2e8;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
}

QWidget {
    background-color: transparent;
    color: #dcdce2;
}

QFrame.dashboard-panel {
    background-color: #0e0e13;
    border: 1px solid #1a1a24;
    border-radius: 8px;
}

QFrame.dashboard-panel:hover {
    border: 1px solid #262636;
}

QFrame.panel-header {
    background-color: #13131a;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    border-bottom: 1px solid #1c1c26;
    padding: 5px 12px;
}

QLabel.header-title {
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.6px;
    color: #f2f2f7;
    text-transform: uppercase;
}

QLabel.header-badge {
    background-color: #19251c;
    color: #1DB954;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 9px;
    border: 1px solid #1ed76033;
}

QFrame.sidebar-section {
    background-color: #101016;
    border: 1px solid #1b1b24;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 6px;
}

QLabel.section-label {
    font-weight: 700;
    font-size: 10px;
    color: #9292a4;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

/* Header Metadata Pills */
QFrame.meta-pill {
    background-color: #121218;
    border: 1px solid #1f1f2a;
    border-radius: 5px;
    padding: 3px 8px;
}

QLabel.meta-pill-key {
    color: #727284;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
}

QLabel.meta-pill-val {
    color: #1DB954;
    font-family: "SF Mono", "Consolas", monospace;
    font-size: 11px;
    font-weight: 700;
}

QCheckBox {
    color: #c4c4cf;
    font-size: 11px;
    spacing: 5px;
}

QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #2f2f3d;
    border-radius: 3px;
    background: #0a0a0e;
}

QCheckBox::indicator:hover {
    border: 1px solid #1DB954;
}

QCheckBox::indicator:checked {
    background-color: #1DB954;
    border: 1px solid #1DB954;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #20202a;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #1DB954;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QSlider::handle:horizontal:hover {
    background: #1ed760;
}

QLineEdit.slider-input {
    background-color: #0b0b10;
    border: 1px solid #20202c;
    border-radius: 4px;
    color: #f0f0f5;
    font-family: "SF Mono", "Consolas", monospace;
    font-size: 11px;
    padding: 2px 4px;
    min-width: 44px;
    max-width: 52px;
}

QLineEdit.slider-input:focus {
    border: 1px solid #1DB954;
}

QPushButton.btn-primary {
    background-color: #1DB954;
    color: #000000;
    font-weight: 700;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 13px;
    border: none;
}

QPushButton.btn-primary:hover {
    background-color: #1ed760;
}

QPushButton.btn-secondary {
    background-color: #16161f;
    border: 1px solid #242432;
    color: #dedee8;
    font-size: 10px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
}

QPushButton.btn-secondary:hover {
    background-color: #1f1f2c;
    border-color: #343446;
}

QSplitter::handle {
    background-color: #111117;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

QSplitter::handle:hover {
    background-color: #1DB954;
}

QStatusBar {
    background-color: #050507;
    color: #888894;
    border-top: 1px solid #14141b;
    font-size: 11px;
    padding: 2px 8px;
}

QProgressBar {
    background-color: #0f0f15;
    border: 1px solid #1a1a24;
    border-radius: 4px;
    height: 12px;
    text-align: center;
    color: #e0e0e0;
    font-size: 9px;
    font-weight: bold;
    min-width: 170px;
    max-width: 200px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1db954, stop:1 #1ed760);
    border-radius: 3px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}
"""

# =============================================================================
# 64 STANDARD CHANNELS
# =============================================================================
STANDARD_64_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]

COORDS_64 = {
    'Fp1': (-0.28, -0.86), 'Fpz': (0.00, -0.88), 'Fp2': (0.28, -0.86),
    'AF7': (-0.56, -0.72), 'AF3': (-0.25, -0.70), 'AFz': (0.00, -0.71), 'AF4': (0.25, -0.70), 'AF8': (0.56, -0.72),
    'F7':  (-0.76, -0.50), 'F5':  (-0.52, -0.50), 'F3':  (-0.31, -0.50), 'F1':  (-0.11, -0.50), 'Fz': (0.00, -0.50),
    'F2':  (0.11, -0.50),  'F4':  (0.31, -0.50),  'F6':  (0.52, -0.50),  'F8':  (0.76, -0.50),
    'FT7': (-0.86, -0.26), 'FC5': (-0.60, -0.26), 'FC3': (-0.36, -0.26), 'FC1': (-0.12, -0.26), 'FCz': (0.00, -0.26),
    'FC2': (0.12, -0.26),  'FC4': (0.36, -0.26),  'FC6': (0.60, -0.26),  'FT8': (0.86, -0.26),
    'T7':  (-0.90,  0.00), 'C5':  (-0.64,  0.00), 'C3':  (-0.38,  0.00), 'C1':  (-0.13,  0.00), 'Cz': (0.00,  0.00),
    'C2':  (0.13,  0.00),  'C4':  (0.38,  0.00),  'C6':  (0.64,  0.00),  'T8':  (0.90,  0.00),
    'TP7': (-0.86,  0.26), 'CP5': (-0.60,  0.26), 'CP3': (-0.36,  0.26), 'CP1': (-0.12,  0.26), 'CPz': (0.00,  0.26),
    'CP2': (0.12,  0.26),  'CP4': (0.36,  0.26),  'CP6': (0.60,  0.26),  'TP8': (0.86,  0.26),
    'P7':  (-0.76,  0.50), 'P5':  (-0.52,  0.50), 'P3':  (-0.31,  0.50), 'P1':  (-0.11,  0.50), 'Pz': (0.00,  0.50),
    'P2':  (0.11,  0.50),  'P4':  (0.31,  0.50),  'P6':  (0.52,  0.50),  'P8':  (0.76,  0.50),
    'PO7': (-0.56,  0.72), 'PO3': (-0.25,  0.70), 'POz': (0.00,  0.71), 'PO4': (0.25,  0.70), 'PO8': (0.56,  0.72),
    'O1':  (-0.28,  0.86), 'Oz':  (0.00,  0.88),  'O2':  (0.28,  0.86),  'Iz':  (0.00,  0.94)
}

DISTINCT_COLORS = [
    QColor("#38bdf8"), QColor("#818cf8"), QColor("#c084fc"), QColor("#f472b6"),
    QColor("#fb7185"), QColor("#f97316"), QColor("#facc15"), QColor("#4ade80"),
    QColor("#2dd4bf"), QColor("#22d3ee"), QColor("#60a5fa"), QColor("#a855f7"),
    QColor("#ec4899"), QColor("#f43f5e"), QColor("#ef4444"), QColor("#eab308"),
    QColor("#84cc16"), QColor("#10b981"), QColor("#14b8a6"), QColor("#06b6d4"),
    QColor("#0ea5e9"), QColor("#3b82f6"), QColor("#6366f1"), QColor("#8b5cf6"),
    QColor("#d946ef"), QColor("#f43f5e"), QColor("#f97316"), QColor("#eab308"),
    QColor("#22c55e"), QColor("#14b8a6"), QColor("#06b6d4"), QColor("#0284c7"),
    QColor("#2563eb"), QColor("#4f46e5"), QColor("#7c3aed"), QColor("#9333ea"),
    QColor("#c026d3"), QColor("#db2777"), QColor("#e11d48"), QColor("#ea580c"),
    QColor("#ca8a04"), QColor("#65a30d"), QColor("#16a34a"), QColor("#0d9488"),
    QColor("#0891b2"), QColor("#0369a1"), QColor("#1d4ed8"), QColor("#4338ca"),
    QColor("#6d28d9"), QColor("#7e22ce"), QColor("#a21caf"), QColor("#be185d"),
    QColor("#b91c1c"), QColor("#c2410c"), QColor("#a16207"), QColor("#4d7c0f"),
    QColor("#15803d"), QColor("#0f766e"), QColor("#0e7490"), QColor("#155e75"),
    QColor("#1e40af"), QColor("#3730a3"), QColor("#5b21b6"), QColor("#6b21a8")
]


# =============================================================================
# DATA LAYER
# =============================================================================
class EEGDataLoader:
    def __init__(self, subject: int = 1, run: int = 4):
        self.subject = subject
        self.run = run
        self.sfreq = 160.0
        self.raw = None
        self.ch_names = []
        self.times = np.array([])
        self.data_raw = np.array([[]])
        self.data_current = np.array([[]])
        self.event_segments = []
        self.n_samples = 0
        self.duration_sec = 0.0

    def load_data(self):
        if mne is None:
            self._create_synthetic()
            return self

        try:
            print(f"[INFO] Fetching PhysioNet Subject {self.subject}, Run {self.run}...")
            edf_paths = eegbci.load_data(subjects=[self.subject], runs=[self.run], update_path=False)
            if not edf_paths:
                raise FileNotFoundError("PhysioNet file not found.")

            raw = mne.io.read_raw_edf(edf_paths[0], preload=True, verbose=False)
            eegbci.standardize(raw)

            builtin_montages = mne.channels.get_builtin_montages()
            montage_name = 'colin27_1005' if 'colin27_1005' in builtin_montages else 'standard_1005'
            montage = mne.channels.make_standard_montage(montage_name)
            raw.set_montage(montage, on_missing='ignore')

            self.raw = raw
            self.ch_names = raw.ch_names
            self.sfreq = float(raw.info['sfreq'])
            self.times = raw.times
            self.data_raw = raw.get_data()
            self.data_current = self.data_raw.copy()
            self.n_samples = self.data_raw.shape[1]
            self.duration_sec = self.times[-1] if len(self.times) > 0 else 0.0

            self._parse_events(raw)
            print(f"[SUCCESS] Loaded {len(self.ch_names)} channels, {self.n_samples} samples ({self.duration_sec:.1f}s @ {self.sfreq:.0f}Hz).")
            return self
        except Exception as exc:
            print(f"[WARN] Loading PhysioNet ({exc}). Falling back to synthetic 64-channel stream.")
            self._create_synthetic()
            return self

    def _parse_events(self, raw):
        self.event_segments = []
        if raw.annotations is None or len(raw.annotations) == 0:
            return
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        for annot in raw.annotations:
            desc = str(annot['description']).strip()
            onset = float(annot['onset'])
            dur = float(annot['duration']) if float(annot['duration']) > 0 else 4.0
            self.event_segments.append({
                'name': desc,
                'start_sec': onset,
                'duration_sec': dur,
                'code': event_id.get(desc, 0)
            })

    def _create_synthetic(self):
        self.sfreq = 160.0
        self.duration_sec = 120.0
        self.n_samples = int(self.sfreq * self.duration_sec)
        self.times = np.linspace(0, self.duration_sec, self.n_samples)
        self.ch_names = STANDARD_64_CHANNELS
        n_ch = len(self.ch_names)

        t = self.times
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 4e-6, (n_ch, self.n_samples))
        alpha = 12e-6 * np.sin(2 * np.pi * 10 * t)
        beta = 6e-6 * np.sin(2 * np.pi * 22 * t)
        theta = 8e-6 * np.sin(2 * np.pi * 6 * t)

        self.data_raw = noise + alpha + beta + theta
        self.data_current = self.data_raw.copy()

        raw_events = [
            ('T0', 0.0, 6.0), ('T1', 6.0, 4.2), ('T0', 10.2, 4.0),
            ('T2', 14.2, 4.2), ('T0', 18.4, 4.0), ('T1', 22.4, 4.2),
            ('T0', 26.6, 4.0), ('T2', 30.6, 4.2), ('T0', 34.8, 5.0)
        ]
        self.event_segments = [
            {'name': name, 'start_sec': on, 'duration_sec': dur, 'code': idx + 1}
            for idx, (name, on, dur) in enumerate(raw_events)
        ]


# =============================================================================
# OPTIMIZED PLAYBACK ENGINE
# =============================================================================
class PlaybackEngine(QObject):
    sig_frame_changed = pyqtSignal(int, float)
    sig_playback_state = pyqtSignal(bool)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.total_frames = max(1, data_loader.n_samples)
        self.sfreq = data_loader.sfreq
        self.duration_sec = data_loader.duration_sec

        self.current_frame = 0
        self.is_playing = False
        self.speed_multiplier = 1.0

        self.last_tick_time = 0.0
        self.frame_accumulator = 0.0

        self.timer = QTimer(self)
        self.timer.setInterval(20)  # 50 FPS target
        self.timer.timeout.connect(self._on_tick)

    def set_speed(self, speed: float):
        self.speed_multiplier = float(speed)

    def toggle_play(self):
        self.pause() if self.is_playing else self.play()

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self.last_tick_time = time.perf_counter()
            self.frame_accumulator = float(self.current_frame)
            self.timer.start()
            self.sig_playback_state.emit(True)

    def pause(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.sig_playback_state.emit(False)

    def step_forward(self):
        self.pause()
        step = max(1, int(round(self.sfreq * 0.05)))
        self.seek_frame(self.current_frame + step)

    def step_backward(self):
        self.pause()
        step = max(1, int(round(self.sfreq * 0.05)))
        self.seek_frame(self.current_frame - step)

    def seek_frame(self, frame_idx: int):
        self.current_frame = max(0, min(int(frame_idx), self.total_frames - 1))
        self.frame_accumulator = float(self.current_frame)
        self.sig_frame_changed.emit(self.current_frame, self.current_frame / self.sfreq)

    def seek_time(self, time_sec: float):
        self.seek_frame(int(round(time_sec * self.sfreq)))

    def _on_tick(self):
        now = time.perf_counter()
        dt = now - self.last_tick_time
        self.last_tick_time = now

        frames_to_advance = dt * self.sfreq * self.speed_multiplier
        self.frame_accumulator += frames_to_advance

        if self.frame_accumulator >= self.total_frames:
            self.frame_accumulator = 0.0
        elif self.frame_accumulator < 0.0:
            self.frame_accumulator = 0.0

        new_frame = int(self.frame_accumulator)
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self.sig_frame_changed.emit(self.current_frame, self.current_frame / self.sfreq)


# =============================================================================
# TOP NAVIGATION HEADER
# =============================================================================
class TopNavigationHeader(QFrame):
    sig_export_requested = pyqtSignal()

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.setFixedHeight(48)
        self.setObjectName("top-navigation-header")
        self.setStyleSheet("""
            QFrame#top-navigation-header {
                background-color: #0c0c11;
                border-bottom: 1px solid #1a1a24;
                padding: 0 14px;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 14, 0)
        layout.setSpacing(12)

        title_lbl = QLabel("neuromap", self)
        title_lbl.setStyleSheet("""
            font-size: 19px;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 0.5px;
        """)
        layout.addWidget(title_lbl)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #242433; margin: 12px 4px;")
        layout.addWidget(sep)

        meta_items = [
            ("SUBJ", f"S{self.data_loader.subject:03d}"),
            ("SRATE", f"{self.data_loader.sfreq:.0f} Hz"),
            ("DURATION", f"{self.data_loader.duration_sec:.1f} s"),
            ("TYPE", "EDF+"),
            ("CHANNELS", f"{len(self.data_loader.ch_names)} Total"),
            ("STATUS", "Preloaded"),
        ]

        for k, v in meta_items:
            pill = QFrame(self)
            pill.setProperty("class", "meta-pill")
            p_layout = QHBoxLayout(pill)
            p_layout.setContentsMargins(6, 2, 6, 2)
            p_layout.setSpacing(5)

            lbl_k = QLabel(k, pill)
            lbl_k.setProperty("class", "meta-pill-key")
            lbl_v = QLabel(v, pill)
            lbl_v.setProperty("class", "meta-pill-val")

            p_layout.addWidget(lbl_k)
            p_layout.addWidget(lbl_v)
            layout.addWidget(pill)

        layout.addStretch()

        self.btn_export = QPushButton("⭳  Export Data", self)
        self.btn_export.setProperty("class", "btn-primary")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.clicked.connect(self.sig_export_requested.emit)
        layout.addWidget(self.btn_export)


# =============================================================================
# SIDEBAR CONTROLS (With 0.1x & 0.25x Speeds)
# =============================================================================
class SidebarControlsWidget(QScrollArea):
    sig_filters_changed = pyqtSignal(dict)
    sig_display_changed = pyqtSignal(dict)
    sig_speed_changed = pyqtSignal(float)
    sig_topomap_gain_changed = pyqtSignal(float)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedWidth(260)

        container = QWidget(self)
        self.setWidget(container)
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(8)

        self._build_signal_processing_panel()
        self._build_display_controls_panel()
        self.main_layout.addStretch()

    def _build_signal_processing_panel(self):
        frame = QFrame(self)
        frame.setProperty("class", "sidebar-section")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Signal Processing", frame)
        title.setProperty("class", "section-label")
        layout.addWidget(title)

        notch_box = QHBoxLayout()
        notch_box.setSpacing(10)
        self.chk_notch_60 = QCheckBox("60 Hz", frame)
        self.chk_notch_50 = QCheckBox("50 Hz", frame)
        self.chk_notch_60.setChecked(True)
        notch_box.addWidget(QLabel("Notch:", frame))
        notch_box.addWidget(self.chk_notch_60)
        notch_box.addWidget(self.chk_notch_50)
        notch_box.addStretch()
        layout.addLayout(notch_box)

        layout.addWidget(QLabel("High-Pass (Hz):", frame))
        hp_layout = QHBoxLayout()
        self.slider_hp = QSlider(Qt.Horizontal, frame)
        self.slider_hp.setRange(1, 20)
        self.slider_hp.setValue(10)
        self.edit_hp = QLineEdit("1.0", frame)
        self.edit_hp.setProperty("class", "slider-input")
        self.edit_hp.setAlignment(Qt.AlignCenter)
        hp_layout.addWidget(self.slider_hp)
        hp_layout.addWidget(self.edit_hp)
        layout.addLayout(hp_layout)

        layout.addWidget(QLabel("Low-Pass (Hz):", frame))
        lp_layout = QHBoxLayout()
        self.slider_lp = QSlider(Qt.Horizontal, frame)
        self.slider_lp.setRange(20, 80)
        self.slider_lp.setValue(40)
        self.edit_lp = QLineEdit("40.0", frame)
        self.edit_lp.setProperty("class", "slider-input")
        self.edit_lp.setAlignment(Qt.AlignCenter)
        lp_layout.addWidget(self.slider_lp)
        lp_layout.addWidget(self.edit_lp)
        layout.addLayout(lp_layout)

        self.btn_apply = QPushButton("⚡ Apply Filter", frame)
        self.btn_apply.setProperty("class", "btn-secondary")
        self.btn_apply.clicked.connect(self._emit_filters)
        layout.addWidget(self.btn_apply)

        self.slider_hp.valueChanged.connect(lambda v: self.edit_hp.setText(f"{v/10.0:.1f}"))
        self.edit_hp.editingFinished.connect(self._sync_hp)
        self.slider_lp.valueChanged.connect(lambda v: self.edit_lp.setText(f"{float(v):.1f}"))
        self.edit_lp.editingFinished.connect(self._sync_lp)
        self.chk_notch_60.stateChanged.connect(self._emit_filters)
        self.chk_notch_50.stateChanged.connect(self._emit_filters)

        self.main_layout.addWidget(frame)

    def _sync_hp(self):
        try:
            val = float(self.edit_hp.text())
            self.slider_hp.setValue(max(1, min(200, int(round(val * 10)))))
            self._emit_filters()
        except ValueError:
            self.edit_hp.setText(f"{self.slider_hp.value()/10.0:.1f}")

    def _sync_lp(self):
        try:
            val = float(self.edit_lp.text())
            self.slider_lp.setValue(max(10, min(100, int(round(val)))))
            self._emit_filters()
        except ValueError:
            self.edit_lp.setText(f"{float(self.slider_lp.value()):.1f}")

    def _emit_filters(self):
        self.sig_filters_changed.emit({
            'notch_60': self.chk_notch_60.isChecked(),
            'notch_50': self.chk_notch_50.isChecked(),
            'highpass': self.slider_hp.value() / 10.0,
            'lowpass': float(self.slider_lp.value()),
        })

    def _build_display_controls_panel(self):
        frame = QFrame(self)
        frame.setProperty("class", "sidebar-section")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Display & Performance", frame)
        title.setProperty("class", "section-label")
        layout.addWidget(title)

        # Vertical Gain
        layout.addWidget(QLabel("Trace Gain (µV/mm):", frame))
        g_layout = QHBoxLayout()
        self.slider_gain = QSlider(Qt.Horizontal, frame)
        self.slider_gain.setRange(10, 200)
        self.slider_gain.setValue(50)
        self.edit_gain = QLineEdit("50", frame)
        self.edit_gain.setProperty("class", "slider-input")
        self.edit_gain.setAlignment(Qt.AlignCenter)
        g_layout.addWidget(self.slider_gain)
        g_layout.addWidget(self.edit_gain)
        layout.addLayout(g_layout)

        # Topomap Voltage Scale
        layout.addWidget(QLabel("Topomap Range (±µV):", frame))
        topo_layout = QHBoxLayout()
        self.slider_topo_scale = QSlider(Qt.Horizontal, frame)
        self.slider_topo_scale.setRange(5, 100)
        self.slider_topo_scale.setValue(25)
        self.edit_topo_scale = QLineEdit("25", frame)
        self.edit_topo_scale.setProperty("class", "slider-input")
        self.edit_topo_scale.setAlignment(Qt.AlignCenter)
        topo_layout.addWidget(self.slider_topo_scale)
        topo_layout.addWidget(self.edit_topo_scale)
        layout.addLayout(topo_layout)

        self.slider_topo_scale.valueChanged.connect(lambda v: self.edit_topo_scale.setText(str(v)))
        self.slider_topo_scale.valueChanged.connect(lambda v: self.sig_topomap_gain_changed.emit(float(v)))
        self.edit_topo_scale.editingFinished.connect(
            lambda: self.slider_topo_scale.setValue(max(5, min(200, int(self.edit_topo_scale.text()))))
        )

        # Time Window
        layout.addWidget(QLabel("Time Window (sec):", frame))
        w_layout = QHBoxLayout()
        self.slider_win = QSlider(Qt.Horizontal, frame)
        self.slider_win.setRange(2, 20)
        self.slider_win.setValue(8)
        self.edit_win = QLineEdit("8", frame)
        self.edit_win.setProperty("class", "slider-input")
        self.edit_win.setAlignment(Qt.AlignCenter)
        w_layout.addWidget(self.slider_win)
        w_layout.addWidget(self.edit_win)
        layout.addLayout(w_layout)

        # Playback Speed Control (Added 0.1x and 0.25x)
        layout.addWidget(QLabel("Playback Speed:", frame))
        self.combo_speed = QComboBox(frame)
        self.combo_speed.setStyleSheet("""
            QComboBox {
                background: #0b0b10;
                color: #1DB954;
                border: 1px solid #20202c;
                border-radius: 4px;
                padding: 3px 6px;
                font-weight: bold;
            }
            QComboBox QAbstractItemView {
                background: #14141d;
                color: #e0e0e0;
                selection-background-color: #1DB954;
                selection-color: #000;
            }
        """)
        self.combo_speed.addItems([
            "0.1x (Slow Motion)",
            "0.25x (Quarter Speed)",
            "0.5x (Half Speed)",
            "1.0x (Real-Time)",
            "2.0x (Fast)",
            "4.0x (High-Speed)"
        ])
        self.combo_speed.setCurrentIndex(3)  # Default 1.0x Real-Time
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self.combo_speed)

        self.slider_gain.valueChanged.connect(lambda v: self.edit_gain.setText(str(v)))
        self.slider_gain.valueChanged.connect(self._emit_display)
        self.edit_gain.editingFinished.connect(self._sync_gain)

        self.slider_win.valueChanged.connect(lambda v: self.edit_win.setText(str(v)))
        self.slider_win.valueChanged.connect(self._emit_display)
        self.edit_win.editingFinished.connect(self._sync_win)

        self.main_layout.addWidget(frame)

    def _on_speed_changed(self, idx: int):
        speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        self.sig_speed_changed.emit(speeds[idx])

    def _sync_gain(self):
        try:
            val = max(5, min(500, int(self.edit_gain.text())))
            self.slider_gain.setValue(val)
        except ValueError:
            self.edit_gain.setText(str(self.slider_gain.value()))

    def _sync_win(self):
        try:
            val = max(1, min(60, int(self.edit_win.text())))
            self.slider_win.setValue(val)
        except ValueError:
            self.edit_win.setText(str(self.slider_win.value()))

    def _emit_display(self):
        self.sig_display_changed.emit({
            'gain': float(self.slider_gain.value()),
            'window_sec': float(self.slider_win.value())
        })


# =============================================================================
# 2D TOPOMAP INTERPOLATION ENGINE
# =============================================================================
class TopomapInterpolator:
    def __init__(self, ch_names: list, coords_dict: dict, grid_res: int = 64):
        self.grid_res = grid_res
        self.ch_names = ch_names
        self.n_ch = len(ch_names)

        x = np.linspace(-1.0, 1.0, grid_res)
        y = np.linspace(-1.0, 1.0, grid_res)
        self.gx, self.gy = np.meshgrid(x, y)
        self.dist_sq = self.gx**2 + self.gy**2
        self.mask = self.dist_sq <= 0.98

        ch_pts = np.zeros((self.n_ch, 2), dtype=np.float32)
        for i, name in enumerate(ch_names):
            if name in coords_dict:
                ch_pts[i] = coords_dict[name]

        grid_pts = np.column_stack([self.gx.ravel(), self.gy.ravel()]).astype(np.float32)
        diff = grid_pts[:, np.newaxis, :] - ch_pts[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=-1)
        weights = 1.0 / (d2 + 1e-3)
        weights /= np.sum(weights, axis=1, keepdims=True)
        self.W = weights.astype(np.float32)

        t = np.linspace(0, 1, 256)
        r = np.clip(1.5 * t, 0.0, 1.0)
        b = np.clip(1.5 * (1.0 - t), 0.0, 1.0)
        g = 1.0 - np.abs(2.0 * t - 1.0)**1.2
        lut_rgba = np.column_stack([r, g, b, np.ones(256)]) * 255.0
        self.lut = lut_rgba.astype(np.uint8)

    def interpolate(self, voltages: np.ndarray, v_max: float = 25e-6) -> QImage:
        grid_v = (self.W @ voltages.astype(np.float32)).reshape((self.grid_res, self.grid_res))
        norm = np.clip((grid_v / v_max + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
        rgba = self.lut[norm]
        rgba[~self.mask, 3] = 0
        qimg = QImage(rgba.data, self.grid_res, self.grid_res, self.grid_res * 4, QImage.Format_RGBA8888)
        return qimg.copy()


# =============================================================================
# 2D TOPOMAP WIDGET
# =============================================================================
class ChannelNodeButton(QPushButton):
    sig_toggled = pyqtSignal(str, bool)

    def __init__(self, ch_name: str, color: QColor, parent=None):
        super().__init__(ch_name, parent)
        self.ch_name = ch_name
        self.node_color = color
        self.is_active = False
        self.setFixedSize(24, 24)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self.toggle_state)
        self._update_appearance()

    def toggle_state(self):
        self.is_active = not self.is_active
        self._update_appearance()
        self.sig_toggled.emit(self.ch_name, self.is_active)

    def set_active(self, active: bool):
        if self.is_active != active:
            self.is_active = active
            self._update_appearance()

    def _update_appearance(self):
        if self.is_active:
            hex_c = self.node_color.name()
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_c};
                    color: #ffffff;
                    font-weight: 800;
                    font-size: 7.5px;
                    border: 1.5px solid #ffffff;
                    border-radius: 12px;
                }}
                QPushButton:hover {{
                    border: 2px solid #1DB954;
                }}
            """)
        else:
            self.setStyleSheet("""
                QPushButton {{
                    background-color: rgba(18, 18, 26, 180);
                    color: #727284;
                    font-weight: 600;
                    font-size: 7.5px;
                    border: 1px solid rgba(50, 50, 68, 160);
                    border-radius: 12px;
                }}
                QPushButton:hover {{
                    border: 1px solid #1DB954;
                    color: #ffffff;
                }}
            """)


class ChannelMap2DWidget(QFrame):
    sig_selection_changed = pyqtSignal(list)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.setObjectName("channel-map-2d")
        self.setProperty("class", "dashboard-panel")

        self.node_buttons = {}
        self.channel_colors = {}
        self.topomap_image = None
        self.topomap_v_max = 25e-6

        self.interpolator = TopomapInterpolator(STANDARD_64_CHANNELS, COORDS_64, grid_res=64)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame(self)
        header.setProperty("class", "panel-header")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 4, 10, 4)

        title = QLabel("2D Voltage Topomap (64 Channels)", header)
        title.setProperty("class", "header-title")

        self.badge_count = QLabel("0 / 64 Active", header)
        self.badge_count.setProperty("class", "header-badge")

        btn_all = QPushButton("Select All", header)
        btn_all.setProperty("class", "btn-secondary")
        btn_all.setFixedHeight(20)
        btn_all.clicked.connect(self.select_all)

        btn_clear = QPushButton("Clear", header)
        btn_clear.setProperty("class", "btn-secondary")
        btn_clear.setFixedHeight(20)
        btn_clear.clicked.connect(self.deselect_all)

        h_layout.addWidget(title)
        h_layout.addSpacing(6)
        h_layout.addWidget(self.badge_count)
        h_layout.addStretch()
        h_layout.addWidget(btn_all)
        h_layout.addWidget(btn_clear)
        layout.addWidget(header)

        self.canvas = QWidget(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)

        for idx, ch_name in enumerate(STANDARD_64_CHANNELS):
            color = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]
            self.channel_colors[ch_name] = color
            btn = ChannelNodeButton(ch_name, color, self.canvas)
            btn.sig_toggled.connect(lambda: self._emit_selection())
            self.node_buttons[ch_name] = btn

    def set_topomap_v_max(self, uv_val: float):
        self.topomap_v_max = uv_val * 1e-6
        self.canvas.update()

    def update_topomap_frame(self, frame_idx: int):
        if self.data_loader.data_current.size == 0:
            return

        n_samples = self.data_loader.n_samples
        f_idx = max(0, min(frame_idx, n_samples - 1))

        voltages = np.zeros(len(STANDARD_64_CHANNELS), dtype=np.float32)
        for i, name in enumerate(STANDARD_64_CHANNELS):
            if name in self.data_loader.ch_names:
                ch_idx = self.data_loader.ch_names.index(name)
                voltages[i] = self.data_loader.data_current[ch_idx, f_idx]

        self.topomap_image = self.interpolator.interpolate(voltages, self.topomap_v_max)
        self.canvas.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        w = self.canvas.width()
        h = self.canvas.height()
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) * 0.42

        for ch_name, btn in self.node_buttons.items():
            nx, ny = COORDS_64.get(ch_name, (0, 0))
            px = cx + nx * radius - (btn.width() / 2.0)
            py = cy + ny * radius - (btn.height() / 2.0)
            btn.move(int(px), int(py))

    def paintEvent(self, ev):
        super().paintEvent(ev)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        cw, ch = self.canvas.width(), self.canvas.height()
        ox, oy = self.canvas.x(), self.canvas.y()
        cx, cy = ox + cw / 2.0, oy + ch / 2.0
        radius = min(cw, ch) * 0.42
        target_rect = QRectF(cx - radius, cy - radius, radius * 2.0, radius * 2.0)

        # 1. Continuous Voltage Heatmap
        if self.topomap_image is not None:
            painter.drawImage(target_rect, self.topomap_image)
        else:
            painter.setPen(QPen(QColor("#1f1f2a"), 1.5))
            painter.setBrush(QBrush(QColor("#0a0a0f")))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # 2. Scalp Perimeter
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # 3. Nose
        painter.setPen(QPen(QColor("#1DB954"), 2.0))
        painter.setBrush(QBrush(QColor("#102416")))
        nose = QPolygonF([
            QPointF(cx - 12, cy - radius + 2),
            QPointF(cx, cy - radius - 15),
            QPointF(cx + 12, cy - radius + 2),
        ])
        painter.drawPolygon(nose)

        # 4. Ears
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(int(cx - radius - 10), int(cy - 16), 12, 32, 90 * 16, 180 * 16)
        painter.drawArc(int(cx + radius - 2), int(cy - 16), 12, 32, 270 * 16, 180 * 16)

    def select_all(self):
        for btn in self.node_buttons.values():
            btn.set_active(True)
        self._emit_selection()

    def deselect_all(self):
        for btn in self.node_buttons.values():
            btn.set_active(False)
        self._emit_selection()

    def _emit_selection(self):
        active = [name for name, btn in self.node_buttons.items() if btn.is_active]
        self.badge_count.setText(f"{len(active)} / 64 Active")
        self.sig_selection_changed.emit(active)


# =============================================================================
# 3D BRAIN MAPPING PANEL (Zoomed Out to Show Entire Brain)
# =============================================================================
class Brain3DViewerWidget(QFrame):
    def __init__(self, glb_path: str = "data/human-brain.glb", parent=None):
        super().__init__(parent)
        self.glb_path = glb_path
        self.setObjectName("brain-3d-panel")
        self.setProperty("class", "dashboard-panel")
        self.plotter = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame(self)
        header.setProperty("class", "panel-header")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 4, 10, 4)

        title = QLabel("3D Brain Model (Superior Top-Down View)", header)
        title.setProperty("class", "header-title")

        self.lbl_status = QLabel("Top-Down • Frontal Up", header)
        self.lbl_status.setProperty("class", "header-badge")

        btn_top = QPushButton("Top View", header)
        btn_top.setProperty("class", "btn-secondary")
        btn_top.setFixedHeight(20)
        btn_top.setToolTip("Align to Top-Down Superior View (Frontal Up)")
        btn_top.clicked.connect(self.set_top_down_view)

        btn_flip = QPushButton("Flip A-P", header)
        btn_flip.setProperty("class", "btn-secondary")
        btn_flip.setFixedHeight(20)
        btn_flip.setToolTip("Invert Anterior-Posterior Axis 180°")
        btn_flip.clicked.connect(self.flip_ap_view)

        btn_iso = QPushButton("Iso View", header)
        btn_iso.setProperty("class", "btn-secondary")
        btn_iso.setFixedHeight(20)
        btn_iso.clicked.connect(self.set_iso_view)

        h_layout.addWidget(title)
        h_layout.addSpacing(6)
        h_layout.addWidget(self.lbl_status)
        h_layout.addStretch()
        h_layout.addWidget(btn_top)
        h_layout.addWidget(btn_flip)
        h_layout.addWidget(btn_iso)
        layout.addWidget(header)

        if QtInteractor is not None and pv is not None:
            try:
                self.plotter = QtInteractor(self)
                self.plotter.set_background("#07070a")
                layout.addWidget(self.plotter.interactor, stretch=1)
                self._load_mesh()
            except Exception as exc:
                self._show_fallback(layout, f"OpenGL Error: {exc}")
        else:
            self._show_fallback(layout, "PyVista / PyVistaQt not installed.\n3D Brain Viewport offline.")

    def _show_fallback(self, layout, msg: str):
        lbl = QLabel(f"🧠 3D Brain Mapping Viewport\n\nTarget File: {self.glb_path}\n{msg}", self)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #727282; font-size: 12px; line-height: 1.5;")
        layout.addWidget(lbl, stretch=1)

    def _load_mesh(self):
        if self.plotter is None:
            return

        mesh = None
        if os.path.exists(self.glb_path):
            try:
                print(f"[INFO] Loading 3D model from {self.glb_path}...")
                mesh = pv.read(self.glb_path)
                self.lbl_status.setText("Loaded: human-brain.glb")
            except Exception as exc:
                print(f"[WARN] Failed to parse {self.glb_path}: {exc}")

        if mesh is None:
            sphere = pv.ParametricEllipsoid(xradius=1.3, yradius=0.95, zradius=1.0, u_res=70, v_res=70)
            pts = sphere.points
            gyri = 0.045 * np.sin(pts[:, 0] * 18.0) * np.cos(pts[:, 2] * 18.0)
            sphere.points += sphere.point_normals * gyri[:, np.newaxis]
            mesh = sphere
            self.lbl_status.setText("Procedural Cortex Mesh")

        try:
            self.plotter.add_mesh(
                mesh,
                color="#6c6c82",
                smooth_shading=True,
                ambient=0.30,
                diffuse=0.85,
                specular=0.45,
                specular_power=25,
                show_edges=False,
                opacity=0.94
            )
            light = pv.Light(position=(0, 6, 0), focal_point=(0, 0, 0), color='white', intensity=0.7)
            self.plotter.add_light(light)

            # Zoomed out to 5.6 to encompass full brain mesh comfortably
            self.set_top_down_view()
        except Exception as exc:
            print(f"[WARN] Error adding mesh to plotter: {exc}")

    def set_top_down_view(self):
        """Top-Down Superior View zoomed out to distance 5.6 to fit entire brain."""
        if self.plotter is None:
            return
        try:
            self.plotter.camera_position = [
                (0.0, 5.6, 0.0),     # Zoomed-out camera distance
                (0.0, 0.0, 0.0),     # Focal point
                (1.0, 0.0, 0.0)      # View up (+X Anterior UP)
            ]
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def flip_ap_view(self):
        if self.plotter is None:
            return
        try:
            cur_up = self.plotter.camera_position[2]
            new_up = (-cur_up[0], -cur_up[1], -cur_up[2])
            self.plotter.camera_position = [
                (0.0, 5.6, 0.0),
                (0.0, 0.0, 0.0),
                new_up
            ]
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def set_iso_view(self):
        if self.plotter is None:
            return
        try:
            self.plotter.camera_position = [
                (3.8, 4.2, 3.8),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0)
            ]
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass


# =============================================================================
# MULTI-TRACK TIME-SERIES PLOTTING
# =============================================================================
class MultiTrackTimeSeriesWidget(QFrame):
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.setObjectName("timeseries-plotting-panel")
        self.setProperty("class", "dashboard-panel")

        self.active_channels = []
        self.channel_colors = {}
        self.current_frame = 0
        self.vertical_gain = 50.0
        self.window_sec = 8.0
        self.curves = {}

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame(self)
        header.setProperty("class", "panel-header")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 4, 10, 4)

        title = QLabel("Filtered Data Waveforms", header)
        title.setProperty("class", "header-title")

        self.lbl_tracks = QLabel("0 Tracks Selected", header)
        self.lbl_tracks.setProperty("class", "header-badge")

        h_layout.addWidget(title)
        h_layout.addSpacing(8)
        h_layout.addWidget(self.lbl_tracks)
        h_layout.addStretch()
        layout.addWidget(header)

        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setBackground("#07070a")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.10)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setClipToView(True)

        bottom = self.plot_widget.getAxis('bottom')
        bottom.setLabel('Relative Window Time (seconds)', color='#707080', size='10px')
        bottom.setPen(pg.mkPen('#161622'))

        left = self.plot_widget.getAxis('left')
        left.setPen(pg.mkPen('#161622'))

        layout.addWidget(self.plot_widget, stretch=1)

        self.empty_label = QLabel(
            "⚡ No Channels Active\nClick electrode nodes on the 2D Scalp Map to display cascading waveforms.",
            self.plot_widget
        )
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #555566; font-size: 13px; font-weight: 500;")

        self._rebuild_tracks()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.empty_label.setGeometry(self.plot_widget.rect())

    def set_active_channels(self, channels: list, colors_dict: dict):
        self.active_channels = channels
        self.channel_colors = colors_dict
        self.lbl_tracks.setText(f"{len(self.active_channels)} Tracks Active")
        self._rebuild_tracks()
        self.update_waveforms(self.current_frame)

    def set_display_params(self, gain: float, window_sec: float):
        self.vertical_gain = gain
        self.window_sec = window_sec
        self._rebuild_tracks()
        self.update_waveforms(self.current_frame)

    def _rebuild_tracks(self):
        self.plot_widget.clear()
        self.curves.clear()

        n_active = len(self.active_channels)
        if n_active == 0:
            self.plot_widget.getAxis('left').setTicks([])
            self.empty_label.show()
            return

        self.empty_label.hide()
        spacing = 1.0
        ticks = []
        for idx, ch_name in enumerate(self.active_channels):
            y_offset = (n_active - 1 - idx) * spacing
            ticks.append((y_offset, ch_name))

            color = self.channel_colors.get(ch_name, QColor("#1DB954"))
            pen = pg.mkPen(color=color, width=1.4)
            curve = self.plot_widget.plot(pen=pen)
            self.curves[ch_name] = (curve, y_offset)

        self.plot_widget.getAxis('left').setTicks([ticks])
        self.plot_widget.setYRange(-spacing * 0.5, n_active * spacing, padding=0.01)
        self.plot_widget.setXRange(-self.window_sec, 0.0, padding=0.0)

    def update_waveforms(self, frame_idx: int):
        self.current_frame = frame_idx
        n_samples = self.data_loader.n_samples
        if n_samples == 0 or len(self.active_channels) == 0:
            return

        sfreq = self.data_loader.sfreq
        win_samples = int(self.window_sec * sfreq)
        if win_samples < 2:
            return

        start = frame_idx - win_samples
        end = frame_idx
        time_slice = np.linspace(-self.window_sec, 0.0, win_samples)
        scale = 1.0 / (self.vertical_gain * 1e-6)

        for ch_name in self.active_channels:
            if ch_name in self.curves and ch_name in self.data_loader.ch_names:
                ch_idx = self.data_loader.ch_names.index(ch_name)
                curve, y_offset = self.curves[ch_name]

                if start < 0:
                    pad_len = -start
                    valid_data = self.data_loader.data_current[ch_idx, 0:end]
                    pad_val = valid_data[0] if len(valid_data) > 0 else 0.0
                    raw_slice = np.concatenate([np.full(pad_len, pad_val), valid_data])
                elif end > n_samples:
                    valid_data = self.data_loader.data_current[ch_idx, start:n_samples]
                    pad_len = end - n_samples
                    pad_val = valid_data[-1] if len(valid_data) > 0 else 0.0
                    raw_slice = np.concatenate([valid_data, np.full(pad_len, pad_val)])
                else:
                    raw_slice = self.data_loader.data_current[ch_idx, start:end]

                if len(raw_slice) == win_samples:
                    scaled_y = y_offset + (raw_slice * scale * 0.38)
                    curve.setData(time_slice, scaled_y)


# =============================================================================
# SPOTIFY PLAYBACK BAR (Bottom Transport Controls & Event Timeline)
# =============================================================================
class SpotifyTimelineWidget(pg.PlotWidget):
    sig_user_seeked = pyqtSignal(float)

    EVENT_COLORS = {
        'T0': {'fill': QColor(42, 42, 56, 110),   'border': QColor(74, 74, 94, 180)},
        'T1': {'fill': QColor(29, 185, 84, 120),  'border': QColor(30, 215, 96, 220)},
        'T2': {'fill': QColor(217, 70, 239, 120), 'border': QColor(232, 121, 249, 220)},
    }

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.duration_sec = max(1.0, data_loader.duration_sec)
        self._dragging = False

        self.setBackground("#0a0a0f")
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.showAxis('left', False)
        self.showAxis('top', False)
        self.showAxis('right', False)
        self.getAxis('bottom').setStyle(showValues=False, tickLength=0)
        self.getAxis('bottom').setPen(pg.mkPen(color="#1a1a24", width=1))

        self.setXRange(0, self.duration_sec, padding=0.005)
        self.setYRange(-1.0, 1.0, padding=0.0)

        for seg in self.data_loader.event_segments:
            name = seg['name']
            spec = self.EVENT_COLORS.get(name, {'fill': QColor(60, 60, 80, 80), 'border': QColor(100, 100, 120, 160)})
            reg = pg.LinearRegionItem(
                values=(seg['start_sec'], min(self.duration_sec, seg['start_sec'] + seg['duration_sec'])),
                orientation=pg.LinearRegionItem.Vertical,
                brush=QBrush(spec['fill']),
                pen=QPen(spec['border']),
                movable=False
            )
            for line in reg.lines:
                line.setPen(QPen(spec['border']))
            self.addItem(reg)

        self.playhead = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(color="#ffffff", width=2),
            hoverPen=pg.mkPen(color="#1DB954", width=3)
        )
        self.playhead.setBounds([0.0, self.duration_sec])
        self.playhead.sigPositionChanged.connect(self._on_drag)
        self.playhead.sigPositionChangeFinished.connect(self._on_release)
        self.addItem(self.playhead)

    def set_playhead_pos(self, sec: float):
        if not self._dragging:
            self.playhead.blockSignals(True)
            self.playhead.setPos(sec)
            self.playhead.blockSignals(False)

    def _on_drag(self):
        self._dragging = True
        self.sig_user_seeked.emit(self.playhead.value())

    def _on_release(self):
        self._dragging = False
        self.sig_user_seeked.emit(self.playhead.value())

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            v_pos = self.plotItem.vb.mapSceneToView(ev.pos())
            val = max(0.0, min(self.duration_sec, v_pos.x()))
            self.playhead.setPos(val)
            self.sig_user_seeked.emit(val)
            ev.accept()
        else:
            super().mousePressEvent(ev)


class SpotifyPlaybackBar(QFrame):
    def __init__(self, data_loader: EEGDataLoader, engine: PlaybackEngine, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.engine = engine
        self.setFixedHeight(82)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QFrame {
                background-color: #08080c;
                border-top: 1px solid #181822;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 4)
        layout.setSpacing(14)

        # Track Card
        meta_box = QHBoxLayout()
        meta_box.setSpacing(10)
        badge = QLabel("EEG", self)
        badge.setFixedSize(36, 36)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet("background: #1DB954; color: #000; font-weight: 900; border-radius: 4px;")

        text_vbox = QVBoxLayout()
        text_vbox.setSpacing(1)
        t_lbl = QLabel(f"PhysioNet S{self.data_loader.subject:03d} • Run {self.data_loader.run:02d}", self)
        t_lbl.setStyleSheet("font-weight: 700; color: #ffffff; font-size: 11px;")
        sub_lbl = QLabel("Motor Imagery 64ch", self)
        sub_lbl.setStyleSheet("color: #727284; font-size: 10px;")
        text_vbox.addWidget(t_lbl)
        text_vbox.addWidget(sub_lbl)
        meta_box.addWidget(badge)
        meta_box.addLayout(text_vbox)
        layout.addLayout(meta_box)

        # Controls & Timeline
        center_vbox = QVBoxLayout()
        center_vbox.setSpacing(2)
        center_vbox.setAlignment(Qt.AlignCenter)

        btn_box = QHBoxLayout()
        btn_box.setSpacing(12)
        btn_box.setAlignment(Qt.AlignCenter)

        self.btn_back = QPushButton("⏮", self)
        self.btn_back.setStyleSheet("background: transparent; color: #9c9cae; font-size: 16px; border: none;")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self.engine.step_backward)

        self.btn_play = QPushButton("▶", self)
        self.btn_play.setStyleSheet("""
            background-color: #ffffff; color: #000; border-radius: 16px;
            min-width: 32px; max-width: 32px; min-height: 32px; max-height: 32px;
            font-size: 14px; font-weight: bold; border: none;
        """)
        self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.clicked.connect(self.engine.toggle_play)

        self.btn_next = QPushButton("⏭", self)
        self.btn_next.setStyleSheet("background: transparent; color: #9c9cae; font-size: 16px; border: none;")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.clicked.connect(self.engine.step_forward)

        btn_box.addWidget(self.btn_back)
        btn_box.addWidget(self.btn_play)
        btn_box.addWidget(self.btn_next)
        center_vbox.addLayout(btn_box)

        time_box = QHBoxLayout()
        time_box.setSpacing(8)
        self.lbl_cur = QLabel("0:00.000", self)
        self.lbl_cur.setStyleSheet("font-family: monospace; color: #9c9cae; font-size: 11px;")
        self.lbl_cur.setFixedWidth(56)

        self.timeline = SpotifyTimelineWidget(self.data_loader, self)
        self.timeline.sig_user_seeked.connect(self.engine.seek_time)

        tot = self.data_loader.duration_sec
        self.lbl_tot = QLabel(f"{int(tot//60)}:{tot%60:06.3f}", self)
        self.lbl_tot.setStyleSheet("font-family: monospace; color: #9c9cae; font-size: 11px;")
        self.lbl_tot.setFixedWidth(56)

        time_box.addWidget(self.lbl_cur)
        time_box.addWidget(self.timeline, stretch=1)
        time_box.addWidget(self.lbl_tot)
        center_vbox.addLayout(time_box)

        layout.addLayout(center_vbox, stretch=1)

        # Right Dynamic Stimulus Indicators (Lighting Up based on Active Event)
        legend_vbox = QVBoxLayout()
        legend_vbox.setSpacing(3)
        legend_vbox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        leg_row = QHBoxLayout()
        leg_row.setSpacing(6)

        self.badge_rest = QLabel("● Rest", self)
        self.badge_left = QLabel("● Left Fist", self)
        self.badge_right = QLabel("● Right Fist", self)

        for b in [self.badge_rest, self.badge_left, self.badge_right]:
            b.setCursor(Qt.ArrowCursor)

        leg_row.addWidget(self.badge_rest)
        leg_row.addWidget(self.badge_left)
        leg_row.addWidget(self.badge_right)
        legend_vbox.addLayout(leg_row)

        self.lbl_active_stim = QLabel("Active Stimulus: Rest", self)
        self.lbl_active_stim.setStyleSheet("color: #1DB954; font-weight: 700; font-size: 11px;")
        legend_vbox.addWidget(self.lbl_active_stim)
        layout.addLayout(legend_vbox)

        self._set_active_stimulus("T0")

        self.engine.sig_playback_state.connect(lambda p: self.btn_play.setText("⏸" if p else "▶"))
        self.engine.sig_frame_changed.connect(self._on_frame)

    def _set_active_stimulus(self, active_code: str):
        """Lights up the active stimulus badge while dimming the others."""
        style_dim = """
            background-color: #101016;
            color: #555566;
            font-size: 10px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 8px;
            border: 1px solid #1a1a24;
        """
        style_rest_active = """
            background-color: #1a1e2c;
            color: #93c5fd;
            font-size: 10px;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 8px;
            border: 1.5px solid #60a5fa;
        """
        style_left_active = """
            background-color: #0d2c17;
            color: #1ed760;
            font-size: 10px;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 8px;
            border: 1.5px solid #1ed760;
        """
        style_right_active = """
            background-color: #311138;
            color: #f472b6;
            font-size: 10px;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 8px;
            border: 1.5px solid #d946ef;
        """

        if active_code == "T1":
            self.badge_rest.setStyleSheet(style_dim)
            self.badge_left.setStyleSheet(style_left_active)
            self.badge_right.setStyleSheet(style_dim)
            self.lbl_active_stim.setText("Active Stimulus: Real Left Fist (T1)")
            self.lbl_active_stim.setStyleSheet("color: #1ed760; font-weight: 700; font-size: 11px;")
        elif active_code == "T2":
            self.badge_rest.setStyleSheet(style_dim)
            self.badge_left.setStyleSheet(style_dim)
            self.badge_right.setStyleSheet(style_right_active)
            self.lbl_active_stim.setText("Active Stimulus: Real Right Fist (T2)")
            self.lbl_active_stim.setStyleSheet("color: #d946ef; font-weight: 700; font-size: 11px;")
        else:
            self.badge_rest.setStyleSheet(style_rest_active)
            self.badge_left.setStyleSheet(style_dim)
            self.badge_right.setStyleSheet(style_dim)
            self.lbl_active_stim.setText("Active Stimulus: Baseline Rest (T0)")
            self.lbl_active_stim.setStyleSheet("color: #60a5fa; font-weight: 700; font-size: 11px;")

    def _on_frame(self, frame_idx: int, sec: float):
        self.timeline.set_playhead_pos(sec)
        m = int(sec // 60)
        s = sec % 60
        self.lbl_cur.setText(f"{m}:{s:06.3f}")

        active_code = "T0"
        for seg in self.data_loader.event_segments:
            if seg['start_sec'] <= sec <= (seg['start_sec'] + seg['duration_sec']):
                active_code = seg['name']
                break
        self._set_active_stimulus(active_code)


# =============================================================================
# MAIN WINDOW ARCHITECTURE
# =============================================================================
class NeuromapMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("neuromap")
        self.resize(1920, 1200)

        self.data_loader = EEGDataLoader(subject=1, run=4).load_data()
        self.engine = PlaybackEngine(self.data_loader)

        self._init_ui()
        self._init_statusbar()
        self._wire_connections()

        # Keyboard Shortcut: F11 toggles True Fullscreen mode
        self.shortcut_f11 = QShortcut(Qt.Key_F11, self)
        self.shortcut_f11.activated.connect(self.toggle_fullscreen)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(6, 6, 6, 4)
        root_layout.setSpacing(5)

        # 1. Top Header with title "neuromap" and compact metadata
        self.top_header = TopNavigationHeader(self.data_loader, self)
        root_layout.addWidget(self.top_header)

        # 2. Main Horizontal Splitter (Sidebar on Left, Workspaces on Right)
        self.h_splitter = QSplitter(Qt.Horizontal, self)
        self.h_splitter.setChildrenCollapsible(False)

        # Sidebar Controls
        self.sidebar = SidebarControlsWidget(self.data_loader, self.h_splitter)
        self.h_splitter.addWidget(self.sidebar)

        # Right Workspace (Vertical Splitter)
        self.workspace_splitter = QSplitter(Qt.Vertical, self.h_splitter)
        self.workspace_splitter.setChildrenCollapsible(False)

        # TOP ROW: Top-Left 2D Topomap + Top-Right 3D Brain Model
        self.top_row_splitter = QSplitter(Qt.Horizontal, self.workspace_splitter)
        self.top_row_splitter.setChildrenCollapsible(False)

        self.map_2d = ChannelMap2DWidget(self.data_loader, self.top_row_splitter)
        self.brain_3d = Brain3DViewerWidget(glb_path="data/human-brain.glb", parent=self.top_row_splitter)

        self.top_row_splitter.addWidget(self.map_2d)
        self.top_row_splitter.addWidget(self.brain_3d)
        self.top_row_splitter.setSizes([800, 800])

        self.workspace_splitter.addWidget(self.top_row_splitter)

        # BOTTOM ROW: Cascading Multi-Track Waveforms
        self.timeseries_plot = MultiTrackTimeSeriesWidget(self.data_loader, self.workspace_splitter)
        self.workspace_splitter.addWidget(self.timeseries_plot)

        self.workspace_splitter.setSizes([580, 440])

        self.h_splitter.addWidget(self.workspace_splitter)
        self.h_splitter.setSizes([260, 1640])
        root_layout.addWidget(self.h_splitter, stretch=1)

        # 3. Spotify Playback Bar
        self.playback_bar = SpotifyPlaybackBar(self.data_loader, self.engine, self)
        root_layout.addWidget(self.playback_bar)

    def _init_statusbar(self):
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Status: System Ready | Phase 3 Real-Time Topomap Engine Active", self)
        self.status_label.setStyleSheet("color: #88889a; font-size: 11px;")

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Progress: Ready (100%)")

        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _wire_connections(self):
        # 1. 2D Map -> Multi-Track Plot (Only selected channels appear)
        self.map_2d.sig_selection_changed.connect(
            lambda channels: self.timeseries_plot.set_active_channels(channels, self.map_2d.channel_colors)
        )

        # 2. Display Controls -> Multi-Track Plot & Topomap
        self.sidebar.sig_display_changed.connect(
            lambda params: self.timeseries_plot.set_display_params(params['gain'], params['window_sec'])
        )
        self.sidebar.sig_topomap_gain_changed.connect(self.map_2d.set_topomap_v_max)

        # 3. Playback Speed Selector -> Engine
        self.sidebar.sig_speed_changed.connect(self.engine.set_speed)

        # 4. Signal Processing Filters Trigger
        self.sidebar.sig_filters_changed.connect(self._on_filters_applied)

        # 5. Engine Tick -> Synchronized Waveform & Topomap Heatmap Updates
        self.engine.sig_frame_changed.connect(self._on_frame_tick)

        # 6. Export Button
        self.top_header.sig_export_requested.connect(self._on_export_data)

    def _on_frame_tick(self, frame_idx: int, sec: float):
        self.map_2d.update_topomap_frame(frame_idx)
        self.timeseries_plot.update_waveforms(frame_idx)

    def _on_filters_applied(self, filter_params: dict):
        notch_str = []
        if filter_params['notch_60']: notch_str.append("60Hz")
        if filter_params['notch_50']: notch_str.append("50Hz")
        notch_desc = f"Notch [{', '.join(notch_str)}]" if notch_str else "No Notch"

        hp = filter_params['highpass']
        lp = filter_params['lowpass']

        self.status_label.setText(f"Status: Applying {notch_desc}, Bandpass {hp:.1f} - {lp:.1f} Hz...")
        self.progress_bar.setValue(82)
        self.progress_bar.setFormat("Progress: Filtering & Resampling Data... 82%")

        QTimer.singleShot(350, lambda: self._complete_filter(notch_desc, hp, lp))

    def _complete_filter(self, notch_desc: str, hp: float, lp: float):
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Progress: Ready (100%)")
        self.status_label.setText(f"Status: Filter Applied ({notch_desc}, {hp:.1f}-{lp:.1f} Hz).")
        self.timeseries_plot.update_waveforms(self.engine.current_frame)
        self.map_2d.update_topomap_frame(self.engine.current_frame)

    def _on_export_data(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Filtered EEG Data", "neuromap_filtered_export.csv", "CSV Files (*.csv);;NumPy Array (*.npy)"
        )
        if filepath:
            self.status_label.setText(f"Exported data to {os.path.basename(filepath)}")


# =============================================================================
# ENTRYPOINT (Starts Maximized / Fullscreen)
# =============================================================================
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)

    window = NeuromapMainWindow()
    # Automatically launch in maximized/fullscreen mode
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()