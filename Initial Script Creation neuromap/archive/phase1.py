#!/usr/bin/env python3
"""
Closed-Loop EEG Visualization Dashboard - Phase 1: Environment & Layout Scaffold
Standards:
- PyQt5 High-DPI Enabled
- PyQtGraph Anti-aliasing Enabled
- Strict OOP modular architecture
- 1920x1200 Display Geometry
- QSplitter top-to-bottom and left-to-right hierarchy
- Modern Dark QSS blending Spotify, OpenBCI, and Suite2p aesthetics
"""

import sys
import os
import numpy as np

# PyQt5 Imports
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QStatusBar,
    QFrame,
    QSlider,
    QSizePolicy,
)

# Optional / External Science & Visualization Libraries
try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background="#0d0d11", foreground="#c8c8cf")
except ImportError:
    pg = None

try:
    import mne
    from mne.datasets import eegbci
except ImportError:
    mne = None

try:
    import pyvista as pv
    import pyvistaqt as pvqt
except ImportError:
    pv = None
    pvqt = None


# =============================================================================
# QSS THEME STYLING
# Blending Spotify controls (#1DB954 accent, sleek black playback bar),
# OpenBCI modular panels & channel badges, and Suite2p high-density dark scientific theme.
# =============================================================================
DARK_QSS = """
/* Base Application Style */
QMainWindow {
    background-color: #0b0b0e;
    color: #e0e0e6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
}

QWidget {
    background-color: transparent;
    color: #dedee4;
}

/* Panel Containers (Suite2p & OpenBCI inspired) */
QFrame.panel-card {
    background-color: #131318;
    border: 1px solid #23232c;
    border-radius: 8px;
}

QFrame.panel-card:hover {
    border: 1px solid #2f2f3d;
}

/* Header Bars inside Panels */
QFrame.panel-header {
    background-color: #191921;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    border-bottom: 1px solid #24242f;
    padding: 6px 12px;
}

QLabel.panel-title {
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
    color: #f0f0f5;
    text-transform: uppercase;
}

QLabel.panel-badge {
    background-color: #242433;
    color: #1DB954;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    border: 1px solid #1ed76033;
}

/* QSplitter Customization */
QSplitter::handle {
    background-color: #17171f;
    margin: 1px;
}

QSplitter::handle:horizontal {
    width: 5px;
}

QSplitter::handle:vertical {
    height: 5px;
}

QSplitter::handle:hover {
    background-color: #1DB954;
}

/* Spotify Playback Bar (Bottom Section) */
QFrame#spotify-playback-bar {
    background-color: #0f0f13;
    border-top: 1px solid #22222b;
    padding: 10px 20px;
}

QLabel.track-title {
    font-weight: 600;
    font-size: 13px;
    color: #ffffff;
}

QLabel.track-subtitle {
    font-size: 11px;
    color: #92929e;
}

QLabel.time-label {
    font-family: "SF Mono", "Consolas", "Courier New", monospace;
    font-size: 11px;
    color: #8c8c99;
}

/* Spotify Control Buttons */
QPushButton.spotify-btn-subtle {
    background: transparent;
    border: none;
    color: #a0a0b0;
    font-size: 16px;
    padding: 6px;
    border-radius: 16px;
}

QPushButton.spotify-btn-subtle:hover {
    color: #ffffff;
    background-color: #21212b;
}

QPushButton.spotify-play-btn {
    background-color: #ffffff;
    color: #000000;
    border: none;
    border-radius: 20px;
    min-width: 40px;
    max-width: 40px;
    min-height: 40px;
    max-height: 40px;
    font-size: 18px;
    font-weight: bold;
}

QPushButton.spotify-play-btn:hover {
    background-color: #1DB954;
    color: #ffffff;
}

/* Spotify Style Timeline Slider */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #2f2f3c;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #1DB954;
    border-radius: 2px;
}

QSlider::sub-page:horizontal:hover {
    background: #1ed760;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: none;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QSlider::handle:horizontal:hover {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #ffffff;
}

/* Status Bar */
QStatusBar {
    background-color: #0a0a0d;
    color: #8e8e99;
    border-top: 1px solid #1c1c24;
    font-size: 11px;
    padding: 2px 10px;
}

QStatusBar::item {
    border: none;
}
"""


# =============================================================================
# DATA LAYER (MNE PhysioNet Motor Movement/Imagery Loader)
# =============================================================================
class EEGDataLoader:
    """
    Handles robust loading and preprocessing of MNE PhysioNet EEG data.
    Subject 1, Run 4 (Motor execution: left vs right fist), 1-40 Hz bandpass filter.
    Includes graceful offline fallback generation for self-contained testing.
    """
    def __init__(self, subject: int = 1, run: int = 4, l_freq: float = 1.0, h_freq: float = 40.0):
        self.subject = subject
        self.run = run
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.raw = None
        self.events = None
        self.event_id = None
        self.ch_names = []
        self.times = np.array([])
        self.data = np.array([[]])
        self.sfreq = 160.0

    def load_and_preprocess(self):
        """Loads dataset and applies bandpass filtering with try/except error protection."""
        if mne is None:
            print("[WARN] MNE is not installed. Initializing synthetic EEG fallback dataset.")
            self._create_synthetic_dataset("MNE library unavailable")
            return self

        try:
            print(f"[INFO] Fetching PhysioNet Subject {self.subject}, Run {self.run}...")
            edf_paths = eegbci.load_data(subjects=[self.subject], runs=[self.run], update_path=False)
            if not edf_paths:
                raise FileNotFoundError("PhysioNet data files could not be retrieved.")

            edf_path = edf_paths[0]
            print(f"[INFO] Reading Raw EDF from {edf_path}...")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

            # Strip trailing dots from channel names ('Fc5.' -> 'FC5')
            eegbci.standardize(raw)
            # Set standard 10-05 montage
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')

            # Apply 1.0 - 40.0 Hz bandpass filter
            print(f"[INFO] Applying {self.l_freq}-{self.h_freq} Hz bandpass filter...")
            raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin', verbose=False)

            # Extract events and annotations
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            self.raw = raw
            self.events = events
            self.event_id = event_id
            self.ch_names = raw.ch_names
            self.sfreq = raw.info['sfreq']
            self.times = raw.times
            self.data = raw.get_data()  # Shape: (n_channels, n_times)
            print(f"[SUCCESS] Loaded {len(self.ch_names)} channels, {self.data.shape[1]} samples at {self.sfreq} Hz.")
            return self

        except Exception as exc:
            print(f"[WARN] Error loading PhysioNet dataset ({exc}). Falling back to synthetic 64-channel dataset.")
            self._create_synthetic_dataset(str(exc))
            return self

    def _create_synthetic_dataset(self, reason: str):
        """Generates synthetic 64-channel 10-20 EEG stream for offline scaffold verification."""
        self.sfreq = 160.0
        n_samples = int(self.sfreq * 60)  # 60 seconds
        self.times = np.linspace(0, 60, n_samples)
        standard_64 = [
            'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
        ]
        self.ch_names = standard_64
        n_ch = len(standard_64)

        t = self.times
        rng = np.random.default_rng(42)
        base_noise = rng.normal(0, 5e-6, (n_ch, n_samples))
        alpha = 15e-6 * np.sin(2 * np.pi * 10 * t)
        beta = 8e-6 * np.sin(2 * np.pi * 20 * t)
        theta = 12e-6 * np.sin(2 * np.pi * 6 * t)

        synthetic_data = base_noise + alpha + beta + theta
        self.data = synthetic_data
        self.event_id = {'T0': 1, 'T1': 2, 'T2': 3}
        self.events = np.array([
            [int(5 * self.sfreq), 0, 1],
            [int(15 * self.sfreq), 0, 2],
            [int(30 * self.sfreq), 0, 3],
            [int(45 * self.sfreq), 0, 2]
        ])
        print(f"[INFO] Synthetic EEG dataset ready ({n_ch} channels, {n_samples} samples, reason: {reason}).")


# =============================================================================
# MODULAR UI COMPONENT CLASSES
# =============================================================================

class Topomap2DWidget(QFrame):
    """
    Top-Left: 2D Scalp Topomap scaffold.
    Aesthetic: OpenBCI inspired circular head schematic with orientation markers.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topomap-2d-panel")
        self.setProperty("class", "panel-card")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel Header
        header = QFrame(self)
        header.setProperty("class", "panel-header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)

        title = QLabel("2D Scalp Topomap", header)
        title.setProperty("class", "panel-title")
        badge = QLabel("OpenBCI Overhead View", header)
        badge.setProperty("class", "panel-badge")

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(badge)
        layout.addWidget(header)

        # Content Area
        self.content_area = QFrame(self)
        content_layout = QVBoxLayout(self.content_area)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setAlignment(Qt.AlignCenter)

        self.placeholder_lbl = QLabel(
            "⚡ 2D Electrode Scalp Projection (Phase 3 Engine)\n"
            "• Interactive multi-select nodes (OpenBCI color hierarchy)\n"
            "• Real-time interpolation & voltage colormapping",
            self.content_area
        )
        self.placeholder_lbl.setAlignment(Qt.AlignCenter)
        self.placeholder_lbl.setStyleSheet("color: #727282; font-size: 13px; line-height: 1.6;")
        content_layout.addWidget(self.placeholder_lbl)

        layout.addWidget(self.content_area, stretch=1)


class Head3DWidget(QFrame):
    """
    Top-Right: 3D Anatomical Scalp / Mesh Viewport scaffold.
    Aesthetic: Suite2p dark frame + PyVista 3D viewport placeholder.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("head-3d-panel")
        self.setProperty("class", "panel-card")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel Header
        header = QFrame(self)
        header.setProperty("class", "panel-header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)

        title = QLabel("3D Scalp Interactor", header)
        title.setProperty("class", "panel-title")
        badge = QLabel("PyVista 10-05 Space", header)
        badge.setProperty("class", "panel-badge")

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(badge)
        layout.addWidget(header)

        # Content Area
        self.content_area = QFrame(self)
        content_layout = QVBoxLayout(self.content_area)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setAlignment(Qt.AlignCenter)

        self.placeholder_lbl = QLabel(
            "🧠 3D Human Head Mesh & Scalp Interactor (Phase 4 Engine)\n"
            "• male_head_base_mesh.glb alignment & decimation\n"
            "• Dynamic 3D electrode sphere voltage colormapping",
            self.content_area
        )
        self.placeholder_lbl.setAlignment(Qt.AlignCenter)
        self.placeholder_lbl.setStyleSheet("color: #727282; font-size: 13px; line-height: 1.6;")
        content_layout.addWidget(self.placeholder_lbl)

        layout.addWidget(self.content_area, stretch=1)


class TimeSeriesWidget(QFrame):
    """
    Middle Section: Multi-Channel Scrolling EEG Time Series scaffold.
    Aesthetic: OpenBCI channel pill labels + Suite2p high-density time-series plots.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("timeseries-panel")
        self.setProperty("class", "panel-card")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel Header
        header = QFrame(self)
        header.setProperty("class", "panel-header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)

        title = QLabel("Selected Channel Time-Series", header)
        title.setProperty("class", "panel-title")

        filter_info = QLabel("Bandpass: 1.0 - 40.0 Hz  |  PhysioNet S001 R04", header)
        filter_info.setStyleSheet("color: #888899; font-size: 11px;")

        badge = QLabel("Multi-Select Active", header)
        badge.setProperty("class", "panel-badge")

        header_layout.addWidget(title)
        header_layout.addSpacing(16)
        header_layout.addWidget(filter_info)
        header_layout.addStretch()
        header_layout.addWidget(badge)
        layout.addWidget(header)

        # Content Area
        self.content_area = QFrame(self)
        content_layout = QVBoxLayout(self.content_area)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setAlignment(Qt.AlignCenter)

        self.placeholder_lbl = QLabel(
            "📈 Multi-Channel Scrolling Voltage Traces (Phase 3 Engine)\n"
            "• Dynamic channel waveforms linked to selected 2D/3D scalp nodes\n"
            "• Node-matching RGB trace styling and microvolt RMS readouts",
            self.content_area
        )
        self.placeholder_lbl.setAlignment(Qt.AlignCenter)
        self.placeholder_lbl.setStyleSheet("color: #727282; font-size: 13px; line-height: 1.6;")
        content_layout.addWidget(self.placeholder_lbl)

        layout.addWidget(self.content_area, stretch=1)


class SpotifyPlaybackWidget(QFrame):
    """
    Bottom Section: Modern Spotify-style EEG Playback Control Bar.
    Aesthetic: Pitch black background (#0f0f13), Spotify green accent (#1DB954),
    Step Back (⏮), Play/Pause (⏯), Step Forward (⏭), draggable progress timeline.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("spotify-playback-bar")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(95)
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 8, 20, 8)
        main_layout.setSpacing(20)

        # --- Left: Dataset & Run Metadata ---
        left_box = QFrame(self)
        left_layout = QHBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Album / Dataset Icon Badge
        art_badge = QLabel("EEG", left_box)
        art_badge.setFixedSize(48, 48)
        art_badge.setAlignment(Qt.AlignCenter)
        art_badge.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1db954, stop:1 #12542a);
            color: #ffffff;
            font-weight: 900;
            font-size: 14px;
            border-radius: 6px;
        """)

        meta_layout = QVBoxLayout()
        meta_layout.setAlignment(Qt.AlignVCenter)
        meta_layout.setSpacing(2)
        title_lbl = QLabel("PhysioNet Motor Imagery S001", left_box)
        title_lbl.setProperty("class", "track-title")
        subtitle_lbl = QLabel("Run 4: Left vs Right Fist • 1-40Hz", left_box)
        subtitle_lbl.setProperty("class", "track-subtitle")

        meta_layout.addWidget(title_lbl)
        meta_layout.addWidget(subtitle_lbl)

        left_layout.addWidget(art_badge)
        left_layout.addLayout(meta_layout)
        left_box.setFixedWidth(280)
        main_layout.addWidget(left_box)

        # --- Center: Spotify Transport Controls & Timeline ---
        center_box = QFrame(self)
        center_layout = QVBoxLayout(center_box)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)
        center_layout.setAlignment(Qt.AlignCenter)

        # Upper row: Transport buttons
        controls_layout = QHBoxLayout()
        controls_layout.setAlignment(Qt.AlignCenter)
        controls_layout.setSpacing(16)

        self.btn_step_back = QPushButton("⏮", center_box)
        self.btn_step_back.setProperty("class", "spotify-btn-subtle")
        self.btn_step_back.setToolTip("Step Back Frame")

        self.btn_play_pause = QPushButton("▶", center_box)
        self.btn_play_pause.setProperty("class", "spotify-play-btn")
        self.btn_play_pause.setToolTip("Play / Pause (30ms loop)")

        self.btn_step_forward = QPushButton("⏭", center_box)
        self.btn_step_forward.setProperty("class", "spotify-btn-subtle")
        self.btn_step_forward.setToolTip("Step Forward Frame")

        controls_layout.addWidget(self.btn_step_back)
        controls_layout.addWidget(self.btn_play_pause)
        controls_layout.addWidget(self.btn_step_forward)

        center_layout.addLayout(controls_layout)

        # Lower row: Time labels and progress timeline slider
        timeline_layout = QHBoxLayout()
        timeline_layout.setSpacing(10)

        self.time_current = QLabel("0:00.000", center_box)
        self.time_current.setProperty("class", "time-label")

        self.timeline_slider = QSlider(Qt.Horizontal, center_box)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setCursor(Qt.PointingHandCursor)

        self.time_total = QLabel("1:00.000", center_box)
        self.time_total.setProperty("class", "time-label")

        timeline_layout.addWidget(self.time_current)
        timeline_layout.addWidget(self.timeline_slider)
        timeline_layout.addWidget(self.time_total)

        center_layout.addLayout(timeline_layout)
        main_layout.addWidget(center_box, stretch=1)

        # --- Right: Speed & Event Legend Status ---
        right_box = QFrame(self)
        right_layout = QHBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        rate_badge = QLabel("30 FPS / 30ms", right_box)
        rate_badge.setProperty("class", "panel-badge")
        right_layout.addWidget(rate_badge)
        right_box.setFixedWidth(200)

        main_layout.addWidget(right_box)


# =============================================================================
# MAIN WINDOW SCAFFOLD
# =============================================================================
class EEGMainWindow(QMainWindow):
    """
    Main Window container for the EEG visualization dashboard.
    Geometry: 1920x1200
    Layout Structure (via QSplitter):
      - Main Vertical Splitter:
          - Top Row (Horizontal Splitter):
              - Left: 2D Scalp Topomap
              - Right: 3D Anatomical Scalp/Mesh Interactor
          - Middle: Multi-Channel Scrolling Voltage Traces
          - Bottom: Spotify Playback Control Bar
    """
    def __init__(self, data_loader: EEGDataLoader):
        super().__init__()
        self.data_loader = data_loader
        self.setWindowTitle("NeuroInsight • Closed-Loop EEG Dashboard (Phase 1 Scaffold)")
        self.resize(1920, 1200)

        self._init_ui()
        self._init_statusbar()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 8)
        root_layout.setSpacing(8)

        # Main Vertical Splitter
        self.main_vertical_splitter = QSplitter(Qt.Vertical, central_widget)
        self.main_vertical_splitter.setChildrenCollapsible(False)

        # 1. Top Row: Horizontal Splitter (2D Topomap + 3D Head)
        self.top_horizontal_splitter = QSplitter(Qt.Horizontal)
        self.top_horizontal_splitter.setChildrenCollapsible(False)

        self.topomap_widget = Topomap2DWidget(self.top_horizontal_splitter)
        self.head_3d_widget = Head3DWidget(self.top_horizontal_splitter)

        self.top_horizontal_splitter.addWidget(self.topomap_widget)
        self.top_horizontal_splitter.addWidget(self.head_3d_widget)
        self.top_horizontal_splitter.setSizes([960, 960])

        self.main_vertical_splitter.addWidget(self.top_horizontal_splitter)

        # 2. Middle Section: Time-Series Scrolling Traces
        self.timeseries_widget = TimeSeriesWidget(self.main_vertical_splitter)
        self.main_vertical_splitter.addWidget(self.timeseries_widget)

        # 3. Bottom Section: Spotify Playback Bar
        self.playback_widget = SpotifyPlaybackWidget(self.main_vertical_splitter)
        self.main_vertical_splitter.addWidget(self.playback_widget)

        # Proportional vertical sizing: Top (550px), Middle (450px), Bottom (100px)
        self.main_vertical_splitter.setSizes([550, 450, 100])

        root_layout.addWidget(self.main_vertical_splitter)

    def _init_statusbar(self):
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        ch_count = len(self.data_loader.ch_names)
        s_count = self.data_loader.data.shape[1] if self.data_loader.data.size > 0 else 0
        dur_sec = s_count / self.data_loader.sfreq if self.data_loader.sfreq > 0 else 0

        self.status_bar.showMessage(
            f"Ready | Loaded Channels: {ch_count} | Samples: {s_count} ({dur_sec:.1f}s @ {self.data_loader.sfreq:.0f}Hz) | High-DPI & Anti-aliasing active."
        )


# =============================================================================
# APPLICATION ENTRYPOINT
# =============================================================================
def main():
    # 1. Enable High-DPI Scaling & Crisp Pixmaps (Crucial for 1920x1200 high-res displays)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)

    # 2. Load and Preprocess MNE Dataset
    data_loader = EEGDataLoader(subject=1, run=4, l_freq=1.0, h_freq=40.0)
    data_loader.load_and_preprocess()

    # 3. Instantiate & Display Main Window
    window = EEGMainWindow(data_loader)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()