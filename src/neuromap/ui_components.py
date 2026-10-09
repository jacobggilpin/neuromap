"""
neuromap - Core UI Components
Includes:
- WaveformsWidget: Cascading multi-channel voltage traces (Phase 4.03 sliding window)
- SpotifyTimelineWidget: Interactive timeline with GFP envelope mini-map & prominent playhead
- SpotifyPlaybackBar: Transport bar with frame-by-frame controls (⏮, ▶, ⏭)
- HeaderWidget: Branding & CSV export
- ControlSidebarWidget: Collapsible panel for speed, gain, filters, and FastICA controls
"""
from typing import List, Dict, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QBrush
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel,
    QPushButton, QSlider, QLineEdit, QCheckBox, QComboBox
)
import pyqtgraph as pg

try:
    from .config import EVENT_COLOR_MAP
    from .engine import EEGDataLoader, PlaybackEngine
except ImportError:
    from config import EVENT_COLOR_MAP
    from engine import EEGDataLoader, PlaybackEngine

class WaveformsWidget(QWidget):
    """
    Cascading Multi-Channel Raw EEG Waveforms:
    - Renders ONLY user-selected channels (starts with 0 active on launch).
    - Phase 4.03 standard sliding window slicing via EEGDataLoader.get_window_data.
    - Zero font metric recalculation in render loop (avoids GraphicsView jitter).
    - Precomputed timeline X-axis buffer for 60+ FPS performance.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.active_channels: List[str] = []
        self.gain = 1.0
        self.window_sec = 4.0
        self.cached_t_axis: Optional[np.ndarray] = None
        self._cached_n_pts: int = 0

        self._init_ui()

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

        self.title_lbl = QLabel("<span style='color: #00FFA3;'>●</span> RAW EEG WAVEFORMS (0 CHANNELS SELECTED)")
        self.title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(self.title_lbl)

        h_layout.addStretch()

        self.info_lbl = QLabel("Select channels in Topomap to observe raw traces")
        self.info_lbl.setStyleSheet("color: #7A889B; font-size: 10px;")
        h_layout.addWidget(self.info_lbl)

        layout.addWidget(header)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideAxis('left')

        bottom_axis = self.plot_widget.getAxis('bottom')
        bottom_axis.setPen(pg.mkPen('#252D3C', width=1))
        bottom_axis.setTextPen(pg.mkPen('#8C9BAE'))

        self.curve_items: Dict[str, pg.PlotDataItem] = {}
        self.channel_labels: Dict[str, pg.TextItem] = {}

        layout.addWidget(self.plot_widget)

    def set_active_channels(self, channels: List[str]):
        self.active_channels = list(channels)
        self.plot_widget.clear()
        self.curve_items.clear()
        self.channel_labels.clear()

        n_active = len(self.active_channels)
        if n_active == 0:
            self.title_lbl.setText("<span style='color: #00FFA3;'>●</span> RAW EEG WAVEFORMS (0 CHANNELS SELECTED)")
            self.info_lbl.setText("Select channels in Topomap to observe raw traces")
            self.plot_widget.setYRange(-10, 10)
            return

        self.title_lbl.setText(f"<span style='color: #00FFA3;'>●</span> RAW EEG WAVEFORMS ({n_active} ACTIVE)")
        self.info_lbl.setText(f"Gain: {self.gain:.1f}x • Window: {self.window_sec:.1f}s")

        y_spacing = 75.0
        total_y_range = max(100.0, n_active * y_spacing + 50.0)
        self.plot_widget.setYRange(-30.0, total_y_range)

        for idx, ch in enumerate(self.active_channels):
            c_hex = self.data_loader.channel_colors.get(ch, '#1DB954')
            curve = pg.PlotDataItem(pen=pg.mkPen(c_hex, width=1.5))
            self.plot_widget.addItem(curve)
            self.curve_items[ch] = curve

            lbl = pg.TextItem(f"{ch}", color=c_hex, anchor=(1.0, 0.5))
            lbl.setFont(QFont("Consolas", 9, QFont.Bold))
            self.plot_widget.addItem(lbl)
            lbl.setPos(-0.08, idx * y_spacing)
            self.channel_labels[ch] = lbl

    def update_window(self, win_sec: float):
        self.window_sec = win_sec
        self.cached_t_axis = None
        self._cached_n_pts = 0

    def update_frame(self, current_sample: int):
        if len(self.active_channels) == 0:
            return

        n_pts = int(self.window_sec * self.data_loader.sfreq)
        if n_pts <= 1:
            return

        if self.cached_t_axis is None or self._cached_n_pts != n_pts:
            self.cached_t_axis = np.linspace(-self.window_sec, 0.0, n_pts, endpoint=True)
            self._cached_n_pts = n_pts

        self.plot_widget.setXRange(-self.window_sec, 0.0, padding=0.01)

        y_spacing = 75.0
        for idx, ch in enumerate(self.active_channels):
            if ch in self.data_loader.channel_names and ch in self.curve_items:
                ch_idx = self.data_loader.channel_names.index(ch)
                # Phase 4.03 Standard Sliding Window Slicing
                chunk = self.data_loader.get_window_data(ch_idx, current_sample, n_pts)
                y_offset = idx * y_spacing
                scaled_y = (chunk * self.gain) + y_offset
                self.curve_items[ch].setData(self.cached_t_axis, scaled_y)


# ==============================================================================
# SPOTIFY PLAYBACK BAR (FRAME-BY-FRAME STEPPING & PROMINENT PLAYHEAD)
# ==============================================================================

class SpotifyTimelineWidget(pg.PlotWidget):
    """
    Spotify-Style Interactive Timeline with GFP Mini-Map Waveform Navigation:
    - Renders the continuous Global Field Power (GFP) envelope across the entire dataset.
    - Overlays colored stimulus event regions (T0 Rest, T1 Left Fist, T2 Right Fist).
    - Prominent bold playhead scrubber line (width=3.5).
    """
    seek_requested = pyqtSignal(float)

    def __init__(self, data_loader: EEGDataLoader, parent=None, engine=None):
        super().__init__(parent=parent)
        self.data_loader = data_loader
        self.engine = engine if engine is not None else getattr(parent, 'engine', None)
        self.is_dragging = False

        self.upper_curve = None
        self.lower_curve = None

        self._init_ui()

    def _init_ui(self):
        self.setBackground('#0E0F14')
        self.setFixedHeight(28)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(x=False, y=False)
        self.setXRange(0.0, self.data_loader.duration, padding=0.005)
        self.setYRange(-0.5, 0.5)

        # Baseline rail (Z=0)
        rail = pg.PlotCurveItem([0.0, self.data_loader.duration], [0.0, 0.0], pen=pg.mkPen('#1C222E', width=4.0))
        rail.setZValue(0)
        self.addItem(rail)

        # Mini-Map Waveform Envelope (Z=1)
        if self.data_loader.gfp_envelope is not None:
            t_axis = np.linspace(0.0, self.data_loader.duration, len(self.data_loader.gfp_envelope))
            gfp_y = self.data_loader.gfp_envelope
            self.upper_curve = pg.PlotCurveItem(t_axis, gfp_y, fillLevel=0.0, brush=pg.mkBrush('#142232'), pen=pg.mkPen('#21344B', width=1.0))
            self.upper_curve.setZValue(1)
            self.addItem(self.upper_curve)

            self.lower_curve = pg.PlotCurveItem(t_axis, -gfp_y, fillLevel=0.0, brush=pg.mkBrush('#142232'), pen=pg.mkPen('#21344B', width=1.0))
            self.lower_curve.setZValue(1)
            self.addItem(self.lower_curve)

        # Event regions (Z=2)
        for ev in self.data_loader.events:
            cfg = EVENT_COLOR_MAP.get(ev.event_id, {'bg': '#1E2530', 'border': '#7A889B'})
            brush_c = QColor(cfg['bg'])
            brush_c.setAlpha(160)
            region = pg.LinearRegionItem(
                [ev.start_time, ev.end_time], movable=False,
                brush=QBrush(brush_c), pen=pg.mkPen(cfg['border'], width=1.0)
            )
            region.setZValue(2)
            self.addItem(region)

        # Phase 4.03: Prominent Thicker Playhead Scrubber Line (Z=5, width=3.5)
        self.playhead_line = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False,
            pen=pg.mkPen('#FFFFFF', width=3.5)
        )
        self.playhead_line.setZValue(5)
        self.addItem(self.playhead_line)

    def refresh_minimap(self):
        """Updates the GFP waveform envelope curves when filtering or FastICA changes."""
        if self.upper_curve is not None and self.data_loader.gfp_envelope is not None:
            t_axis = np.linspace(0.0, self.data_loader.duration, len(self.data_loader.gfp_envelope))
            gfp_y = self.data_loader.gfp_envelope
            self.upper_curve.setData(t_axis, gfp_y)
            self.lower_curve.setData(t_axis, -gfp_y)

    def update_playhead(self, time_sec: float):
        if not self.is_dragging:
            self.playhead_line.setValue(time_sec)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.is_dragging = True
            self._handle_scrub(ev.pos().x())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.is_dragging:
            self._handle_scrub(ev.pos().x())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.is_dragging = False
            self._handle_scrub(ev.pos().x())
        super().mouseReleaseEvent(ev)

    def _handle_scrub(self, mouse_x: float):
        width = self.width()
        if width > 0:
            frac = max(0.0, min(1.0, mouse_x / width))
            t_sec = frac * self.data_loader.duration
            self.playhead_line.setValue(t_sec)
            self.seek_requested.emit(t_sec)


class SpotifyPlaybackBar(QFrame):
    """
    Spotify-Style Transport Bar matching Phase 2 Format:
    - Left: [EEG] badge + 'PhysioNet S001 • Run 04' / 'Motor Imagery 64ch'
    - Center Top: Step Back 1 Frame (⏮), White Circular Play/Pause (▶), Step Forward 1 Frame (⏭)
    - Center Bottom: '0:00.000' + Full Interactive Timeline + '2:04.994'
    - Right: '• Rest  • Left Fist  • Right Fist' legend + 'Active Stimulus: Rest'
    """
    def __init__(self, data_loader: EEGDataLoader, engine: PlaybackEngine, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.engine = engine

        self._init_ui()

    def _init_ui(self):
        self.setFixedHeight(72)
        self.setStyleSheet("""
            QFrame {
                background: #08080C;
                border-top: 1px solid #1A1F2B;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(16)

        # Left: EEG Icon Box + Dataset Info
        left_layout = QHBoxLayout()
        left_layout.setSpacing(10)

        eeg_badge = QLabel("EEG")
        eeg_badge.setFixedSize(38, 38)
        eeg_badge.setAlignment(Qt.AlignCenter)
        eeg_badge.setStyleSheet("""
            background: #1DB954;
            color: #000000;
            font-weight: 900;
            font-size: 11px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        """)
        left_layout.addWidget(eeg_badge)

        meta_layout = QVBoxLayout()
        meta_layout.setSpacing(2)
        meta_layout.setAlignment(Qt.AlignVCenter)

        title_lbl = QLabel("PhysioNet S001 • Run 04")
        title_lbl.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold;")
        sub_lbl = QLabel("Motor Imagery 64ch • 160 Hz")
        sub_lbl.setStyleSheet("color: #8C9BAE; font-size: 10px;")

        meta_layout.addWidget(title_lbl)
        meta_layout.addWidget(sub_lbl)
        left_layout.addLayout(meta_layout)

        layout.addLayout(left_layout, stretch=0)
        layout.addSpacing(16)

        # Center: Playback Controls & Interactive Timeline Bar
        center_layout = QVBoxLayout()
        center_layout.setSpacing(4)
        center_layout.setAlignment(Qt.AlignCenter)

        # Button row: Step Back (⏮), Play/Pause (▶), Step Forward (⏭)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(14)
        btn_row.setAlignment(Qt.AlignCenter)

        # Phase 4.04: Frame-by-Frame Backward (1 sample / 6.25ms) with NoFocus
        self.btn_prev = QPushButton("⏮")
        self.btn_prev.setFixedSize(28, 28)
        self.btn_prev.setToolTip("Previous Frame (1 sample / 6.25ms) [Left Arrow]")
        self.btn_prev.setFocusPolicy(Qt.NoFocus)
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #8C9BAE;
                font-size: 14px;
                border: none;
                outline: none;
            }
            QPushButton:hover {
                color: #FFFFFF;
                background: transparent;
            }
            QPushButton:pressed {
                color: #00FFA3;
                background: transparent;
            }
            QPushButton:focus {
                outline: none;
                border: none;
            }
        """)
        self.btn_prev.clicked.connect(lambda: self.engine.step_frame_backward(1))

        # Play/Pause (▶) with NoFocus
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(34, 34)
        self.btn_play.setToolTip("Play / Pause (Space)")
        self.btn_play.setFocusPolicy(Qt.NoFocus)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                color: #000000;
                font-size: 14px;
                border-radius: 17px;
                font-weight: bold;
                padding-left: 2px;
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background: #1DB954;
                color: #000000;
            }
            QPushButton:pressed {
                background: #179B46;
                color: #000000;
            }
            QPushButton:focus {
                outline: none;
                border: none;
            }
        """)
        self.btn_play.clicked.connect(self.engine.toggle_play)

        # Phase 4.04: Frame-by-Frame Forward (1 sample / 6.25ms) with NoFocus
        self.btn_next = QPushButton("⏭")
        self.btn_next.setFixedSize(28, 28)
        self.btn_next.setToolTip("Next Frame (1 sample / 6.25ms) [Right Arrow]")
        self.btn_next.setFocusPolicy(Qt.NoFocus)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #8C9BAE;
                font-size: 14px;
                border: none;
                outline: none;
            }
            QPushButton:hover {
                color: #FFFFFF;
                background: transparent;
            }
            QPushButton:pressed {
                color: #00FFA3;
                background: transparent;
            }
            QPushButton:focus {
                outline: none;
                border: none;
            }
        """)
        self.btn_next.clicked.connect(lambda: self.engine.step_frame_forward(1))

        btn_row.addWidget(self.btn_prev)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_next)
        center_layout.addLayout(btn_row)

        # Timeline row: Time Elapsed + Timeline Plot + Total Duration
        time_row = QHBoxLayout()
        time_row.setSpacing(8)

        self.time_cur_lbl = QLabel("0:00.000")
        self.time_cur_lbl.setFixedWidth(52)
        self.time_cur_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_cur_lbl.setStyleSheet("color: #8C9BAE; font-family: 'Consolas', monospace; font-size: 10px;")

        self.timeline = SpotifyTimelineWidget(self.data_loader, self, engine=self.engine)
        self.timeline.seek_requested.connect(self.engine.seek_time)

        total_sec = self.data_loader.duration
        mins = int(total_sec // 60)
        secs = total_sec % 60
        self.time_dur_lbl = QLabel(f"{mins}:{secs:06.3f}")
        self.time_dur_lbl.setFixedWidth(52)
        self.time_dur_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.time_dur_lbl.setStyleSheet("color: #8C9BAE; font-family: 'Consolas', monospace; font-size: 10px;")

        time_row.addWidget(self.time_cur_lbl)
        time_row.addWidget(self.timeline, stretch=1)
        time_row.addWidget(self.time_dur_lbl)
        center_layout.addLayout(time_row)

        layout.addLayout(center_layout, stretch=1)
        layout.addSpacing(16)

        # Right: Stimulus Legend & Real-Time Active State
        right_layout = QVBoxLayout()
        right_layout.setSpacing(4)
        right_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        legend_row = QHBoxLayout()
        legend_row.setSpacing(8)
        legend_row.setAlignment(Qt.AlignRight)

        for eid in ['T0', 'T1', 'T2']:
            cfg = EVENT_COLOR_MAP[eid]
            b_col = cfg['border']
            n_str = cfg['name']
            lbl = QLabel(f"<span style='color: {b_col};'>●</span> {n_str}")
            lbl.setStyleSheet("color: #A0AEC0; font-size: 10px; font-weight: bold;")
            legend_row.addWidget(lbl)
        right_layout.addLayout(legend_row)

        self.stimulus_state_lbl = QLabel("Active Stimulus: Rest")
        self.stimulus_state_lbl.setStyleSheet("""
            color: #00E5FF;
            font-size: 10px;
            font-weight: bold;
            background: #141A24;
            border: 1px solid #1E2B3D;
            border-radius: 3px;
            padding: 2px 6px;
        """)
        right_layout.addWidget(self.stimulus_state_lbl, alignment=Qt.AlignRight)

        layout.addLayout(right_layout, stretch=0)

    def update_frame(self, sample_idx: int, t_sec: float):
        mins = int(t_sec // 60)
        secs = t_sec % 60
        self.time_cur_lbl.setText(f"{mins}:{secs:06.3f}")
        self.timeline.update_playhead(t_sec)

        active_ev = None
        for ev in self.data_loader.events:
            if ev.start_time <= t_sec <= ev.end_time:
                active_ev = ev
                break

        if active_ev:
            cfg = EVENT_COLOR_MAP.get(active_ev.event_id, {'name': active_ev.event_id, 'text': '#00FFA3', 'bg': '#0D332D', 'border': '#00FFA3'})
            self.stimulus_state_lbl.setText(f"Active Stimulus: {cfg['name']}")
            self.stimulus_state_lbl.setStyleSheet(f"""
                color: {cfg['text']};
                font-size: 10px;
                font-weight: bold;
                background: {cfg['bg']};
                border: 1px solid {cfg['border']};
                border-radius: 3px;
                padding: 2px 6px;
            """)
        else:
            self.stimulus_state_lbl.setText("Active Stimulus: Baseline")
            self.stimulus_state_lbl.setStyleSheet("""
                color: #7A889B;
                font-size: 10px;
                background: #14171E;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 2px 6px;
            """)

    def update_playback_state(self, is_playing: bool):
        if is_playing:
            self.btn_play.setText("⏸")
            self.btn_play.setStyleSheet("""
                QPushButton {
                    background: #1DB954;
                    color: #000000;
                    font-size: 14px;
                    border-radius: 17px;
                    font-weight: bold;
                    padding-left: 0px;
                    border: none;
                    outline: none;
                }
                QPushButton:hover {
                    background: #23D260;
                }
                QPushButton:focus {
                    outline: none;
                    border: none;
                }
            """)
        else:
            self.btn_play.setText("▶")
            self.btn_play.setStyleSheet("""
                QPushButton {
                    background: #FFFFFF;
                    color: #000000;
                    font-size: 14px;
                    border-radius: 17px;
                    font-weight: bold;
                    padding-left: 2px;
                    border: none;
                    outline: none;
                }
                QPushButton:hover {
                    background: #1DB954;
                    color: #000000;
                }
                QPushButton:focus {
                    outline: none;
                    border: none;
                }
            """)


# ==============================================================================
# HEADER WIDGET & COLLAPSIBLE CONTROL SIDEBAR
# ==============================================================================

class HeaderWidget(QFrame):
    export_requested = pyqtSignal()
    toggle_sidebar_requested = pyqtSignal()

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._init_ui()

    def _init_ui(self):
        self.setFixedHeight(48)
        self.setStyleSheet("""
            QFrame {
                background: #060709;
                border-bottom: 1px solid #14171E;
            }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(14)

        logo = QLabel("NEUROMAP")
        logo.setStyleSheet("color: #FFFFFF; font-size: 15px; font-weight: 900; letter-spacing: 2px;")
        layout.addWidget(logo)

        tag = QLabel("RESEARCH SUITE v4.04")
        tag.setStyleSheet("color: #00FFA3; background: #0D332D; border: 1px solid #00FFA3; border-radius: 3px; font-size: 9px; font-weight: bold; padding: 2px 6px;")
        layout.addWidget(tag)

        layout.addStretch()

        ch_badge = QLabel("64 Channels • 160Hz")
        ch_badge.setStyleSheet("color: #8C9BAE; font-size: 11px;")
        layout.addWidget(ch_badge)

        btn_export = QPushButton("Export Filtered CSV")
        btn_export.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #DDE2EB;
                border: 1px solid #232B3B;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1F2533;
                border-color: #00FFA3;
                color: #00FFA3;
            }
        """)
        btn_export.clicked.connect(self.export_requested.emit)
        layout.addWidget(btn_export)


class ControlSidebarWidget(QFrame):
    """
    Phase 4.04 Collapsible Control Sidebar:
    - Speed selector limited to [0.1x, 0.25x, 0.5x, 1.0x].
    - FastICA & ASR Quick Status Indicators.
    - Collapsible from 240px to 36px.
    """
    def __init__(self, data_loader: EEGDataLoader, engine: PlaybackEngine, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.engine = engine
        self.is_collapsed = False
        self._init_ui()

    def _init_ui(self):
        self.setFixedWidth(240)
        self.setStyleSheet("""
            QFrame#SidebarRoot {
                background: #090A0E;
                border-right: 1px solid #14171E;
            }
            QLabel {
                color: #8C9BAE;
                font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #181C26;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 11px;
                font-weight: bold;
                color: #00FFA3;
            }
        """)
        self.setObjectName("SidebarRoot")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Header with Collapse Toggle Button
        h_bar = QHBoxLayout()
        h_bar.setContentsMargins(4, 2, 4, 4)

        self.title_lbl = QLabel("CONTROL PANEL")
        self.title_lbl.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: 900; letter-spacing: 1px;")
        h_bar.addWidget(self.title_lbl)
        h_bar.addStretch()

        self.btn_toggle = QPushButton("◀")
        self.btn_toggle.setFixedSize(24, 24)
        self.btn_toggle.setToolTip("Collapse / Expand Sidebar (F4)")
        self.btn_toggle.setFocusPolicy(Qt.NoFocus)
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                outline: none;
            }
            QPushButton:hover {
                background: #1F2533;
                color: #00FFA3;
                border-color: #00FFA3;
            }
            QPushButton:focus {
                outline: none;
            }
        """)
        self.btn_toggle.clicked.connect(self.toggle_collapse)
        h_bar.addWidget(self.btn_toggle)
        main_layout.addLayout(h_bar)

        # Collapsed vertical strip label
        self.collapsed_lbl = QLabel("C\nO\nN\nT\nR\nO\nL\nS")
        self.collapsed_lbl.setAlignment(Qt.AlignCenter)
        self.collapsed_lbl.setStyleSheet("color: #4A5568; font-size: 10px; font-weight: 900; letter-spacing: 3px;")
        self.collapsed_lbl.setVisible(False)
        main_layout.addWidget(self.collapsed_lbl, stretch=1)

        # Body Container
        self.controls_body = QWidget()
        body_layout = QVBoxLayout(self.controls_body)
        body_layout.setContentsMargins(4, 4, 4, 4)
        body_layout.setSpacing(11)

        # Speed Selector
        speed_box = QHBoxLayout()
        speed_lbl = QLabel("Speed:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.1x", "0.25x", "0.5x", "1.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        speed_box.addWidget(speed_lbl)
        speed_box.addWidget(self.speed_combo, stretch=1)
        body_layout.addLayout(speed_box)

        # Gain Slider
        gain_box = QVBoxLayout()
        self.gain_val_lbl = QLabel("Signal Gain: 1.0x")
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(2, 50)
        self.gain_slider.setValue(10)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_box.addWidget(self.gain_val_lbl)
        gain_box.addWidget(self.gain_slider)
        body_layout.addLayout(gain_box)

        # Time Window Slider
        win_box = QVBoxLayout()
        self.win_val_lbl = QLabel("Time Window: 4.0s")
        self.win_slider = QSlider(Qt.Horizontal)
        self.win_slider.setRange(1, 10)
        self.win_slider.setValue(4)
        self.win_slider.valueChanged.connect(self._on_win_changed)
        win_box.addWidget(self.win_val_lbl)
        win_box.addWidget(self.win_slider)
        body_layout.addLayout(win_box)

        # Topomap Scale Slider
        topo_box = QVBoxLayout()
        self.topo_val_lbl = QLabel("Topomap Limit: ±50 µV")
        self.topo_slider = QSlider(Qt.Horizontal)
        self.topo_slider.setRange(10, 150)
        self.topo_slider.setValue(50)
        self.topo_slider.valueChanged.connect(self._on_topo_changed)
        topo_box.addWidget(self.topo_val_lbl)
        topo_box.addWidget(self.topo_slider)
        body_layout.addLayout(topo_box)

        body_layout.addSpacing(4)

        # FastICA Artifact Removal Section
        ica_title = QLabel("FASTICA ARTIFACT REMOVAL")
        ica_title.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: 900; letter-spacing: 1px;")
        body_layout.addWidget(ica_title)

        self.chk_enable_ica = QCheckBox("Enable ICA Cleaning")
        self.chk_enable_ica.setChecked(False)
        self.chk_enable_ica.setToolTip("Instantly toggles cleaned vs. raw EEG across waveforms and topomaps")
        body_layout.addWidget(self.chk_enable_ica)

        preset_box = QVBoxLayout()
        preset_box.setSpacing(3)
        preset_lbl = QLabel("Artifact Mode Preset:")
        preset_lbl.setStyleSheet("font-size: 10px; color: #8C9BAE;")
        self.combo_ica_preset = QComboBox()
        self.combo_ica_preset.addItems([
            "Ocular / Blinks (IC0)",
            "Ocular + Saccades (IC0, IC1)",
            "Aggressive EOG + EMG (IC0, IC1, IC2)"
        ])
        preset_box.addWidget(preset_lbl)
        preset_box.addWidget(self.combo_ica_preset)
        body_layout.addLayout(preset_box)

        self.btn_recompute_ica = QPushButton("⚡ Re-compute FastICA")
        self.btn_recompute_ica.setToolTip("Run on-demand FastICA spatial decomposition across 64 channels")
        self.btn_recompute_ica.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #00FFA3;
                border: 1px solid #1F2737;
                border-radius: 4px;
                padding: 5px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1F2737;
                border-color: #00FFA3;
            }
        """)
        body_layout.addWidget(self.btn_recompute_ica)

        self.ica_status_lbl = QLabel("ICA: Standby (On-Demand)")
        self.ica_status_lbl.setStyleSheet("color: #7A889B; font-size: 9px; font-style: italic;")
        body_layout.addWidget(self.ica_status_lbl)

        body_layout.addSpacing(4)

        # DSP Filter Settings
        filter_title = QLabel("DSP BANDPASS FILTERS")
        filter_title.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: 900; letter-spacing: 1px;")
        body_layout.addWidget(filter_title)

        f_grid = QGridLayout()
        f_grid.setSpacing(5)

        f_grid.addWidget(QLabel("Low (Hz):"), 0, 0)
        self.f_low = QLineEdit("1.0")
        f_grid.addWidget(self.f_low, 0, 1)

        f_grid.addWidget(QLabel("High (Hz):"), 1, 0)
        self.f_high = QLineEdit("40.0")
        f_grid.addWidget(self.f_high, 1, 1)

        f_grid.addWidget(QLabel("Notch (Hz):"), 2, 0)
        self.f_notch = QLineEdit("60.0")
        f_grid.addWidget(self.f_notch, 2, 1)

        body_layout.addLayout(f_grid)

        btn_apply_filters = QPushButton("Apply Filters")
        btn_apply_filters.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #00FFA3;
                border: 1px solid #1F2737;
                border-radius: 4px;
                padding: 5px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1F2737;
                border-color: #00FFA3;
            }
        """)
        btn_apply_filters.clicked.connect(self._on_apply_filters)
        body_layout.addWidget(btn_apply_filters)

        body_layout.addStretch()

        help_box = QLabel("Navigation:\n• Space: Play/Pause\n• Left/Right: Frame by frame\n• F4: Toggle Sidebar")
        help_box.setStyleSheet("color: #4A5568; font-size: 10px; line-height: 14px;")
        body_layout.addWidget(help_box)

        main_layout.addWidget(self.controls_body, stretch=1)

    def toggle_collapse(self):
        self.is_collapsed = not self.is_collapsed
        if self.is_collapsed:
            self.controls_body.setVisible(False)
            self.title_lbl.setVisible(False)
            self.collapsed_lbl.setVisible(True)
            self.setFixedWidth(36)
            self.btn_toggle.setText("▶")
            self.btn_toggle.setToolTip("Expand Sidebar (F4)")
        else:
            self.collapsed_lbl.setVisible(False)
            self.title_lbl.setVisible(True)
            self.controls_body.setVisible(True)
            self.setFixedWidth(240)
            self.btn_toggle.setText("◀")
            self.btn_toggle.setToolTip("Collapse Sidebar (F4)")

    def _on_speed_changed(self, text: str):
        speed_map = {"0.1x": 0.1, "0.25x": 0.25, "0.5x": 0.5, "1.0x": 1.0}
        self.engine.set_speed(speed_map.get(text, 1.0))

    def _on_gain_changed(self, val: int):
        gain = val / 10.0
        self.gain_val_lbl.setText(f"Signal Gain: {gain:.1f}x")

    def _on_win_changed(self, val: int):
        self.win_val_lbl.setText(f"Time Window: {val:.1f}s")

    def _on_topo_changed(self, val: int):
        self.topo_val_lbl.setText(f"Topomap Limit: ±{val} µV")

    def _on_apply_filters(self):
        try:
            l_val = float(self.f_low.text())
            h_val = float(self.f_high.text())
            n_val = float(self.f_notch.text())
            self.data_loader.apply_dsp_filters(l_val, h_val, n_val)
        except ValueError:
            print("[WARN] Invalid DSP filter parameters entered.")


# ==============================================================================
# MAIN APPLICATION WINDOW
# ==============================================================================

