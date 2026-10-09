"""
neuromap - Professional Closed-Loop EEG Visualization Dashboard
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Phase 4.02 Architecture:
- Top-Left: Analysis Workstation Panel (Clean placeholder for Phase 5 analytical modules)
- Top-Right: Dual Topomap Container with Mode Toggle [2D Topomap] | [3D Brain]
  * 2D Mode:
    - Phase 3 clean aesthetics (no filled-in grey background, transparent outside/inside outline).
    - No circular borders around unselected electrode points (crisp white text labels only).
    - Selected channels display a vivid circular badge with thick white border.
    - Integrated Voltage Colorbar (µV) with dynamic scale ticks matching unified colormap.
    - High-Resolution (128x128) Smooth Regularized Interpolation with anti-aliased edge masking.
  * 3D Mode:
    - Full cortical coverage (frontal, temporal, parietal, occipital lobes).
    - Strict bilateral symmetry (Z <-> -Z) across all 27 pairs.
    - Strict midline Z=0 pinning bridging the longitudinal fissure.
    - Outward normal offset (+0.055) preventing mesh submersion.
    - Cold-Standby: Lazily initialized upon first toggle, completely dormant when unselected.
- Zero-Overhead Dissolved Section Culling:
  - If topomap, analysis workstation, or waveforms are dragged closed (width/height < 25),
    rendering is immediately bypassed and 3D engine is paused.
- Phase 4.02 System Enhancements:
  1. Centralized Channel State Manager (single source of truth across all widgets).
  2. Shared Interpolation Matrix Caching at Startup (instantaneous 2D/3D access).
  3. Unified Colormap Specification (100% color parity across 2D, 3D, and colorbars).
  4. Collapsible Control Sidebar with persistent toggle button (expands workspace).
  5. Mini-Map Navigation in Timeline (Global Field Power envelope waveform track).
"""

import sys
import os
import time
import math
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import scipy.signal

# PyQt5 GUI Framework
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QPointF, QRectF, QSize
)
from PyQt5.QtGui import (
    QColor, QFont, QPen, QBrush, QPainter, QLinearGradient, QRadialGradient,
    QPolygonF, QIcon, QKeySequence, QTransform
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QLabel, QPushButton, QSlider, QLineEdit,
    QCheckBox, QComboBox, QHeaderView, QStatusBar, QProgressBar,
    QFileDialog, QShortcut, QFrame, QSizePolicy, QStackedWidget
)

# PyQtGraph High-Performance Scientific Graphics
import pyqtgraph as pg

# CRITICAL ENGINE SAFETY: Keep useOpenGL=False when running PyVistaQt on Windows
# to prevent fatal OpenGL context collisions (0xC0000005) between QOpenGLWidget and VTK.
pg.setConfigOptions(useOpenGL=False, antialias=True)

# PyVista 3D Visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# MNE-Python for Neurophysiology
try:
    import mne
    from mne.datasets import eegbci
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# ==============================================================================
# DATA STRUCTURES & CONSTANTS (EXACTLY 64 CHANNELS)
# ==============================================================================

STANDARD_64_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'T9', 'T10',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'Iz'
]

PALETTE_COLORS = [
    '#1DB954', '#00E5FF', '#E040FB', '#FFD600', '#FF5252', '#69F0AE',
    '#448AFF', '#FF6E40', '#EEFF41', '#B388FF', '#18FFFF', '#FF4081',
    '#64FFDA', '#B2FF59', '#FFAB40', '#7C4DFF', '#00B0FF', '#FF5722',
    '#40C4FF', '#A7FFEB', '#FFD180', '#FF80AB', '#EA80FC', '#82B1FF'
]

EVENT_COLOR_MAP = {
    'T0': {'name': 'Rest', 'bg': '#1E2530', 'border': '#00E5FF', 'text': '#00E5FF', 'dot': '#00E5FF'},
    'T1': {'name': 'Left Fist', 'bg': '#0D332D', 'border': '#00FFA3', 'text': '#00FFA3', 'dot': '#00FFA3'},
    'T2': {'name': 'Right Fist', 'bg': '#331238', 'border': '#BA68C8', 'text': '#BA68C8', 'dot': '#BA68C8'}
}

@dataclass
class StimulusEvent:
    event_id: str
    label: str
    start_time: float
    end_time: float
    start_sample: int
    end_sample: int

MONTAGE_2D_COORDS = {
    'Fp1': (-0.30, 0.85), 'Fpz': (0.00, 0.88), 'Fp2': (0.30, 0.85),
    'AF7': (-0.60, 0.70), 'AF3': (-0.35, 0.68), 'AFz': (0.00, 0.70), 'AF4': (0.35, 0.68), 'AF8': (0.60, 0.70),
    'F7': (-0.75, 0.50), 'F5': (-0.52, 0.48), 'F3': (-0.32, 0.47), 'F1': (-0.12, 0.46),
    'Fz': (0.00, 0.46), 'F2': (0.12, 0.46), 'F4': (0.32, 0.47), 'F6': (0.52, 0.48), 'F8': (0.75, 0.50),
    'FT7': (-0.85, 0.25), 'FC5': (-0.62, 0.24), 'FC3': (-0.38, 0.24), 'FC1': (-0.15, 0.24),
    'FCz': (0.00, 0.24), 'FC2': (0.15, 0.24), 'FC4': (0.38, 0.24), 'FC6': (0.62, 0.24), 'FT8': (0.85, 0.25),
    'T7': (-0.90, 0.00), 'C5': (-0.68, 0.00), 'C3': (-0.45, 0.00), 'C1': (-0.20, 0.00),
    'Cz': (0.00, 0.00), 'C2': (0.20, 0.00), 'C4': (0.45, 0.00), 'C6': (0.68, 0.00), 'T8': (0.90, 0.00),
    'T9': (-0.96, -0.12), 'T10': (0.96, -0.12),
    'TP7': (-0.85, -0.25), 'CP5': (-0.62, -0.24), 'CP3': (-0.38, -0.24), 'CP1': (-0.15, -0.24),
    'CPz': (0.00, -0.24), 'CP2': (0.15, -0.24), 'CP4': (0.38, -0.24), 'CP6': (0.62, -0.24), 'TP8': (0.85, -0.25),
    'P7': (-0.75, -0.50), 'P5': (-0.52, -0.48), 'P3': (-0.32, -0.47), 'P1': (-0.12, -0.46),
    'Pz': (0.00, -0.46), 'P2': (0.12, -0.46), 'P4': (0.32, -0.47), 'P6': (0.52, -0.48), 'P8': (0.75, -0.50),
    'PO7': (-0.60, -0.70), 'PO3': (-0.35, -0.68), 'POz': (0.00, -0.70), 'PO4': (0.35, -0.68), 'PO8': (0.60, -0.70),
    'O1': (-0.30, -0.85), 'Oz': (0.00, -0.88), 'O2': (0.30, -0.85), 'Iz': (0.00, -0.96)
}


# ==============================================================================
# UNIFIED COLORMAP SPECIFICATION (2D & 3D PARITY)
# ==============================================================================
# Canonical Coolwarm Diverging Palette:
# -V: Deep Blue (#1E88E5) -> Cyan (#00E5FF) -> Neutral 0V (#12151C) -> Coral (#FF6E40) -> +V: Crimson (#E53935)
COLORMAP_STOPS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
COLORMAP_COLORS_RGBA = np.array([
    [30, 136, 229, 240],   # -V Deep Blue
    [0, 229, 255, 230],    # Cyan
    [18, 21, 28, 210],     # 0V Neutral Dark
    [255, 110, 64, 230],   # Coral Orange
    [229, 57, 53, 240]     # +V Deep Crimson Red
], dtype=np.float32)

PYVISTA_CMAP = ['#1E88E5', '#00E5FF', '#12151C', '#FF6E40', '#E53935']

def generate_unified_lut_256() -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.uint8)
    for c in range(4):
        lut[:, c] = np.interp(np.linspace(0, 1, 256), COLORMAP_STOPS, COLORMAP_COLORS_RGBA[:, c]).astype(np.uint8)
    return lut

UNIFIED_LUT_256 = generate_unified_lut_256()


# ==============================================================================
# CENTRALIZED CHANNEL STATE & CACHE MANAGERS
# ==============================================================================

class CentralizedChannelState(QObject):
    """
    Phase 4.02 Centralized Channel State:
    Single source of truth for channel activation across 2D Topomap, 3D Brain,
    Cascading Waveforms, and Header Badges.
    """
    selection_changed = pyqtSignal(set)

    def __init__(self, channel_names: List[str]):
        super().__init__()
        self.channel_names = list(channel_names)
        self.selected_channels: Set[str] = set()

    def toggle_channel(self, ch: str) -> bool:
        if ch in self.selected_channels:
            self.selected_channels.remove(ch)
            active = False
        else:
            self.selected_channels.add(ch)
            active = True
        self.selection_changed.emit(self.selected_channels)
        return active

    def select_all(self):
        self.selected_channels = set(self.channel_names)
        self.selection_changed.emit(self.selected_channels)

    def clear_all(self):
        self.selected_channels.clear()
        self.selection_changed.emit(self.selected_channels)

    def is_selected(self, ch: str) -> bool:
        return ch in self.selected_channels

    @property
    def count(self) -> int:
        return len(self.selected_channels)


class SharedInterpolationCache:
    """
    Phase 4.02 Shared Interpolation Matrix Cache:
    Precomputes high-resolution (128x128) 2D smooth regularized IDW matrix and
    anti-aliased circular alpha mask at application startup.
    Also caches 3D IDW matrix upon first computation.
    """
    def __init__(self, channel_names: List[str]):
        self.channel_names = channel_names
        self.grid_res_2d = 128
        
        # 1. High-Resolution 2D Grid
        x = np.linspace(-1.05, 1.05, self.grid_res_2d)
        y = np.linspace(-1.05, 1.05, self.grid_res_2d)
        self.grid_x, self.grid_y = np.meshgrid(x, y, indexing='ij')
        r_grid = np.sqrt(self.grid_x**2 + self.grid_y**2)

        # Anti-aliased smoothstep boundary mask
        self.alpha_mask_2d = np.clip((1.015 - r_grid) / 0.030, 0.0, 1.0)

        # 2. Extract Valid 2D Coordinates
        self.node_positions_2d = []
        self.valid_ch_names_2d = []
        for ch in self.channel_names:
            if ch in MONTAGE_2D_COORDS:
                self.node_positions_2d.append(MONTAGE_2D_COORDS[ch])
                self.valid_ch_names_2d.append(ch)

        coords_arr = np.array(self.node_positions_2d, dtype=np.float32)
        grid_pts = np.vstack([self.grid_x.ravel(), self.grid_y.ravel()]).T

        # 3. Regularized Multiquadric IDW Kernel (avoids bullseye artifacts)
        diff = grid_pts[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        smoothing = 0.012
        weights = 1.0 / ((dist_sq + smoothing) ** 1.15)
        weights /= np.sum(weights, axis=1, keepdims=True)
        self.idw_2d_matrix = weights.astype(np.float32)

        # 3D Cache slots
        self.idw_3d_matrix: Optional[np.ndarray] = None
        self.elec_coords_3d: Dict[str, np.ndarray] = {}
        self.valid_ch_names_3d: List[str] = []


# ==============================================================================
# DATA LOADER & SIGNAL PROCESSING
# ==============================================================================

class EEGDataLoader:
    """
    Loads MNE PhysioNet Motor Movement/Imagery dataset or generates high-fidelity
    synthetic 64-channel EEG with alpha rhythms and motor imagery desynchronization.
    Applies zero-phase Butterworth bandpass & notch filters.
    """
    def __init__(self):
        self.sfreq: float = 160.0
        self.channel_names: List[str] = STANDARD_64_CHANNELS.copy()
        self.raw_data: Optional[np.ndarray] = None
        self.filtered_data: Optional[np.ndarray] = None
        self.n_channels: int = 64
        self.n_samples: int = 0
        self.duration: float = 0.0
        self.events: List[StimulusEvent] = []
        self.channel_colors: Dict[str, str] = {}

        for i, ch in enumerate(self.channel_names):
            self.channel_colors[ch] = PALETTE_COLORS[i % len(PALETTE_COLORS)]

        # Centralized State and Caches
        self.channel_state = CentralizedChannelState(self.channel_names)
        self.interpolation_cache = SharedInterpolationCache(self.channel_names)
        self.gfp_envelope: Optional[np.ndarray] = None

    def load_dataset(self) -> bool:
        if MNE_AVAILABLE:
            try:
                print("[INFO] Fetching PhysioNet Subject 1, Run 4...")
                raw_fnames = eegbci.load_data(1, [4], update_path=False)
                raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
                
                raw_chs = [ch.strip('.') for ch in raw.ch_names]
                std_upper_map = {ch.upper(): ch for ch in STANDARD_64_CHANNELS}
                
                matched_indices = []
                final_names = []
                for idx, r_ch in enumerate(raw_chs):
                    r_clean = r_ch.upper()
                    if r_clean in std_upper_map:
                        matched_indices.append(idx)
                        final_names.append(std_upper_map[r_clean])

                data = raw.get_data()
                if len(matched_indices) >= 60:
                    data = data[matched_indices, :]
                    self.channel_names = STANDARD_64_CHANNELS.copy()
                    
                    if data.shape[0] < 64:
                        pad_ch = np.zeros((64 - data.shape[0], data.shape[1]), dtype=data.dtype)
                        data = np.vstack([data, pad_ch])
                    
                    self.raw_data = data[:64, :] * 1e6
                    self.sfreq = float(raw.info['sfreq'])
                    self.n_channels, self.n_samples = self.raw_data.shape
                    self.duration = self.n_samples / self.sfreq
                    
                    try:
                        events_arr, event_dict = mne.events_from_annotations(raw, verbose=False)
                        rev_dict = {v: k for k, v in event_dict.items()}
                        for ev in events_arr:
                            sample = int(ev[0])
                            eid = rev_dict.get(ev[2], f"T{ev[2]}")
                            t_start = sample / self.sfreq
                            self.events.append(StimulusEvent(
                                event_id=eid,
                                label=EVENT_COLOR_MAP.get(eid, {}).get('name', eid),
                                start_time=t_start,
                                end_time=t_start + 4.0,
                                start_sample=sample,
                                end_sample=min(self.n_samples, sample + int(4.0 * self.sfreq))
                            ))
                    except Exception:
                        self._generate_synthetic_events()

                    print(f"[SUCCESS] Loaded {self.n_channels} channels, {self.n_samples} samples ({self.duration:.1f}s @ {self.sfreq:.0f}Hz).")
                    self.apply_dsp_filters()
                    self._compute_gfp_envelope()
                    return True
            except Exception as e:
                print(f"[WARN] MNE loading failed ({e}). Falling back to procedural 64-channel EEG.")

        self._generate_procedural_eeg()
        self.apply_dsp_filters()
        self._compute_gfp_envelope()
        return True

    def _generate_procedural_eeg(self):
        self.sfreq = 160.0
        self.duration = 125.0
        self.n_samples = int(self.sfreq * self.duration)
        self.n_channels = 64
        self.channel_names = STANDARD_64_CHANNELS.copy()

        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        self.raw_data = np.zeros((self.n_channels, self.n_samples), dtype=np.float32)

        for i in range(self.n_channels):
            pink_noise = np.cumsum(np.random.randn(self.n_samples)) * 0.4
            pink_noise -= scipy.signal.savgol_filter(pink_noise, 501, 2)
            alpha_carrier = 14.0 * np.sin(2 * np.pi * 10.2 * t + (i * 0.18))
            beta_carrier = 5.0 * np.sin(2 * np.pi * 21.0 * t + (i * 0.35))
            theta_mod = 8.0 * np.sin(2 * np.pi * 6.0 * t)
            drift = 12.0 * np.sin(2 * np.pi * 0.3 * t)
            line_noise = 2.5 * np.sin(2 * np.pi * 60.0 * t)

            self.raw_data[i, :] = pink_noise + alpha_carrier + beta_carrier + theta_mod + drift + line_noise

        self._generate_synthetic_events()
        print(f"[SUCCESS] Generated procedural EEG ({self.n_channels} channels, {self.n_samples} samples).")

    def _generate_synthetic_events(self):
        self.events.clear()
        cycle = [('T0', 4.0), ('T1', 4.0), ('T0', 4.0), ('T2', 4.0)]
        curr_t = 2.0
        idx = 0
        while curr_t < (self.duration - 4.0):
            eid, dur = cycle[idx % len(cycle)]
            s_samp = int(curr_t * self.sfreq)
            e_samp = int((curr_t + dur) * self.sfreq)
            self.events.append(StimulusEvent(
                event_id=eid,
                label=EVENT_COLOR_MAP[eid]['name'],
                start_time=curr_t,
                end_time=curr_t + dur,
                start_sample=s_samp,
                end_sample=e_samp
            ))
            curr_t += dur
            idx += 1

    def apply_dsp_filters(self, l_freq: float = 1.0, h_freq: float = 40.0, notch_freq: float = 60.0):
        print(f"[INFO] Filtering EEG: Bandpass [{l_freq}-{h_freq} Hz], Notch [{notch_freq} Hz]...")
        self.filtered_data = np.zeros_like(self.raw_data)
        nyq = 0.5 * self.sfreq

        low = max(0.001, l_freq / nyq)
        high = min(0.999, h_freq / nyq)
        b_band, a_band = scipy.signal.butter(3, [low, high], btype='bandpass')

        w0 = notch_freq / nyq
        b_notch, a_notch = scipy.signal.iirnotch(w0, 30.0)

        for i in range(self.n_channels):
            s = self.raw_data[i, :]
            s_bp = scipy.signal.filtfilt(b_band, a_band, s)
            s_clean = scipy.signal.filtfilt(b_notch, a_notch, s_bp)
            self.filtered_data[i, :] = s_clean

        self._compute_gfp_envelope()

    def _compute_gfp_envelope(self):
        """Computes Global Field Power (GFP) envelope for mini-map timeline navigation."""
        if self.filtered_data is None:
            return
        mean_v = np.mean(self.filtered_data, axis=0)
        gfp = np.sqrt(np.mean((self.filtered_data - mean_v)**2, axis=0))
        target_pts = 1200
        step = max(1, self.n_samples // target_pts)
        gfp_sub = gfp[::step][:target_pts]
        denom = (np.max(gfp_sub) - np.min(gfp_sub)) + 1e-6
        self.gfp_envelope = ((gfp_sub - np.min(gfp_sub)) / denom) * 0.7 - 0.35


# ==============================================================================
# PLAYBACK ENGINE (HIGH PRECISION TICKER)
# ==============================================================================

class PlaybackEngine(QObject):
    frame_changed = pyqtSignal(int, float)
    state_changed = pyqtSignal(bool)

    def __init__(self, data_loader: EEGDataLoader):
        super().__init__()
        self.data_loader = data_loader
        self.is_playing = False
        self.playback_speed = 1.0
        self.current_sample = 0

        self.timer = QTimer()
        self.timer.setInterval(30)  # ~33 FPS wall-clock ticker
        self.timer.timeout.connect(self._on_tick)

        self._last_tick_time = time.perf_counter()

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self._last_tick_time = time.perf_counter()
            self.timer.start()
            self.state_changed.emit(True)

    def pause(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.state_changed.emit(False)

    def toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed: float):
        self.playback_speed = max(0.1, min(8.0, speed))

    def seek_sample(self, sample_idx: int):
        self.current_sample = max(0, min(self.data_loader.n_samples - 1, sample_idx))
        t_sec = self.current_sample / self.data_loader.sfreq
        self.frame_changed.emit(self.current_sample, t_sec)

    def seek_time(self, t_sec: float):
        self.seek_sample(int(t_sec * self.data_loader.sfreq))

    def step_forward(self, seconds: float = 1.0):
        self.seek_sample(self.current_sample + int(seconds * self.data_loader.sfreq))

    def step_backward(self, seconds: float = 1.0):
        self.seek_sample(self.current_sample - int(seconds * self.data_loader.sfreq))

    def _on_tick(self):
        now = time.perf_counter()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        delta_samples = dt * self.data_loader.sfreq * self.playback_speed
        self.current_sample += int(round(delta_samples))

        if self.current_sample >= self.data_loader.n_samples:
            self.current_sample = 0

        t_sec = self.current_sample / self.data_loader.sfreq
        self.frame_changed.emit(self.current_sample, t_sec)


# ==============================================================================
# PHASE 4.02: 2D SCALP TOPOMAP WITH VOLTAGE COLORBAR & SMOOTH INTERPOLATION
# ==============================================================================

class TopomapColorbarWidget(QWidget):
    """
    Sleek Horizontal Voltage Colorbar for 2D Topomap:
    - Features clean title: 'Voltage (µV)'
    - Renders exact gradient using UNIFIED_COLORMAP stops (-V Blue -> Cyan -> Dark -> Coral -> +V Red)
    - Dynamically updates tick values (-V, 0.0, +V) in sync with Topomap Limit slider.
    """
    def __init__(self, v_scale: float = 50.0, parent=None):
        super().__init__(parent)
        self.v_scale = v_scale
        self.setFixedHeight(30)
        self.setFixedWidth(240)
        self.setStyleSheet("background: transparent;")

    def set_v_scale(self, v_scale: float):
        self.v_scale = v_scale
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Title: Voltage (µV)
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor("#8C9BAE"))
        painter.drawText(QRectF(0, 0, w, 11), Qt.AlignCenter, "Voltage (µV)")

        # Gradient Bar
        bar_x = 10
        bar_y = 12
        bar_w = w - 20
        bar_h = 6

        gradient = QLinearGradient(bar_x, bar_y, bar_x + bar_w, bar_y)
        gradient.setColorAt(0.0, QColor(30, 136, 229))   # -V Blue
        gradient.setColorAt(0.25, QColor(0, 229, 255))   # Cyan
        gradient.setColorAt(0.50, QColor(18, 21, 28))    # 0V Neutral Dark
        gradient.setColorAt(0.75, QColor(255, 110, 64))  # Coral Orange
        gradient.setColorAt(1.0, QColor(229, 57, 53))    # +V Crimson Red

        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor("#252D3C"), 1))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)

        # Tick Labels
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor("#A0AEC0"))
        painter.drawText(QRectF(bar_x, bar_y + bar_h + 1, 60, 11), Qt.AlignLeft, f"-{self.v_scale:.0f}")
        painter.drawText(QRectF(bar_x + (bar_w // 2) - 25, bar_y + bar_h + 1, 50, 11), Qt.AlignCenter, "0")
        painter.drawText(QRectF(bar_x + bar_w - 60, bar_y + bar_h + 1, 60, 11), Qt.AlignRight, f"+{self.v_scale:.0f}")


class Topomap2DWidget(QWidget):
    """
    2D Voltage Topographic Map:
    - High-Resolution (128x128) Regularized Multiquadric Interpolation for smooth gradients.
    - Anti-aliased smoothstep circular alpha mask eliminating pixel staircase edges.
    - Phase 3 aesthetics restored:
      * NO filled-in grey background (100% transparent outside & inside).
      * NO circular borders/dots around unselected electrode points (crisp white text labels only).
      * Selected channels feature prominent circular badges with white border.
      * Upward-pointing green triangle nose and crisp white head/ear outlines.
    - Integrated dynamic Voltage Colorbar dock at bottom center.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.v_scale = 50.0

        # Retrieve cached interpolation structures
        self.cache = self.data_loader.interpolation_cache
        self.grid_res = self.cache.grid_res_2d
        self.idw_matrix = self.cache.idw_2d_matrix
        self.alpha_mask = self.cache.alpha_mask_2d
        self.valid_ch_names = self.cache.valid_ch_names_2d

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(0)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        self.plot_widget.setRange(xRange=[-1.20, 1.20], yRange=[-1.20, 1.20])

        # Continuous RGBA heatmap item (Z=1)
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.img_item.setZValue(1)

        self._draw_head_schematic()

        # Scatter item for interactive clicking & selected channel badges (Z=10)
        self.scatter = pg.ScatterPlotItem(hoverable=True)
        self.scatter.sigClicked.connect(self._on_electrode_clicked)
        self.scatter.setZValue(10)
        self.plot_widget.addItem(self.scatter)

        # Crisp centered text labels (Z=15)
        self.label_items: Dict[str, pg.TextItem] = {}
        for ch in self.valid_ch_names:
            x, y = MONTAGE_2D_COORDS[ch]
            lbl = pg.TextItem(ch, color='#FFFFFF', anchor=(0.5, 0.5))
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setZValue(15)
            self.plot_widget.addItem(lbl)
            lbl.setPos(x, y)
            self.label_items[ch] = lbl

        layout.addWidget(self.plot_widget, stretch=1)

        # Bottom Colorbar Dock
        cbar_container = QWidget()
        cbar_container.setFixedHeight(30)
        cb_layout = QHBoxLayout(cbar_container)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setAlignment(Qt.AlignCenter)
        self.colorbar = TopomapColorbarWidget(self.v_scale, self)
        cb_layout.addWidget(self.colorbar)
        layout.addWidget(cbar_container)

        self._refresh_node_styles()

        # Initialize baseline image as 100% transparent (no filled-in grey disk)
        init_rgba = np.zeros((self.grid_res, self.grid_res, 4), dtype=np.uint8)
        self.img_item.setImage(init_rgba, autoLevels=False)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

    def _draw_head_schematic(self):
        theta = np.linspace(0, 2 * np.pi, 250)
        r = 1.0

        # White Circular Head Outline (Z=5)
        head_curve = pg.PlotCurveItem(r * np.cos(theta), r * np.sin(theta), pen=pg.mkPen('#FFFFFF', width=2.0))
        head_curve.setZValue(5)
        self.plot_widget.addItem(head_curve)

        # Upward-pointing Green Nose Triangle (Z=6) matching reference
        nose_x = np.array([-0.12, 0.0, 0.12, -0.12])
        nose_y = np.array([0.98, 1.15, 0.98, 0.98])
        nose_curve = pg.PlotCurveItem(nose_x, nose_y, pen=pg.mkPen('#00FFA3', width=2.5))
        nose_curve.setZValue(6)
        self.plot_widget.addItem(nose_curve)

        # White Left and Right Ears (Z=5)
        ear_l = pg.PlotCurveItem(
            np.array([-0.99, -1.08, -1.08, -0.99]),
            np.array([0.15, 0.08, -0.08, -0.15]),
            pen=pg.mkPen('#FFFFFF', width=2.0)
        )
        ear_l.setZValue(5)
        self.plot_widget.addItem(ear_l)

        ear_r = pg.PlotCurveItem(
            np.array([0.99, 1.08, 1.08, 0.99]),
            np.array([0.15, 0.08, -0.08, -0.15]),
            pen=pg.mkPen('#FFFFFF', width=2.0)
        )
        ear_r.setZValue(5)
        self.plot_widget.addItem(ear_r)

    def set_v_scale(self, v_scale: float):
        self.v_scale = v_scale
        self.colorbar.set_v_scale(v_scale)

    def update_voltage_frame(self, sample_idx: int):
        voltages = self.data_loader.filtered_data[:len(self.valid_ch_names), sample_idx]
        grid_flat = np.dot(self.idw_matrix, voltages)
        v_norm = np.clip((grid_flat + self.v_scale) / (2.0 * self.v_scale), 0.0, 1.0)
        idx = (v_norm * 255.0).astype(np.uint8)

        # Map to RGBA and apply anti-aliased smoothstep alpha mask
        rgba = UNIFIED_LUT_256[idx].reshape((self.grid_res, self.grid_res, 4))
        rgba[:, :, 3] = (rgba[:, :, 3].astype(np.float32) * self.alpha_mask).astype(np.uint8)

        self.img_item.setImage(rgba, autoLevels=False)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

    def _on_electrode_clicked(self, item, points):
        if len(points) == 0:
            return
        ch_name = points[0].data()
        self.data_loader.channel_state.toggle_channel(ch_name)

    def _refresh_node_styles(self):
        spots = []
        for ch in self.valid_ch_names:
            nx, ny = MONTAGE_2D_COORDS[ch]
            is_on = self.data_loader.channel_state.is_selected(ch)
            
            if is_on:
                c_hex = self.data_loader.channel_colors.get(ch, '#00A3FF')
                spots.append({
                    'pos': (nx, ny),
                    'data': ch,
                    'size': 26,
                    'brush': pg.mkBrush(c_hex),
                    'pen': pg.mkPen('#FFFFFF', width=2.5)
                })
            else:
                spots.append({
                    'pos': (nx, ny),
                    'data': ch,
                    'size': 22,
                    'brush': pg.mkBrush(0, 0, 0, 1),
                    'pen': pg.mkPen(0, 0, 0, 0)
                })

            if ch in self.label_items:
                self.label_items[ch].setColor('#FFFFFF')

        self.scatter.setData(spots)


# ==============================================================================
# PHASE 4: 3D CORTICAL TOPOMAP (COLD-STANDBY & RIGOROUS SURFACE PINNING)
# ==============================================================================

class Brain3DWidget(QWidget):
    """
    Phase 4: 3D Topographic Heat Map Engine:
    - Renders data/human-brain.glb (or procedural cortical surface).
    - Decimates mesh (~3,500 vertices) for 60+ FPS rendering.
    - True Anatomical Electrode Placement:
      * Ellipsoid angular projection covering full frontal, temporal, parietal, and occipital lobes.
      * Strict bilateral symmetry (Z <-> -Z) for all 27 pairs.
      * Strict midline pinning (Z = 0.0) bridging the longitudinal fissure.
      * Surface normal / radial outward offset (+0.055) ensuring spheres are pinned to the exterior.
    - Cold-Standby Optimization:
      * Fully stops VTK rendering and IDW math when unselected or dissolved.
      * Disabling electrodes hides both base spheres and selection halos.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.is_active = False

        self.brain_mesh = None
        self.elec_mesh = None
        self.mesh_actor = None
        self.elec_actor = None
        self.highlight_actor = None
        self.scalar_bar_actor = None
        self.pts_per_sphere = 1

        self.show_heatmap = True
        self.show_electrodes = True
        self.show_scalar_bar = False
        self.clim = [-50.0, 50.0]

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3D Viewport Controls Toolbar
        toolbar = QFrame()
        toolbar.setFixedHeight(34)
        toolbar.setStyleSheet("""
            QFrame {
                background: #0B0C10;
                border-bottom: 1px solid #1A1F2C;
            }
            QCheckBox {
                font-size: 10px;
                color: #8C9BAE;
                font-weight: bold;
                spacing: 4px;
            }
            QPushButton {
                background: #151923;
                color: #00FFA3;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1F2737;
                border-color: #00FFA3;
            }
        """)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 2, 10, 2)
        tb_layout.setSpacing(12)

        title_lbl = QLabel("<span style='color: #00FFA3;'>●</span> 3D CORTICAL TOPOMAP")
        title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        tb_layout.addWidget(title_lbl)

        tb_layout.addStretch()

        self.chk_heatmap = QCheckBox("3D Heatmap")
        self.chk_heatmap.setChecked(True)
        self.chk_heatmap.toggled.connect(self._on_toggle_heatmap)
        tb_layout.addWidget(self.chk_heatmap)

        self.chk_elecs = QCheckBox("Electrodes")
        self.chk_elecs.setChecked(True)
        self.chk_elecs.toggled.connect(self._on_toggle_electrodes)
        tb_layout.addWidget(self.chk_elecs)

        self.chk_cbar = QCheckBox("Colorbar")
        self.chk_cbar.setChecked(False)
        self.chk_cbar.toggled.connect(self._on_toggle_colorbar)
        tb_layout.addWidget(self.chk_cbar)

        reset_cam_btn = QPushButton("↺ Top-Down View")
        reset_cam_btn.setToolTip("Reset camera to superior top-down view (frontal cortex up)")
        reset_cam_btn.clicked.connect(self.reset_camera_view)
        tb_layout.addWidget(reset_cam_btn)

        layout.addWidget(toolbar)

        # 3D Viewport
        if PYVISTA_AVAILABLE:
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter)
            self._load_and_prepare_mesh()
            self._precompute_3d_idw()
            self._build_3d_electrodes()
            self.reset_camera_view()
        else:
            lbl = QLabel("PyVista / PyVistaQt not available.\n3D Brain Viewport Disabled.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #7A889B; font-size: 13px; background: #0E0F14;")
            layout.addWidget(lbl)

    def pause_engine(self):
        """Halts all VTK rendering and calculations when 3D is unselected or dissolved."""
        self.is_active = False

    def resume_engine(self):
        """Resumes 3D engine updates upon switching to 3D mode."""
        self.is_active = True
        self.reset_camera_view()

    def _load_and_prepare_mesh(self):
        self.plotter.set_background('#0E0F14')
        model_path = os.path.join("data", "human-brain.glb")
        
        raw_mesh = None
        if os.path.exists(model_path):
            try:
                print(f"[INFO] Loading 3D model from {model_path}...")
                raw_mesh = pv.read(model_path)
                if isinstance(raw_mesh, pv.MultiBlock):
                    raw_mesh = raw_mesh.combine()
                if hasattr(raw_mesh, 'extract_surface'):
                    raw_mesh = raw_mesh.extract_surface()
            except Exception as e:
                print(f"[WARN] Failed to read {model_path}: {e}")
                raw_mesh = None

        if raw_mesh is None:
            print("[INFO] Generating high-fidelity procedural cortical surface...")
            raw_mesh = pv.ParametricEllipsoid(xradius=1.85, yradius=1.25, zradius=1.45)
            raw_mesh.points += 0.05 * np.sin(raw_mesh.points * 8.0)

        center = raw_mesh.center
        raw_mesh.points -= center

        if hasattr(raw_mesh, 'n_points') and raw_mesh.n_points > 5000:
            target_red = max(0.2, min(0.85, 1.0 - (3500.0 / raw_mesh.n_points)))
            try:
                print(f"[INFO] Decimating 3D mesh ({raw_mesh.n_points} vertices) by {target_red*100:.0f}% for 60 FPS performance...")
                raw_mesh = raw_mesh.decimate(target_reduction=target_red)
            except Exception as e:
                print(f"[WARN] Decimation bypassed: {e}")

        raw_mesh.compute_normals(inplace=True)
        self.brain_mesh = raw_mesh
        self.brain_mesh['voltage'] = np.zeros(self.brain_mesh.n_points, dtype=np.float32)

        self.mesh_actor = self.plotter.add_mesh(
            self.brain_mesh,
            scalars='voltage',
            cmap=PYVISTA_CMAP,
            clim=self.clim,
            smooth_shading=True,
            specular=0.35,
            ambient=0.25,
            diffuse=0.85,
            show_scalar_bar=False,
            lighting=True
        )

    def _precompute_3d_idw(self):
        """Precomputes 3D IDW Matrix and caches on SharedInterpolationCache."""
        cache = self.data_loader.interpolation_cache
        if cache.idw_3d_matrix is not None:
            self.idw_3d_matrix = cache.idw_3d_matrix
            self.elec_coord_dict = cache.elec_coords_3d
            self.valid_ch_names = cache.valid_ch_names_3d
            return

        bounds = self.brain_mesh.bounds
        Lx = (bounds[1] - bounds[0]) / 2.0
        Ly = (bounds[3] - bounds[2]) / 2.0
        Lz = (bounds[5] - bounds[4]) / 2.0

        mesh_pts = np.array(self.brain_mesh.points, dtype=np.float32)

        PAIRS = [
            ('Fp1', 'Fp2'), ('AF7', 'AF8'), ('AF3', 'AF4'), ('F7', 'F8'), ('F5', 'F6'),
            ('F3', 'F4'), ('F1', 'F2'), ('FT7', 'FT8'), ('FC5', 'FC6'), ('FC3', 'FC4'),
            ('FC1', 'FC2'), ('T7', 'T8'), ('C5', 'C6'), ('C3', 'C4'), ('C1', 'C2'),
            ('T9', 'T10'), ('TP7', 'TP8'), ('CP5', 'CP6'), ('CP3', 'CP4'), ('CP1', 'CP2'),
            ('P7', 'P8'), ('P5', 'P6'), ('P3', 'P4'), ('P1', 'P2'), ('PO7', 'PO8'),
            ('PO3', 'PO4'), ('O1', 'O2')
        ]
        MIDLINE = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'Iz']

        self.elec_coord_dict = {}
        self.valid_ch_names = []

        def get_ellipsoid_target(ch_name: str) -> np.ndarray:
            x2, y2 = MONTAGE_2D_COORDS[ch_name]
            r = math.sqrt(x2**2 + y2**2)
            theta = min(math.pi * 0.49, r * (math.pi * 0.49 / 0.94))
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            ux = sin_t * (y2 / r) if r > 1e-4 else 0.0
            uz = sin_t * (x2 / r) if r > 1e-4 else 0.0
            uy = cos_t
            return np.array([Lx * ux * 1.05, Ly * uy * 1.05, Lz * uz * 1.05], dtype=np.float32)

        # 1. Place 27 Bilateral Symmetric Pairs
        for l_ch, r_ch in PAIRS:
            target_l = get_ellipsoid_target(l_ch)
            target_r = get_ellipsoid_target(r_ch)

            id_l = self.brain_mesh.find_closest_point(target_l)
            id_r = self.brain_mesh.find_closest_point(target_r)

            pt_l = mesh_pts[id_l]
            pt_r = mesh_pts[id_r]

            x_sym = (pt_l[0] + pt_r[0]) / 2.0
            y_sym = (pt_l[1] + pt_r[1]) / 2.0
            z_mag = (abs(pt_l[2]) + abs(pt_r[2])) / 2.0
            z_mag = max(z_mag, abs(MONTAGE_2D_COORDS[r_ch][0]) * Lz * 0.85)

            pos_r = np.array([x_sym, y_sym, +z_mag], dtype=np.float32)
            pos_l = np.array([x_sym, y_sym, -z_mag], dtype=np.float32)

            u_r = pos_r / (np.linalg.norm(pos_r) + 1e-6)
            u_l = pos_l / (np.linalg.norm(pos_l) + 1e-6)

            self.elec_coord_dict[r_ch] = pos_r + u_r * 0.055
            self.elec_coord_dict[l_ch] = pos_l + u_l * 0.055

        # 2. Place 10 Midline Channels bridging the longitudinal fissure
        for ch in MIDLINE:
            target = get_ellipsoid_target(ch)
            target[2] = 0.0
            best_id = self.brain_mesh.find_closest_point(target)
            pt = mesh_pts[best_id].copy()

            near_x = (np.abs(mesh_pts[:, 0] - pt[0]) < (0.08 * Lx)) & (np.abs(mesh_pts[:, 2]) < (0.22 * Lz))
            if np.any(near_x):
                pt[1] = max(pt[1], float(np.max(mesh_pts[near_x, 1])))

            pt[2] = 0.0
            u = pt / (np.linalg.norm(pt) + 1e-6)
            self.elec_coord_dict[ch] = pt + u * 0.055

        elec_pts = []
        for ch in STANDARD_64_CHANNELS:
            if ch in self.elec_coord_dict:
                elec_pts.append(self.elec_coord_dict[ch])
                self.valid_ch_names.append(ch)

        elec_arr = np.array(elec_pts, dtype=np.float32)

        diff = mesh_pts[:, np.newaxis, :] - elec_arr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-4
        weights = 1.0 / (dist ** 2.5)

        inferior_mask = mesh_pts[:, 1] < (-0.25 * Ly)
        weights[inferior_mask, :] *= 0.25

        weights /= np.sum(weights, axis=1, keepdims=True)
        self.idw_3d_matrix = weights.astype(np.float32)

        # Store in shared cache
        cache.idw_3d_matrix = self.idw_3d_matrix
        cache.elec_coords_3d = self.elec_coord_dict
        cache.valid_ch_names_3d = self.valid_ch_names
        print(f"[INFO] Precomputed 3D IDW Matrix: {self.idw_3d_matrix.shape} ({self.brain_mesh.n_points} vertices, 64 channels).")

    def _build_3d_electrodes(self):
        coords = np.array([self.elec_coord_dict[ch] for ch in self.valid_ch_names], dtype=np.float32)
        cloud = pv.PolyData(coords)

        sphere_geom = pv.Sphere(radius=0.055, theta_resolution=10, phi_resolution=10)
        self.pts_per_sphere = sphere_geom.n_points

        self.elec_mesh = cloud.glyph(geom=sphere_geom, scale=False, orient=False)
        self.elec_mesh['voltage'] = np.zeros(self.elec_mesh.n_points, dtype=np.float32)

        self.elec_actor = self.plotter.add_mesh(
            self.elec_mesh,
            scalars='voltage',
            cmap=PYVISTA_CMAP,
            clim=self.clim,
            smooth_shading=True,
            specular=0.8,
            show_scalar_bar=False,
            lighting=True
        )

    def reset_camera_view(self):
        """Top-down superior view looking straight down (+Y down), frontal cortex (+X) up."""
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        self.plotter.camera_position = [(0.0, 5.6, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def update_voltage_frame(self, voltages_64: np.ndarray, v_scale: float = 50.0):
        if not self.is_active or not PYVISTA_AVAILABLE or self.brain_mesh is None:
            return

        if self.show_heatmap and self.idw_3d_matrix is not None:
            cortex_scalars = np.dot(self.idw_3d_matrix, voltages_64[:len(self.valid_ch_names)])
            self.brain_mesh['voltage'] = cortex_scalars

        if self.show_electrodes and self.elec_mesh is not None:
            expanded_v = np.repeat(voltages_64[:len(self.valid_ch_names)], self.pts_per_sphere)
            self.elec_mesh['voltage'] = expanded_v

        self.plotter.render()

    def set_clim(self, v_scale: float):
        self.clim = [-float(v_scale), float(v_scale)]
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        if self.mesh_actor:
            mapper = self.mesh_actor.GetMapper()
            if mapper:
                mapper.SetScalarRange(self.clim[0], self.clim[1])
        if self.elec_actor:
            mapper = self.elec_actor.GetMapper()
            if mapper:
                mapper.SetScalarRange(self.clim[0], self.clim[1])
        if self.is_active:
            self.plotter.render()

    def set_selected_channels(self, selected_set: Set[str]):
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return

        if self.highlight_actor is not None:
            self.plotter.remove_actor(self.highlight_actor)
            self.highlight_actor = None

        if len(selected_set) > 0 and self.show_electrodes:
            sel_pts = []
            for ch in selected_set:
                if ch in self.elec_coord_dict:
                    sel_pts.append(self.elec_coord_dict[ch])
            if len(sel_pts) > 0:
                cloud = pv.PolyData(np.array(sel_pts, dtype=np.float32))
                ring_geom = pv.Sphere(radius=0.08, theta_resolution=14, phi_resolution=14)
                halo_mesh = cloud.glyph(geom=ring_geom, scale=False, orient=False)
                self.highlight_actor = self.plotter.add_mesh(
                    halo_mesh,
                    color='#FFFFFF',
                    style='wireframe',
                    line_width=3,
                    specular=1.0,
                    lighting=False
                )
        if self.is_active:
            self.plotter.render()

    def _on_toggle_heatmap(self, checked: bool):
        self.show_heatmap = checked
        if not PYVISTA_AVAILABLE or self.mesh_actor is None:
            return
        mapper = self.mesh_actor.GetMapper()
        if mapper:
            mapper.SetScalarVisibility(checked)
        if self.is_active:
            self.plotter.render()

    def _on_toggle_electrodes(self, checked: bool):
        self.show_electrodes = checked
        if not PYVISTA_AVAILABLE:
            return
        if self.elec_actor is not None:
            self.elec_actor.SetVisibility(checked)
        if self.highlight_actor is not None:
            self.highlight_actor.SetVisibility(checked)
        if self.is_active:
            self.plotter.render()

    def _on_toggle_colorbar(self, checked: bool):
        self.show_scalar_bar = checked
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        if checked:
            self.scalar_bar_actor = self.plotter.add_scalar_bar(
                title="Voltage (µV)",
                n_labels=5,
                color='#DDE2EB',
                label_font_size=10,
                title_font_size=11,
                interactive=False
            )
        else:
            self.plotter.remove_scalar_bar()
        if self.is_active:
            self.plotter.render()


# ==============================================================================
# TOPOMAP CONTAINER (TOGGLEABLE 2D MAP / 3D BRAIN WITH ZERO BACKGROUND OVERHEAD)
# ==============================================================================

class TopomapContainerWidget(QWidget):
    """
    Unified Topomap Viewport:
    - Hosts a QStackedWidget switching between Page 0 (2D Topomap) and Page 1 (3D Brain).
    - Segmented Mode Selector: [ 2D Topomap ] | [ 3D Brain ].
    - Lazy 3D Initialization: 3D engine is only allocated upon first click of [3D Brain].
    - Zero background overhead: When 2D is active, 3D engine is completely stopped.
    """
    mode_changed = pyqtSignal(str)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.current_mode = "2D"
        self.widget_3d: Optional[Brain3DWidget] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header Bar matching reference
        header = QFrame()
        header.setFixedHeight(34)
        header.setStyleSheet("""
            QFrame {
                background: #0B0C10;
                border-bottom: 1px solid #1A1F2C;
            }
            QPushButton {
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1E2431;
                color: #E2E8F0;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 2, 10, 2)
        h_layout.setSpacing(8)

        self.title_lbl = QLabel("2D VOLTAGE TOPOMAP (64 CHANNELS)")
        self.title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(self.title_lbl)

        # Active channels badge
        self.active_badge = QLabel("0 / 64 Active")
        self.active_badge.setStyleSheet("""
            background: #0D332D;
            color: #00FFA3;
            border: 1px solid #00FFA3;
            border-radius: 9px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: bold;
        """)
        h_layout.addWidget(self.active_badge)

        h_layout.addStretch()

        # [Select All] and [Clear] buttons
        self.btn_select_all = QPushButton("Select All")
        self.btn_clear = QPushButton("Clear")
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_clear.clicked.connect(self._on_clear)
        h_layout.addWidget(self.btn_select_all)
        h_layout.addWidget(self.btn_clear)

        # Segmented Mode Switcher
        self.btn_2d = QPushButton("2D Topomap")
        self.btn_3d = QPushButton("3D Brain")
        self.btn_2d.clicked.connect(lambda: self.set_mode("2D"))
        self.btn_3d.clicked.connect(lambda: self.set_mode("3D"))

        h_layout.addWidget(self.btn_2d)
        h_layout.addWidget(self.btn_3d)

        layout.addWidget(header)

        # Stacked Widget
        self.stack = QStackedWidget()
        self.widget_2d = Topomap2DWidget(self.data_loader, self)
        self.stack.addWidget(self.widget_2d)  # Index 0

        # Placeholder for lazy 3D loading (Index 1)
        self.placeholder_3d = QWidget()
        p_layout = QVBoxLayout(self.placeholder_3d)
        p_lbl = QLabel("Click [3D Brain] to initialize 3D Cortical Engine...")
        p_lbl.setAlignment(Qt.AlignCenter)
        p_lbl.setStyleSheet("color: #4A5568; font-size: 11px;")
        p_layout.addWidget(p_lbl)
        self.stack.addWidget(self.placeholder_3d)

        layout.addWidget(self.stack, stretch=1)
        self._refresh_button_styles()

    def set_mode(self, mode: str):
        if mode == self.current_mode:
            return
        self.current_mode = mode

        if mode == "2D":
            self.stack.setCurrentIndex(0)
            self.title_lbl.setText("2D VOLTAGE TOPOMAP (64 CHANNELS)")
            self.btn_select_all.setVisible(True)
            self.btn_clear.setVisible(True)
            if self.widget_3d is not None:
                self.widget_3d.pause_engine()
        else:
            if self.widget_3d is None:
                print("[INFO] Cold-starting 3D Cortical Engine...")
                self.widget_3d = Brain3DWidget(self.data_loader, self)
                self.stack.removeWidget(self.placeholder_3d)
                self.stack.addWidget(self.widget_3d)
                self.widget_3d.set_selected_channels(self.data_loader.channel_state.selected_channels)
                self.widget_3d.set_clim(self.widget_2d.v_scale)

            self.stack.setCurrentIndex(1)
            self.title_lbl.setText("3D CORTICAL TOPOMAP (PHASE 4 ENGINE)")
            self.btn_select_all.setVisible(False)
            self.btn_clear.setVisible(False)
            self.widget_3d.resume_engine()

        self._refresh_button_styles()
        self.mode_changed.emit(self.current_mode)

    def update_active_badge(self, count: int):
        self.active_badge.setText(f"{count} / 64 Active")

    def _on_select_all(self):
        self.data_loader.channel_state.select_all()

    def _on_clear(self):
        self.data_loader.channel_state.clear_all()

    def _refresh_button_styles(self):
        active_style = """
            background: #1B2B24;
            color: #00FFA3;
            border: 1px solid #00FFA3;
            border-radius: 3px;
            padding: 3px 10px;
            font-size: 10px;
            font-weight: bold;
        """
        inactive_style = """
            background: #14171E;
            color: #8C9BAE;
            border: 1px solid #232B3B;
            border-radius: 3px;
            padding: 3px 10px;
            font-size: 10px;
            font-weight: bold;
        """
        self.btn_2d.setStyleSheet(active_style if self.current_mode == "2D" else inactive_style)
        self.btn_3d.setStyleSheet(active_style if self.current_mode == "3D" else inactive_style)


# ==============================================================================
# ANALYSIS WORKSTATION PANEL (TOP-LEFT SECTION - PREPARED FOR PHASE 5)
# ==============================================================================

class AnalysisPanelWidget(QWidget):
    """
    Analysis Workstation Panel (Top-Left Section):
    Replaces former top-left position. Clean, modern placeholder panel structured
    for Phase 5 analytical, spectral, connectivity, and epoch decomposition modules.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
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

        title_lbl = QLabel("<span style='color: #00E5FF;'>●</span> ANALYSIS WORKSTATION")
        title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(title_lbl)

        h_layout.addStretch()

        status_lbl = QLabel("Phase 5 Integration Sandbox • Standby")
        status_lbl.setStyleSheet("color: #7A889B; font-size: 10px;")
        h_layout.addWidget(status_lbl)

        layout.addWidget(header)

        # Body Canvas
        canvas = QFrame()
        canvas.setStyleSheet("background: #0E0F14; border: none;")
        c_layout = QVBoxLayout(canvas)
        c_layout.setAlignment(Qt.AlignCenter)

        placeholder_title = QLabel("ANALYSIS WORKSTATION")
        placeholder_title.setStyleSheet("color: #2D3748; font-size: 16px; font-weight: 800; letter-spacing: 2px;")
        placeholder_title.setAlignment(Qt.AlignCenter)

        placeholder_sub = QLabel(
            "Spectral Decomposition • Wavelet Time-Frequency • Microstates • Functional Connectivity\n"
            "Allocated for Phase 5 Advanced Signal Analysis & Processing Engine"
        )
        placeholder_sub.setStyleSheet("color: #4A5568; font-size: 11px; line-height: 18px;")
        placeholder_sub.setAlignment(Qt.AlignCenter)

        c_layout.addWidget(placeholder_title)
        c_layout.addSpacing(6)
        c_layout.addWidget(placeholder_sub)

        layout.addWidget(canvas, stretch=1)


# ==============================================================================
# CASCADING WAVEFORMS WIDGET (DISPLAYED IN MIDDLE/BOTTOM WORKSPACE)
# ==============================================================================

class WaveformsWidget(QWidget):
    """
    Cascading Multi-Channel Raw EEG Waveforms:
    - Renders ONLY user-selected channels (starts with 0 active on launch).
    - Features dedicated header: '● RAW EEG WAVEFORMS' + active channel badge.
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

        cur_t = current_sample / self.data_loader.sfreq
        t_start = max(0.0, cur_t - self.window_sec)
        t_end = cur_t

        s_start = int(t_start * self.data_loader.sfreq)
        s_end = int(t_end * self.data_loader.sfreq)
        n_pts = s_end - s_start

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
                chunk = self.data_loader.filtered_data[ch_idx, s_start:s_end]
                y_offset = idx * y_spacing
                scaled_y = (chunk * self.gain) + y_offset
                self.curve_items[ch].setData(self.cached_t_axis, scaled_y)


# ==============================================================================
# SPOTIFY PLAYBACK BAR WITH MINI-MAP TIMELINE NAVIGATION
# ==============================================================================

class SpotifyTimelineWidget(pg.PlotWidget):
    """
    Spotify-Style Interactive Timeline with GFP Mini-Map Waveform Navigation:
    - Renders the continuous Global Field Power (GFP) envelope across the entire dataset.
    - Overlays colored stimulus event regions (T0 Rest, T1 Left Fist, T2 Right Fist).
    - Scrubbing moves the white playhead line and smoothly seeks playback.
    """
    seek_requested = pyqtSignal(float)

    def __init__(self, data_loader: EEGDataLoader, parent=None, engine=None):
        super().__init__(parent=parent)
        self.data_loader = data_loader
        self.engine = engine if engine is not None else getattr(parent, 'engine', None)
        self.is_dragging = False

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
            upper_curve = pg.PlotCurveItem(t_axis, gfp_y, fillLevel=0.0, brush=pg.mkBrush('#142232'), pen=pg.mkPen('#21344B', width=1.0))
            upper_curve.setZValue(1)
            self.addItem(upper_curve)

            lower_curve = pg.PlotCurveItem(t_axis, -gfp_y, fillLevel=0.0, brush=pg.mkBrush('#142232'), pen=pg.mkPen('#21344B', width=1.0))
            lower_curve.setZValue(1)
            self.addItem(lower_curve)

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

        # White playhead cursor line (Z=5)
        self.playhead_line = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False,
            pen=pg.mkPen('#FFFFFF', width=2.0)
        )
        self.playhead_line.setZValue(5)
        self.addItem(self.playhead_line)

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
    - Center Top: Step Back (⏮), White Circular Play/Pause (▶), Step Forward (⏭)
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

        self.btn_prev = QPushButton("⏮")
        self.btn_prev.setFixedSize(28, 28)
        self.btn_prev.setToolTip("Step Back 1s")
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #8C9BAE;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                color: #FFFFFF;
            }
        """)
        self.btn_prev.clicked.connect(lambda: self.engine.step_backward(1.0))

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(34, 34)
        self.btn_play.setToolTip("Play / Pause (Space)")
        self.btn_play.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                color: #000000;
                font-size: 14px;
                border-radius: 17px;
                font-weight: bold;
                padding-left: 2px;
            }
            QPushButton:hover {
                background: #1DB954;
                color: #000000;
            }
        """)
        self.btn_play.clicked.connect(self.engine.toggle_play)

        self.btn_next = QPushButton("⏭")
        self.btn_next.setFixedSize(28, 28)
        self.btn_next.setToolTip("Step Forward 1s")
        self.btn_next.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #8C9BAE;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                color: #FFFFFF;
            }
        """)
        self.btn_next.clicked.connect(lambda: self.engine.step_forward(1.0))

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
                }
                QPushButton:hover {
                    background: #23D260;
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
                }
                QPushButton:hover {
                    background: #1DB954;
                    color: #000000;
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

        tag = QLabel("RESEARCH SUITE v4.02")
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
    Phase 4.02 Collapsible Control Sidebar:
    - Features a header collapse/expand toggle button (◀ / ▶).
    - Smoothly transitions between expanded (240px) and collapsed (36px).
    - Shrinking expands the workspace horizontally for maximum topomap & waveform area.
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
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1F2533;
                color: #00FFA3;
                border-color: #00FFA3;
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
        body_layout.setSpacing(14)

        # Speed Selector
        speed_box = QHBoxLayout()
        speed_lbl = QLabel("Speed:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x", "4.0x"])
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

        body_layout.addSpacing(6)

        # DSP Filter Settings
        filter_title = QLabel("DSP BANDPASS FILTERS")
        filter_title.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: 900; letter-spacing: 1px;")
        body_layout.addWidget(filter_title)

        f_grid = QGridLayout()
        f_grid.setSpacing(6)

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
                padding: 6px;
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

        help_box = QLabel("Navigation:\n• Space: Play/Pause\n• F4: Toggle Sidebar\n• Scrub timeline to jump")
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
        speed_map = {"0.5x": 0.5, "1.0x": 1.0, "1.5x": 1.5, "2.0x": 2.0, "4.0x": 4.0}
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

class NeuromapMainWindow(QMainWindow):
    """
    Neuromap Main Dashboard Window:
    - Layout Swapped:
      * Top-Left: Analysis Workstation Panel (clean placeholder for Phase 5 modules)
      * Top-Right: Topomap Viewport (Toggle between 2D Topomap and 3D Brain)
    - Middle/Bottom: Cascading Raw Waveforms
    - Bottom Dock: Spotify Playback Bar with Mini-Map
    - Zero-Overhead Dissolved Section Culling
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("neuromap - Professional Closed-Loop EEG Dashboard")
        self.resize(1920, 1200)

        self.data_loader = EEGDataLoader()
        self.data_loader.load_dataset()
        self.engine = PlaybackEngine(self.data_loader)

        self._init_ui()
        self._apply_qss()
        self._connect_signals()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header
        self.header = HeaderWidget(self.data_loader, self)
        self.header.export_requested.connect(self._on_export_data)
        root_layout.addWidget(self.header)

        # Workspace (Collapsible Sidebar + Vertical Splitter)
        main_workspace = QHBoxLayout()
        main_workspace.setContentsMargins(0, 0, 0, 0)
        main_workspace.setSpacing(0)

        self.sidebar = ControlSidebarWidget(self.data_loader, self.engine, self)
        main_workspace.addWidget(self.sidebar)

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setHandleWidth(4)

        # Top Row (Horizontal Splitter: Analysis Workstation on LEFT, Topomap Viewport on RIGHT)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setHandleWidth(4)

        # Top-Left: Analysis Workstation Panel (Swapped to Left)
        self.analysis_panel = AnalysisPanelWidget(self)
        self.top_splitter.addWidget(self.analysis_panel)

        # Top-Right: Toggleable Topomap Container (Swapped to Right)
        self.topomap_container = TopomapContainerWidget(self.data_loader, self)
        self.top_splitter.addWidget(self.topomap_container)

        self.top_splitter.setSizes([960, 960])
        self.main_splitter.addWidget(self.top_splitter)

        # Middle/Bottom Section: Cascading Waveforms
        self.waveforms_widget = WaveformsWidget(self.data_loader, self)
        self.main_splitter.addWidget(self.waveforms_widget)
        self.main_splitter.setSizes([560, 480])

        main_workspace.addWidget(self.main_splitter, stretch=1)
        root_layout.addLayout(main_workspace, stretch=1)

        # Bottom Dock: Spotify Playback Bar (Phase 2 Format + Mini-Map)
        self.playback_bar = SpotifyPlaybackBar(self.data_loader, self.engine, self)
        root_layout.addWidget(self.playback_bar)

        # Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background: #060709; color: #7A889B; border-top: 1px solid #14171E;")
        self.status_label = QLabel("Status: System Ready | Wall-Clock Synced Playback Active")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Progress: Ready (100%)")
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #14171E;
                color: #000000;
                font-weight: bold;
                font-size: 9px;
                border-radius: 3px;
                border: 1px solid #202632;
            }
            QProgressBar::chunk {
                background: #1DB954;
                border-radius: 3px;
            }
        """)

        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)

        # Shortcuts
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.engine.toggle_play)
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_F4), self, activated=self.sidebar.toggle_collapse)

    def _apply_qss(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0E0F14;
                color: #DDE2EB;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QSplitter::handle {
                background: #181C26;
            }
            QSplitter::handle:hover {
                background: #1DB954;
            }
            QCheckBox {
                spacing: 6px;
                font-size: 11px;
                color: #B4C0D0;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #323C4E;
                border-radius: 3px;
                background: #12151C;
            }
            QCheckBox::indicator:checked {
                background: #1DB954;
                border-color: #1DB954;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #252D3C;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #1DB954;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #1DB954;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QLineEdit {
                background: #141720;
                border: 1px solid #2B3444;
                border-radius: 3px;
                color: #E2E8F0;
                padding: 2px 4px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QComboBox {
                background: #141720;
                border: 1px solid #2B3444;
                border-radius: 4px;
                padding: 4px 8px;
                color: #E2E8F0;
                font-size: 11px;
            }
        """)

    def _connect_signals(self):
        self.engine.frame_changed.connect(self._on_frame_update)
        self.engine.state_changed.connect(self.playback_bar.update_playback_state)

        # Centralized Channel State Listener
        self.data_loader.channel_state.selection_changed.connect(self._on_central_channel_selection_changed)

        self.sidebar.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.sidebar.win_slider.valueChanged.connect(self._on_window_changed)
        self.sidebar.topo_slider.valueChanged.connect(self._on_topo_scale_changed)

        # Splitter motion listeners for dissolved section culling
        self.top_splitter.splitterMoved.connect(self._on_splitter_adjusted)
        self.main_splitter.splitterMoved.connect(self._on_splitter_adjusted)

    def _is_widget_active(self, widget: QWidget) -> bool:
        """Determines if a widget section is visible or dissolved/collapsed."""
        if widget is None or widget.isHidden():
            return False
        if widget.width() < 25 or widget.height() < 25:
            return False
        return True

    def _on_splitter_adjusted(self, pos: int, index: int):
        # Pause 3D engine if topomap viewport is dragged closed
        if self.topomap_container.current_mode == "3D":
            if self._is_widget_active(self.topomap_container):
                if self.topomap_container.widget_3d is not None and not self.topomap_container.widget_3d.is_active:
                    self.topomap_container.widget_3d.resume_engine()
            else:
                if self.topomap_container.widget_3d is not None and self.topomap_container.widget_3d.is_active:
                    self.topomap_container.widget_3d.pause_engine()

    def _on_frame_update(self, current_sample: int, current_time: float):
        topomap_visible = self._is_widget_active(self.topomap_container)
        waveforms_visible = self._is_widget_active(self.waveforms_widget)

        # 1. Update Topomap Viewport ONLY if section is visible (not dissolved)
        if topomap_visible:
            if self.topomap_container.current_mode == "2D":
                self.topomap_container.widget_2d.update_voltage_frame(current_sample)
            else:
                if self.topomap_container.widget_3d is not None and self.topomap_container.widget_3d.is_active:
                    voltages_64 = self.data_loader.filtered_data[:len(self.topomap_container.widget_3d.valid_ch_names), current_sample]
                    self.topomap_container.widget_3d.update_voltage_frame(voltages_64, self.topomap_container.widget_2d.v_scale)
        else:
            # If 3D topomap is dissolved/hidden, pause VTK rendering immediately
            if self.topomap_container.widget_3d is not None and self.topomap_container.widget_3d.is_active:
                self.topomap_container.widget_3d.pause_engine()

        # 2. Update Cascading Waveforms ONLY if section is visible (not dissolved)
        if waveforms_visible:
            self.waveforms_widget.update_frame(current_sample)

        # 3. Update Spotify Playback Bar (Always active during playback)
        self.playback_bar.update_frame(current_sample, current_time)

    def _on_central_channel_selection_changed(self, selected_set: Set[str]):
        selected_list = sorted(list(selected_set))
        # Update 2D Topomap nodes
        self.topomap_container.widget_2d._refresh_node_styles()
        # Update 3D Topomap halos if initialized
        if self.topomap_container.widget_3d is not None:
            self.topomap_container.widget_3d.set_selected_channels(selected_set)
        # Update Waveforms list
        self.waveforms_widget.set_active_channels(selected_list)
        # Update Toolbar badge
        self.topomap_container.update_active_badge(len(selected_list))
        # Update Status Bar
        self.status_label.setText(f"Status: {len(selected_list)} of 64 channels selected ({', '.join(selected_list[:6]) if selected_list else 'None'})")

    def _on_gain_changed(self, val: int):
        self.waveforms_widget.gain = float(val) / 10.0

    def _on_window_changed(self, val: int):
        self.waveforms_widget.update_window(float(val))

    def _on_topo_scale_changed(self, val: int):
        v = float(val)
        self.topomap_container.widget_2d.set_v_scale(v)
        if self.topomap_container.widget_3d is not None:
            self.topomap_container.widget_3d.set_clim(v)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _on_export_data(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Filtered EEG", "filtered_eeg_export.csv", "CSV Files (*.csv)")
        if path:
            self.status_label.setText("Exporting filtered dataset to CSV...")
            QApplication.processEvents()
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Sample_Index", "Time_Sec"] + self.data_loader.channel_names)
                for i in range(min(5000, self.data_loader.n_samples)):
                    t_val = i / self.data_loader.sfreq
                    row = [i, f"{t_val:.4f}"] + [f"{self.data_loader.filtered_data[c, i]:.2f}" for c in range(64)]
                    writer.writerow(row)
            self.status_label.setText(f"Status: Export completed to {path}")


# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = NeuromapMainWindow()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()