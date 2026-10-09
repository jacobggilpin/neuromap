"""
neuromap - Professional Closed-Loop EEG Visualization Dashboard
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Phase 4 Final Architecture:
- Top-Left: Dual Topomap Container with Mode Toggle [2D Topomap] | [3D Brain]
  * 2D Mode: High-performance RGBA transparent voltage heatmap + 64 electrode selection nodes
  * 3D Mode: Phase 4 3D Cortical Heatmap with surface-pinned, symmetrically centered electrodes
  * Zero-overhead cold standby: When 2D is active, 3D rendering and IDW math are completely halted
  * Disabling 3D electrodes hides both base spheres and selection halos
- Top-Right: Analysis Workstation Panel (Clean placeholder for Phase 5 analytical modules)
- Middle/Bottom: Cascading Waveforms with restored header and jitter-free rendering
- Bottom Dock: Phase 2 Spotify Playback Bar (EEG badge, centered controls, timeline, legend dots)
- Restored Section Headers for 2D Topomap and Raw Waveforms
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
# DATA LOADER & SIGNAL PROCESSING
# ==============================================================================

class EEGDataLoader:
    """
    Loads MNE PhysioNet Motor Movement/Imagery dataset or generates high-fidelity
    synthetic 64-channel EEG with alpha rhythms and motor imagery desynchronization.
    Applies zero-phase Butterworth bandpass & notch filters.
    """
    def __init__(self):
        self.sfreq = 160.0
        self.channel_names: List[str] = STANDARD_64_CHANNELS.copy()
        self.raw_data: np.ndarray = np.zeros((64, 20000), dtype=np.float32)
        self.filtered_data: np.ndarray = np.zeros((64, 20000), dtype=np.float32)
        self.events: List[StimulusEvent] = []
        self.duration = 125.0
        self.n_samples = 20000
        self.subject_id = "S001"
        self.file_type = "EDF+ (PhysioNet)"
        self.loaded_status = "Synthesizing"

        self.hp_freq = 1.0
        self.lp_freq = 40.0
        self.notch_50 = False
        self.notch_60 = True

        self.channel_colors: Dict[str, str] = {
            ch: PALETTE_COLORS[i % len(PALETTE_COLORS)]
            for i, ch in enumerate(self.channel_names)
        }

    def load_dataset(self) -> bool:
        success = False
        if MNE_AVAILABLE:
            try:
                print("[INFO] Fetching PhysioNet Subject 1, Run 4...")
                mne.set_log_level('WARNING')
                edf_files = eegbci.load_data(1, [4], update_path=False)
                raw = mne.io.read_raw_edf(edf_files[0], preload=True)

                std_upper_map = {ch.upper(): ch for ch in STANDARD_64_CHANNELS}
                raw_ch_map = {}
                for ch in raw.ch_names:
                    clean = ch.strip('.').upper()
                    if clean in std_upper_map:
                        raw_ch_map[ch] = std_upper_map[clean]
                raw.rename_channels(raw_ch_map)

                self.sfreq = float(raw.info['sfreq'])
                
                self.channel_names = STANDARD_64_CHANNELS.copy()
                data = np.zeros((64, raw.n_times), dtype=np.float32)
                for i, ch in enumerate(self.channel_names):
                    if ch in raw.ch_names:
                        ch_idx = raw.ch_names.index(ch)
                        data[i] = raw.get_data()[ch_idx]
                    else:
                        data[i] = np.random.randn(raw.n_times) * 2e-6
                
                self.raw_data = (data * 1e6).astype(np.float32)  # Convert to µV
                self.n_samples = self.raw_data.shape[1]
                self.duration = self.n_samples / self.sfreq
                self.subject_id = "S001"
                self.file_type = "PhysioNet EDF+"
                self.loaded_status = "Online MNE"

                # Extract MNE Annotations
                self.events.clear()
                annots = raw.annotations
                if annots and len(annots) > 0:
                    for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
                        eid = str(desc).strip()
                        label = EVENT_COLOR_MAP.get(eid, {}).get('name', f"Event {eid}")
                        s_idx = int(onset * self.sfreq)
                        e_idx = int((onset + dur) * self.sfreq)
                        self.events.append(StimulusEvent(
                            event_id=eid, label=label,
                            start_time=float(onset), end_time=float(onset + dur),
                            start_sample=s_idx, end_sample=e_idx
                        ))
                success = True
                print(f"[SUCCESS] Loaded {len(self.channel_names)} channels, {self.n_samples} samples ({self.duration:.1f}s @ {self.sfreq:.0f}Hz).")
            except Exception as e:
                print(f"[WARN] MNE loading fallback to synthetic ({e})...")
                success = False

        if not success:
            self._generate_synthetic_eeg()
        
        self.apply_filters()
        return True

    def _generate_synthetic_eeg(self):
        self.sfreq = 160.0
        self.duration = 125.0
        self.n_samples = int(self.duration * self.sfreq)
        self.channel_names = STANDARD_64_CHANNELS.copy()
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        self.raw_data = np.zeros((64, self.n_samples), dtype=np.float32)

        for i in range(64):
            pink = np.cumsum(np.random.randn(self.n_samples)) * 0.08
            alpha = 12.0 * np.sin(2 * np.pi * 10.2 * t + np.random.uniform(0, 2 * np.pi))
            beta = 5.0 * np.sin(2 * np.pi * 22.0 * t + np.random.uniform(0, 2 * np.pi))
            self.raw_data[i] = (pink + alpha + beta + np.random.randn(self.n_samples) * 3.0).astype(np.float32)

        self.events.clear()
        event_cycle = [('T0', 4.0), ('T1', 4.0), ('T0', 4.0), ('T2', 4.0)]
        curr_t = 2.0
        while curr_t < self.duration - 5.0:
            for eid, dur in event_cycle:
                if curr_t + dur >= self.duration:
                    break
                s_idx = int(curr_t * self.sfreq)
                e_idx = int((curr_t + dur) * self.sfreq)
                label = EVENT_COLOR_MAP.get(eid, {}).get('name', eid)
                self.events.append(StimulusEvent(
                    event_id=eid, label=label,
                    start_time=curr_t, end_time=curr_t + dur,
                    start_sample=s_idx, end_sample=e_idx
                ))
                if eid == 'T1':
                    c4_idx = self.channel_names.index('C4') if 'C4' in self.channel_names else 32
                    self.raw_data[c4_idx, s_idx:e_idx] *= 0.4
                elif eid == 'T2':
                    c3_idx = self.channel_names.index('C3') if 'C3' in self.channel_names else 28
                    self.raw_data[c3_idx, s_idx:e_idx] *= 0.4

                curr_t += dur

        self.subject_id = "S001_SYNTH"
        self.file_type = "Synthetic Motor EEG"
        self.loaded_status = "Offline Fallback"

    def apply_filters(self):
        data = self.raw_data.copy()
        nyq = 0.5 * self.sfreq

        low = max(0.1, self.hp_freq) / nyq
        high = min(self.sfreq * 0.49, self.lp_freq) / nyq
        if low < high and high < 1.0:
            b, a = scipy.signal.butter(4, [low, high], btype='bandpass')
            data = scipy.signal.filtfilt(b, a, data, axis=1)

        if self.notch_60 and (60.0 < nyq):
            b_notch, a_notch = scipy.signal.iirnotch(60.0, 30.0, self.sfreq)
            data = scipy.signal.filtfilt(b_notch, a_notch, data, axis=1)

        if self.notch_50 and (50.0 < nyq):
            b_notch, a_notch = scipy.signal.iirnotch(50.0, 30.0, self.sfreq)
            data = scipy.signal.filtfilt(b_notch, a_notch, data, axis=1)

        self.filtered_data = data.astype(np.float32)

    def get_window_data(self, current_sample: int, window_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        win_samples = int(window_sec * self.sfreq)
        start_idx = current_sample - win_samples
        end_idx = current_sample

        t_axis = np.linspace(-window_sec, 0.0, win_samples, endpoint=False)
        
        if start_idx < 0:
            n_pad = -start_idx
            valid_slice = self.filtered_data[:, 0:max(1, end_idx)]
            pad_block = np.repeat(self.filtered_data[:, [0]], n_pad, axis=1)
            full_block = np.hstack([pad_block, valid_slice])
            if full_block.shape[1] > win_samples:
                full_block = full_block[:, -win_samples:]
            return t_axis, full_block
        else:
            return t_axis, self.filtered_data[:, start_idx:end_idx]


# ==============================================================================
# PLAYBACK ENGINE WITH PRECISION WALL-CLOCK SYNC
# ==============================================================================

class PlaybackEngine(QObject):
    frame_changed = pyqtSignal(int, float)
    state_changed = pyqtSignal(bool)

    def __init__(self, data_loader: EEGDataLoader):
        super().__init__()
        self.data_loader = data_loader
        self.is_playing = False
        self.speed = 1.0
        self.current_sample = 0
        self.timer = QTimer(self)
        self.timer.setInterval(30)  # ~33 FPS target
        self.timer.timeout.connect(self._on_tick)
        self._last_perf_time: float = 0.0

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self._last_perf_time = time.perf_counter()
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
        self.speed = max(0.05, speed)

    def seek_sample(self, sample_idx: int):
        self.current_sample = max(0, min(self.data_loader.n_samples - 1, sample_idx))
        self.frame_changed.emit(self.current_sample, self.current_sample / self.data_loader.sfreq)

    def seek_time(self, time_sec: float):
        self.seek_sample(int(time_sec * self.data_loader.sfreq))

    def step(self, delta_samples: int):
        self.seek_sample(self.current_sample + delta_samples)

    def _on_tick(self):
        now = time.perf_counter()
        dt = now - self._last_perf_time
        self._last_perf_time = now

        advance_samples = dt * self.data_loader.sfreq * self.speed
        new_sample = self.current_sample + advance_samples

        if new_sample >= self.data_loader.n_samples - 1:
            self.current_sample = 0
        else:
            self.current_sample = int(new_sample)

        self.frame_changed.emit(self.current_sample, self.current_sample / self.data_loader.sfreq)


# ==============================================================================
# 2D TOPOMAP WIDGET (RELIABLE HIGH-PERFORMANCE CONTINUOUS VOLTAGE HEATMAP)
# ==============================================================================

class Topomap2DWidget(QWidget):
    channel_toggled = pyqtSignal(str, bool)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.selected_channels: Set[str] = set()
        self.v_scale = 50.0

        self._init_ui()
        self._precompute_idw_matrix()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        self.plot_widget.setRange(xRange=[-1.25, 1.25], yRange=[-1.25, 1.25])

        # Topomap continuous RGBA heatmap item (Layered in front of background at Z=1)
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.img_item.setZValue(1)

        self._draw_head_schematic()

        self.scatter = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen('#1DB954', width=1.5), brush=pg.mkBrush('#14171E'), hoverable=True
        )
        self.scatter.sigClicked.connect(self._on_electrode_clicked)
        self.scatter.setZValue(10)
        self.plot_widget.addItem(self.scatter)

        self.label_items: Dict[str, pg.TextItem] = {}
        for ch in self.data_loader.channel_names:
            if ch in MONTAGE_2D_COORDS:
                x, y = MONTAGE_2D_COORDS[ch]
                lbl = pg.TextItem(ch, color='#7A889B', anchor=(0.5, 1.3))
                lbl.setFont(QFont("Segoe UI", 7, QFont.Bold))
                lbl.setZValue(15)
                self.plot_widget.addItem(lbl)
                lbl.setPos(x, y)
                self.label_items[ch] = lbl

        layout.addWidget(self.plot_widget)

    def _draw_head_schematic(self):
        theta = np.linspace(0, 2 * np.pi, 200)
        r = 1.0
        head_curve = pg.PlotCurveItem(r * np.cos(theta), r * np.sin(theta), pen=pg.mkPen('#303846', width=2.5))
        head_curve.setZValue(5)
        self.plot_widget.addItem(head_curve)

        nose_curve = pg.PlotCurveItem(np.array([-0.14, 0.0, 0.14]), np.array([0.98, 1.15, 0.98]), pen=pg.mkPen('#303846', width=2.5))
        nose_curve.setZValue(5)
        self.plot_widget.addItem(nose_curve)

        ear_l = pg.PlotCurveItem(np.array([-0.99, -1.08, -1.08, -0.99]), np.array([0.15, 0.08, -0.08, -0.15]), pen=pg.mkPen('#303846', width=2.0))
        ear_l.setZValue(5)
        self.plot_widget.addItem(ear_l)

        ear_r = pg.PlotCurveItem(np.array([0.99, 1.08, 1.08, 0.99]), np.array([0.15, 0.08, -0.08, -0.15]), pen=pg.mkPen('#303846', width=2.0))
        ear_r.setZValue(5)
        self.plot_widget.addItem(ear_r)

    def _precompute_idw_matrix(self):
        self.grid_res = 64
        x = np.linspace(-1.05, 1.05, self.grid_res)
        y = np.linspace(-1.05, 1.05, self.grid_res)
        self.grid_x, self.grid_y = np.meshgrid(x, y, indexing='ij')
        self.mask_circle = (self.grid_x**2 + self.grid_y**2) <= 1.00

        # Precompute 256-color RGBA LUT (Coolwarm: Blue -> Cyan -> White -> Coral -> Red)
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        colors = np.array([
            [30, 136, 229, 230],   # -V Deep Blue
            [100, 181, 246, 230],  # Soft Cyan-Blue
            [245, 245, 245, 210],  # 0V Neutral White
            [255, 138, 101, 230],  # Soft Coral
            [229, 57, 53, 230]     # +V Deep Crimson Red
        ], dtype=np.float32)
        
        self.lut_256 = np.zeros((256, 4), dtype=np.uint8)
        for c in range(4):
            self.lut_256[:, c] = np.interp(np.linspace(0, 1, 256), pos, colors[:, c]).astype(np.uint8)

        # Extract 64 channel positions
        self.node_positions = []
        self.valid_ch_names = []
        for ch in self.data_loader.channel_names:
            if ch in MONTAGE_2D_COORDS:
                self.node_positions.append(MONTAGE_2D_COORDS[ch])
                self.valid_ch_names.append(ch)

        coords_arr = np.array(self.node_positions, dtype=np.float32)
        grid_pts = np.vstack([self.grid_x.ravel(), self.grid_y.ravel()]).T

        diff = grid_pts[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-4
        weights = 1.0 / (dist ** 2.5)
        weights /= np.sum(weights, axis=1, keepdims=True)
        self.idw_matrix = weights.astype(np.float32)

        spots = []
        for i, (nx, ny) in enumerate(coords_arr):
            spots.append({'pos': (nx, ny), 'data': self.valid_ch_names[i]})
        self.scatter.setData(spots)

        # Initial baseline image & explicit position rect
        init_rgba = np.zeros((self.grid_res, self.grid_res, 4), dtype=np.uint8)
        init_rgba[self.mask_circle, :] = [245, 245, 245, 180]
        self.img_item.setImage(init_rgba, autoLevels=False)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

    def update_voltage_frame(self, sample_idx: int):
        voltages = self.data_loader.filtered_data[:len(self.valid_ch_names), sample_idx]
        grid_flat = np.dot(self.idw_matrix, voltages)
        v_norm = np.clip((grid_flat + self.v_scale) / (2.0 * self.v_scale), 0.0, 1.0)
        idx = (v_norm * 255.0).astype(np.uint8)

        # Map to RGBA and mask outside head circle to 100% transparency
        rgba = self.lut_256[idx].reshape((self.grid_res, self.grid_res, 4))
        rgba[~self.mask_circle] = 0

        self.img_item.setImage(rgba, autoLevels=False)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

    def _on_electrode_clicked(self, item, points):
        if len(points) == 0:
            return
        ch_name = points[0].data()
        if ch_name in self.selected_channels:
            self.selected_channels.remove(ch_name)
            toggled = False
        else:
            self.selected_channels.add(ch_name)
            toggled = True

        self._refresh_node_styles()
        self.channel_toggled.emit(ch_name, toggled)

    def _refresh_node_styles(self):
        spots = []
        for ch in self.valid_ch_names:
            nx, ny = MONTAGE_2D_COORDS[ch]
            is_on = ch in self.selected_channels
            c_hex = self.data_loader.channel_colors[ch] if is_on else '#1A1E26'
            pen_c = '#FFFFFF' if is_on else '#3E495B'
            size = 18 if is_on else 13
            spots.append({
                'pos': (nx, ny),
                'data': ch,
                'size': size,
                'brush': pg.mkBrush(c_hex),
                'pen': pg.mkPen(pen_c, width=2.0 if is_on else 1.0)
            })
            if ch in self.label_items:
                self.label_items[ch].setColor('#FFFFFF' if is_on else '#7A889B')

        self.scatter.setData(spots)


# ==============================================================================
# PHASE 4: 3D BRAIN VIEWPORT WITH SURFACE-PINNED ELECTRODES
# ==============================================================================

class Brain3DWidget(QWidget):
    """
    Phase 4: 3D Topographic Heat Map Engine:
    - Renders data/human-brain.glb (or procedural cortical surface).
    - Decimates mesh (~3,500 vertices) for 60+ FPS rendering.
    - Pins all 64 electrode spheres directly to the exterior surface using surface normals.
    - Symmetrically centers electrodes on the longitudinal fissure (Z=0).
    - Top-down superior view, frontal cortex pointing UP (distance 5.6).
    - Disabling 3D electrodes hides both base spheres and selection halos.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.brain_mesh = None
        self.elec_mesh = None
        self.mesh_actor = None
        self.elec_actor = None
        self.highlight_actor = None
        self.scalar_bar_actor = None
        self.idw_3d_matrix = None
        self.elec_coord_dict: Dict[str, np.ndarray] = {}
        self.valid_ch_names: List[str] = []
        self.pts_per_sphere = 1
        self.selected_channels: Set[str] = set()

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
            raw_mesh = pv.ParametricEllipsoid(xradius=1.8, yradius=1.4, zradius=1.3)
            raw_mesh.points += 0.05 * np.sin(raw_mesh.points * 8.0)

        # Center mesh at (0, 0, 0) to ensure perfect symmetric alignment with camera
        center = raw_mesh.center
        raw_mesh.points -= center

        # Decimate to ~3,500 vertices for 60+ FPS performance
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
            cmap='coolwarm',
            clim=self.clim,
            smooth_shading=True,
            specular=0.35,
            ambient=0.25,
            diffuse=0.85,
            show_scalar_bar=False,
            lighting=True
        )

    def _precompute_3d_idw(self):
        """
        Pins all 64 electrodes directly onto the exterior surface of the model,
        ensuring bilateral symmetry along the longitudinal fissure (Z=0).
        """
        bounds = self.brain_mesh.bounds
        dx = (bounds[1] - bounds[0]) / 2.0
        dy = (bounds[3] - bounds[2]) / 2.0
        dz = (bounds[5] - bounds[4]) / 2.0

        mesh_pts = np.array(self.brain_mesh.points, dtype=np.float32)
        mesh_norms = np.array(self.brain_mesh.point_normals, dtype=np.float32)

        elec_pts = []
        self.valid_ch_names = []
        self.elec_coord_dict.clear()

        # Coordinate alignment:
        # +X is Anterior (+Y in 2D topomap)
        # +Y is Superior (Top-down view)
        # +Z is Lateral (+X in 2D topomap)
        for ch in STANDARD_64_CHANNELS[:64]:
            if ch in MONTAGE_2D_COORDS:
                x2, y2 = MONTAGE_2D_COORDS[ch]
                r_sq = min(0.96, x2**2 + y2**2)
                y_sup = math.sqrt(max(0.04, 1.0 - r_sq))

                # Strict bilateral symmetry: Z depends strictly on x2
                target_x = (y2 - 0.04) * (dx * 0.88)
                target_z = (x2) * (dz * 0.88) if abs(x2) > 0.02 else 0.0

                # Find vertices near (target_x, target_z) in the horizontal plane
                dists_2d = np.sqrt((mesh_pts[:, 0] - target_x)**2 + (mesh_pts[:, 2] - target_z)**2)
                nearby = (dists_2d < (0.28 * min(dx, dz))) & (mesh_pts[:, 1] >= 0.0)

                if np.any(nearby):
                    # Pick highest superior vertex (outer top cortex surface)
                    sub_indices = np.where(nearby)[0]
                    best_id = sub_indices[np.argmax(mesh_pts[sub_indices, 1])]
                    surf_pt = mesh_pts[best_id]
                    norm = mesh_norms[best_id]
                else:
                    approx_pt = np.array([target_x, y_sup * dy * 1.1, target_z], dtype=np.float32)
                    best_id = self.brain_mesh.find_closest_point(approx_pt)
                    surf_pt = mesh_pts[best_id]
                    norm = mesh_norms[best_id]

                norm_len = np.linalg.norm(norm)
                if norm_len > 1e-4:
                    norm = norm / norm_len
                else:
                    norm = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                # Offset outward along surface normal so entire sphere rests on the exterior
                pinned_coord = surf_pt + norm * 0.055

                # For midline electrodes, enforce exact Z=0.0
                if abs(x2) < 0.02:
                    pinned_coord[2] = 0.0

                elec_pts.append(pinned_coord)
                self.valid_ch_names.append(ch)
                self.elec_coord_dict[ch] = pinned_coord

        elec_arr = np.array(elec_pts, dtype=np.float32)

        # Precompute 3D IDW projection matrix
        diff = mesh_pts[:, np.newaxis, :] - elec_arr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-4
        weights = 1.0 / (dist ** 2.5)

        # Attenuate ventral/inferior brain stem vertices
        inferior_mask = mesh_pts[:, 1] < (-0.2 * dy)
        weights[inferior_mask, :] *= 0.3

        weights /= np.sum(weights, axis=1, keepdims=True)
        self.idw_3d_matrix = weights.astype(np.float32)
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
            cmap='coolwarm',
            clim=self.clim,
            smooth_shading=True,
            specular=0.8,
            show_scalar_bar=False,
            lighting=True
        )

    def reset_camera_view(self):
        """Top-down superior view looking straight down (+Y down), frontal cortex (+X) up, distance 5.6."""
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        self.plotter.camera_position = [(0.0, 5.6, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def update_voltage_frame(self, voltages_64: np.ndarray, v_scale: float = 50.0):
        if not PYVISTA_AVAILABLE or self.brain_mesh is None:
            return

        # 1. Update 3D Cortical Surface Heatmap
        if self.show_heatmap and self.idw_3d_matrix is not None:
            cortex_scalars = np.dot(self.idw_3d_matrix, voltages_64[:len(self.valid_ch_names)])
            self.brain_mesh['voltage'] = cortex_scalars

        # 2. Update 3D Electrode Spheres
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
        self.plotter.render()

    def set_selected_channels(self, selected_set: Set[str]):
        self.selected_channels = set(selected_set)
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return

        if self.highlight_actor is not None:
            self.plotter.remove_actor(self.highlight_actor)
            self.highlight_actor = None

        if len(self.selected_channels) > 0 and self.show_electrodes:
            sel_pts = []
            for ch in self.selected_channels:
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
        self.plotter.render()

    def _on_toggle_heatmap(self, checked: bool):
        self.show_heatmap = checked
        if not PYVISTA_AVAILABLE or self.mesh_actor is None:
            return
        mapper = self.mesh_actor.GetMapper()
        if mapper:
            mapper.SetScalarVisibility(checked)
        self.plotter.render()

    def _on_toggle_electrodes(self, checked: bool):
        self.show_electrodes = checked
        if not PYVISTA_AVAILABLE:
            return
        if self.elec_actor is not None:
            self.elec_actor.SetVisibility(checked)
        # Even selected electrodes disappear when electrodes are toggled off
        if self.highlight_actor is not None:
            self.highlight_actor.SetVisibility(checked)
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
        self.plotter.render()


# ==============================================================================
# TOPOMAP CONTAINER (TOGGLEABLE 2D MAP / 3D BRAIN WITH ZERO BACKGROUND OVERHEAD)
# ==============================================================================

class TopomapContainerWidget(QWidget):
    """
    Unified Topomap Viewport:
    - Hosts a QStackedWidget switching between Page 0 (2D Topomap) and Page 1 (3D Brain).
    - Features a header segmented mode selector: [ 2D Topomap ] | [ 3D Brain ].
    - When 2D mode is active, the 3D engine is completely stopped (zero background GPU/CPU overhead).
    """
    mode_changed = pyqtSignal(str)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.current_mode = "2D"

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header Bar with Mode Toggle Buttons
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
                padding: 3px 10px;
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
        h_layout.setSpacing(10)

        self.title_lbl = QLabel("<span style='color: #00FFA3;'>●</span> 2D TOPOGRAPHIC MAP (10-05 MONTAGE)")
        self.title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(self.title_lbl)

        h_layout.addStretch()

        self.status_ch_lbl = QLabel("64 Channels Ready")
        self.status_ch_lbl.setStyleSheet("color: #7A889B; font-size: 10px; margin-right: 8px;")
        h_layout.addWidget(self.status_ch_lbl)

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
        self.widget_3d = Brain3DWidget(self)

        self.stack.addWidget(self.widget_2d)  # Index 0
        self.stack.addWidget(self.widget_3d)  # Index 1

        layout.addWidget(self.stack, stretch=1)
        self._refresh_button_styles()

    def set_mode(self, mode: str):
        if mode == self.current_mode:
            return
        self.current_mode = mode
        if mode == "2D":
            self.stack.setCurrentIndex(0)
            self.title_lbl.setText("<span style='color: #00FFA3;'>●</span> 2D TOPOGRAPHIC MAP (10-05 MONTAGE)")
        else:
            self.stack.setCurrentIndex(1)
            self.title_lbl.setText("<span style='color: #00FFA3;'>●</span> 3D CORTICAL TOPOMAP (PHASE 4 ENGINE)")
            # Trigger 3D view refresh upon cold switch
            self.widget_3d.reset_camera_view()

        self._refresh_button_styles()
        self.mode_changed.emit(self.current_mode)

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
# ANALYSIS PANEL WIDGET (REPLACING 3D VIEWPORT ON TOP-RIGHT)
# ==============================================================================

class AnalysisPanelWidget(QWidget):
    """
    Analysis Workstation Panel (Top-Right Section):
    Replaces former 3D viewport area. Empty placeholder structured for Phase 5
    analytical, spectral, connectivity, and epoch decomposition modules.
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

        placeholder_title = QLabel("ANALYSIS MODULES")
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
    Stacked multi-track real-time time-series plot.
    Only channels currently toggled 'ON' in the 2D topomap are displayed.
    Traces dynamically color-match their respective 2D electrode nodes.
    Starts with 0 channels active on launch.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.active_channels: List[str] = []
        self.gain = 100.0       # µV per trace height
        self.win_sec = 6.0      # Visible window in seconds

        self._cache_time_axis()
        self._init_ui()

    def _cache_time_axis(self):
        win_samples = int(self.win_sec * self.data_loader.sfreq)
        self.cached_t_axis = np.linspace(-self.win_sec, 0.0, win_samples, endpoint=False)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header Bar for Raw Wave Data Analyzer
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

        title_lbl = QLabel("<span style='color: #00FFA3;'>●</span> CASCADING WAVEFORMS (FILTERED RAW EEG)")
        title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(title_lbl)

        h_layout.addStretch()

        self.info_lbl = QLabel("0 Channels Displayed • Click 2D Nodes to Activate")
        self.info_lbl.setStyleSheet("color: #7A889B; font-size: 10px;")
        h_layout.addWidget(self.info_lbl)

        layout.addWidget(header)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel('bottom', "Time (seconds relative to playhead)", **{'color': '#7A889B', 'font-size': '10pt'})
        self.plot_widget.getAxis('left').setStyle(showValues=False)
        self.plot_widget.setMouseEnabled(x=False, y=False)

        # Empty state prompt text
        self.empty_label = pg.TextItem(
            "No channels active.\nClick electrode nodes on the 2D Topomap to display waveforms.",
            color='#526173', anchor=(0.5, 0.5)
        )
        self.empty_label.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        self.plot_widget.addItem(self.empty_label)
        self.empty_label.setPos(0, 0)

        layout.addWidget(self.plot_widget)
        self.curves: Dict[str, pg.PlotDataItem] = {}
        self.channel_labels: Dict[str, pg.TextItem] = {}

    def set_active_channels(self, channels: List[str]):
        self.active_channels = [ch for ch in channels if ch in self.data_loader.channel_names]
        self.plot_widget.clear()

        if len(self.active_channels) == 0:
            self.curves.clear()
            self.channel_labels.clear()
            self.empty_label.setPos(-self.win_sec / 2.0, 0)
            self.plot_widget.addItem(self.empty_label)
            self.plot_widget.setYRange(-1, 1)
            self.info_lbl.setText("0 Channels Displayed • Click 2D Nodes to Activate")
            return

        self.curves.clear()
        self.channel_labels.clear()
        
        n_ch = len(self.active_channels)
        self.plot_widget.setYRange(-0.8, n_ch - 0.2)
        self.plot_widget.setXRange(-self.win_sec, 0.0)

        for i, ch in enumerate(self.active_channels):
            c_hex = self.data_loader.channel_colors.get(ch, '#1DB954')
            pen = pg.mkPen(color=c_hex, width=1.6)
            curve = self.plot_widget.plot(pen=pen)
            curve.setClipToView(True)
            self.curves[ch] = curve

            # Set label position once here to avoid expensive setPos() calls during frame loop
            lbl = pg.TextItem(ch, color=c_hex, anchor=(1.0, 0.5))
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setPos(-self.win_sec + 0.12, i)
            self.plot_widget.addItem(lbl)
            self.channel_labels[ch] = lbl

        self.info_lbl.setText(f"{n_ch} of 64 Channels Active")

    def update_window(self, win_sec: float):
        self.win_sec = win_sec
        self._cache_time_axis()
        self.set_active_channels(self.active_channels)

    def update_frame(self, current_sample: int):
        if len(self.active_channels) == 0:
            return

        _, block = self.data_loader.get_window_data(current_sample, self.win_sec)
        
        for i, ch in enumerate(self.active_channels):
            ch_idx = self.data_loader.channel_names.index(ch)
            raw_sig = block[ch_idx]
            norm_sig = (raw_sig / self.gain) + i
            self.curves[ch].setData(self.cached_t_axis, norm_sig)


# ==============================================================================
# SPOTIFY TIMELINE WIDGET (DEFINED BEFORE SPOTIFY PLAYBACK BAR)
# ==============================================================================

class SpotifyTimelineWidget(pg.PlotWidget):
    """
    Spotify-Style Full Dataset Interactive Timeline:
    Defined BEFORE SpotifyPlaybackBar to prevent NameError runtime exceptions.
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
        self.setFixedHeight(24)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(x=False, y=False)
        self.setXRange(0.0, self.data_loader.duration, padding=0.005)
        self.setYRange(-0.5, 0.5)

        # Baseline rail
        rail = pg.PlotCurveItem([0.0, self.data_loader.duration], [0.0, 0.0], pen=pg.mkPen('#1C222E', width=6.0))
        self.addItem(rail)

        # Event regions
        for ev in self.data_loader.events:
            cfg = EVENT_COLOR_MAP.get(ev.event_id, {'bg': '#1E2530', 'border': '#7A889B'})
            brush_c = QColor(cfg['bg'])
            brush_c.setAlpha(200)
            region = pg.LinearRegionItem(
                [ev.start_time, ev.end_time], movable=False,
                brush=QBrush(brush_c), pen=pg.mkPen(cfg['border'], width=1.0)
            )
            self.addItem(region)

        # White playhead cursor line
        self.playhead_line = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False,
            pen=pg.mkPen('#FFFFFF', width=2.0)
        )
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


# ==============================================================================
# SPOTIFY PLAYBACK BAR (EXACT PHASE 2 FORMAT)
# ==============================================================================

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
            font-size: 11px;
            font-weight: 900;
            border-radius: 4px;
        """)
        left_layout.addWidget(eeg_badge)

        info_vbox = QVBoxLayout()
        info_vbox.setSpacing(1)
        info_vbox.setAlignment(Qt.AlignVCenter)

        lbl_track_title = QLabel("PhysioNet S001 • Run 04")
        lbl_track_title.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold;")
        lbl_track_sub = QLabel("Motor Imagery 64ch")
        lbl_track_sub.setStyleSheet("color: #7A889B; font-size: 10px;")

        info_vbox.addWidget(lbl_track_title)
        info_vbox.addWidget(lbl_track_sub)
        left_layout.addLayout(info_vbox)

        layout.addLayout(left_layout)

        # Center: Transport Controls (Top) + Timeline Bar (Bottom)
        center_vbox = QVBoxLayout()
        center_vbox.setSpacing(4)
        center_vbox.setAlignment(Qt.AlignCenter)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(16)
        ctrl_row.setAlignment(Qt.AlignCenter)

        self.step_back_btn = QPushButton("⏮")
        self.step_back_btn.setFixedSize(28, 28)
        self.step_back_btn.setToolTip("Step Back (0.1s)")
        self.step_back_btn.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #00E5FF;
                font-size: 13px;
                border: none;
                border-radius: 14px;
            }
            QPushButton:hover {
                background: #1F2737;
            }
        """)
        self.step_back_btn.clicked.connect(lambda: self.engine.step(-int(self.data_loader.sfreq * 0.1)))

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(36, 36)
        self.play_btn.setToolTip("Play / Pause (Spacebar)")
        self.play_btn.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                color: #000000;
                font-size: 16px;
                font-weight: bold;
                border-radius: 18px;
            }
            QPushButton:hover {
                background: #E0E0E0;
            }
            QPushButton:pressed {
                background: #B0B0B0;
            }
        """)
        self.play_btn.clicked.connect(self.engine.toggle_play)

        self.step_fwd_btn = QPushButton("⏭")
        self.step_fwd_btn.setFixedSize(28, 28)
        self.step_fwd_btn.setToolTip("Step Forward (0.1s)")
        self.step_fwd_btn.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #00E5FF;
                font-size: 13px;
                border: none;
                border-radius: 14px;
            }
            QPushButton:hover {
                background: #1F2737;
            }
        """)
        self.step_fwd_btn.clicked.connect(lambda: self.engine.step(int(self.data_loader.sfreq * 0.1)))

        ctrl_row.addWidget(self.step_back_btn)
        ctrl_row.addWidget(self.play_btn)
        ctrl_row.addWidget(self.step_fwd_btn)
        center_vbox.addLayout(ctrl_row)

        # Timeline row
        time_row = QHBoxLayout()
        time_row.setSpacing(10)

        self.cur_time_lbl = QLabel("0:00.000")
        self.cur_time_lbl.setFixedWidth(62)
        self.cur_time_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cur_time_lbl.setStyleSheet("color: #8C9BAE; font-family: 'Consolas', monospace; font-size: 10px;")

        self.timeline = SpotifyTimelineWidget(self.data_loader, parent=self, engine=self.engine)
        self.timeline.seek_requested.connect(self.engine.seek_time)

        tot_time = self.data_loader.duration
        t_m = int(tot_time // 60)
        t_s = tot_time % 60
        self.tot_time_lbl = QLabel(f"{t_m}:{t_s:06.3f}")
        self.tot_time_lbl.setFixedWidth(62)
        self.tot_time_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.tot_time_lbl.setStyleSheet("color: #8C9BAE; font-family: 'Consolas', monospace; font-size: 10px;")

        time_row.addWidget(self.cur_time_lbl)
        time_row.addWidget(self.timeline, stretch=1)
        time_row.addWidget(self.tot_time_lbl)

        center_vbox.addLayout(time_row)
        layout.addLayout(center_vbox, stretch=1)

        # Right: Stimulus Legend & Dynamic Readout
        right_vbox = QVBoxLayout()
        right_vbox.setSpacing(3)
        right_vbox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        legend_lbl = QLabel(
            "<span style='color: #00E5FF;'>●</span> Rest &nbsp;&nbsp; "
            "<span style='color: #00FFA3;'>●</span> Left Fist &nbsp;&nbsp; "
            "<span style='color: #BA68C8;'>●</span> Right Fist"
        )
        legend_lbl.setStyleSheet("font-size: 10px; color: #8C9BAE; font-weight: bold;")
        legend_lbl.setAlignment(Qt.AlignRight)

        self.active_stim_lbl = QLabel("Active Stimulus: Rest")
        self.active_stim_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #00FFA3;")
        self.active_stim_lbl.setAlignment(Qt.AlignRight)

        right_vbox.addWidget(legend_lbl)
        right_vbox.addWidget(self.active_stim_lbl)
        layout.addLayout(right_vbox)

    def update_playback_state(self, is_playing: bool):
        self.play_btn.setText("⏸" if is_playing else "▶")

    def update_frame(self, current_sample: int, current_time: float):
        cur_min = int(current_time // 60)
        cur_sec = current_time % 60
        self.cur_time_lbl.setText(f"{cur_min}:{cur_sec:06.3f}")
        self.timeline.update_playhead(current_time)

        # Find current active stimulus
        active_label = "Rest"
        active_color = "#00FFA3"
        for ev in self.data_loader.events:
            if ev.start_time <= current_time <= ev.end_time:
                active_label = ev.label
                active_color = EVENT_COLOR_MAP.get(ev.event_id, {}).get('text', '#00FFA3')
                break

        self.active_stim_lbl.setText(f"Active Stimulus: {active_label}")
        self.active_stim_lbl.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {active_color};")


# ==============================================================================
# TOP HEADER & COMPACT METADATA PILLS
# ==============================================================================

class HeaderWidget(QFrame):
    export_requested = pyqtSignal()

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._init_ui()

    def _init_ui(self):
        self.setFixedHeight(50)
        self.setStyleSheet("""
            QFrame {
                background: #0E0F14;
                border-bottom: 1px solid #1A1F2B;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(14)

        title_label = QLabel("neuromap")
        title_label.setStyleSheet("color: #FFFFFF; font-size: 20px; font-weight: 800; letter-spacing: 1px;")
        layout.addWidget(title_label)

        layout.addSpacing(16)

        pills = [
            ("SUBJ", self.data_loader.subject_id),
            ("SRATE", f"{self.data_loader.sfreq:.0f} Hz"),
            ("DURATION", f"{self.data_loader.duration:.1f} s"),
            ("CHANNELS", f"{len(self.data_loader.channel_names)} EEG"),
            ("TYPE", self.data_loader.file_type),
            ("STATUS", self.data_loader.loaded_status)
        ]

        for tag, val in pills:
            pill = QLabel(f"<span style='color: #6A788B; font-weight: bold;'>{tag}:</span> <span style='color: #E2E8F0;'>{val}</span>")
            pill.setStyleSheet("""
                background: #14171F;
                border: 1px solid #202632;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            """)
            layout.addWidget(pill)

        layout.addStretch()

        export_btn = QPushButton("⭳ Export Data")
        export_btn.setStyleSheet("""
            QPushButton {
                background: #1A2230;
                color: #00E5FF;
                border: 1px solid #00E5FF;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #00E5FF;
                color: #0E0F14;
            }
        """)
        export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(export_btn)


# ==============================================================================
# LEFT SIDEBAR CONTROLS
# ==============================================================================

class ControlSidebarWidget(QWidget):
    settings_changed = pyqtSignal()

    def __init__(self, data_loader: EEGDataLoader, engine: PlaybackEngine, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.engine = engine
        self._init_ui()

    def _init_ui(self):
        self.setFixedWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Signal Processing Controls
        lbl_sig = QLabel("SIGNAL PROCESSING")
        lbl_sig.setStyleSheet("color: #7A889B; font-size: 11px; font-weight: bold; letter-spacing: 0.8px;")
        layout.addWidget(lbl_sig)

        notch_box = QHBoxLayout()
        self.chk_notch60 = QCheckBox("60 Hz Notch")
        self.chk_notch60.setChecked(True)
        self.chk_notch60.toggled.connect(self._on_notch_toggled)

        self.chk_notch50 = QCheckBox("50 Hz Notch")
        self.chk_notch50.setChecked(False)
        self.chk_notch50.toggled.connect(self._on_notch_toggled)

        notch_box.addWidget(self.chk_notch60)
        notch_box.addWidget(self.chk_notch50)
        layout.addLayout(notch_box)

        hp_row = QHBoxLayout()
        self.hp_slider = QSlider(Qt.Horizontal)
        self.hp_slider.setRange(1, 20)
        self.hp_slider.setValue(1)
        self.hp_edit = QLineEdit("1.0")
        self.hp_edit.setFixedWidth(45)
        self.hp_slider.valueChanged.connect(self._on_hp_slider)
        self.hp_edit.returnPressed.connect(self._on_hp_edit)
        hp_row.addWidget(QLabel("HP:"))
        hp_row.addWidget(self.hp_slider)
        hp_row.addWidget(self.hp_edit)
        hp_row.addWidget(QLabel("Hz"))
        layout.addLayout(hp_row)

        lp_row = QHBoxLayout()
        self.lp_slider = QSlider(Qt.Horizontal)
        self.lp_slider.setRange(20, 70)
        self.lp_slider.setValue(40)
        self.lp_edit = QLineEdit("40.0")
        self.lp_edit.setFixedWidth(45)
        self.lp_slider.valueChanged.connect(self._on_lp_slider)
        self.lp_edit.returnPressed.connect(self._on_lp_edit)
        lp_row.addWidget(QLabel("LP:"))
        lp_row.addWidget(self.lp_slider)
        lp_row.addWidget(self.lp_edit)
        lp_row.addWidget(QLabel("Hz"))
        layout.addLayout(lp_row)

        layout.addSpacing(14)

        # Display Controls
        lbl_disp = QLabel("DISPLAY CONTROLS")
        lbl_disp.setStyleSheet("color: #7A889B; font-size: 11px; font-weight: bold; letter-spacing: 0.8px;")
        layout.addWidget(lbl_disp)

        gain_row = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(20, 300)
        self.gain_slider.setValue(100)
        self.gain_edit = QLineEdit("100")
        self.gain_edit.setFixedWidth(45)
        self.gain_slider.valueChanged.connect(lambda v: self.gain_edit.setText(str(v)))
        gain_row.addWidget(QLabel("Gain:"))
        gain_row.addWidget(self.gain_slider)
        gain_row.addWidget(self.gain_edit)
        gain_row.addWidget(QLabel("µV"))
        layout.addLayout(gain_row)

        topo_row = QHBoxLayout()
        self.topo_slider = QSlider(Qt.Horizontal)
        self.topo_slider.setRange(20, 100)
        self.topo_slider.setValue(50)
        topo_row.addWidget(QLabel("Topomap:"))
        topo_row.addWidget(self.topo_slider)
        topo_row.addWidget(QLabel("±µV"))
        layout.addLayout(topo_row)

        win_row = QHBoxLayout()
        self.win_slider = QSlider(Qt.Horizontal)
        self.win_slider.setRange(2, 15)
        self.win_slider.setValue(6)
        self.win_edit = QLineEdit("6.0")
        self.win_edit.setFixedWidth(45)
        self.win_slider.valueChanged.connect(lambda v: self.win_edit.setText(f"{v:.1f}"))
        win_row.addWidget(QLabel("Window:"))
        win_row.addWidget(self.win_slider)
        win_row.addWidget(self.win_edit)
        win_row.addWidget(QLabel("s"))
        layout.addLayout(win_row)

        # Playback Speed (with 0.1x and 0.25x settings)
        speed_row = QHBoxLayout()
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.1x", "0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "4.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        speed_row.addWidget(QLabel("Speed:"))
        speed_row.addWidget(self.speed_combo)
        layout.addLayout(speed_row)

        layout.addStretch()

    def _on_notch_toggled(self):
        self.data_loader.notch_60 = self.chk_notch60.isChecked()
        self.data_loader.notch_50 = self.chk_notch50.isChecked()
        self.data_loader.apply_filters()
        self.settings_changed.emit()

    def _on_hp_slider(self, val):
        self.hp_edit.setText(f"{val:.1f}")
        self.data_loader.hp_freq = float(val)
        self.data_loader.apply_filters()
        self.settings_changed.emit()

    def _on_hp_edit(self):
        try:
            val = float(self.hp_edit.text())
            self.hp_slider.setValue(int(val))
        except ValueError:
            pass

    def _on_lp_slider(self, val):
        self.lp_edit.setText(f"{val:.1f}")
        self.data_loader.lp_freq = float(val)
        self.data_loader.apply_filters()
        self.settings_changed.emit()

    def _on_lp_edit(self):
        try:
            val = float(self.lp_edit.text())
            self.lp_slider.setValue(int(val))
        except ValueError:
            pass

    def _on_speed_changed(self, text: str):
        val = float(text.replace('x', ''))
        self.engine.set_speed(val)


# ==============================================================================
# MAIN APPLICATION WINDOW
# ==============================================================================

class NeuromapMainWindow(QMainWindow):
    """
    Main Application Window:
    - Top row (QSplitter):
      * Left: TopomapContainerWidget (Toggleable 2D Map / 3D Brain)
      * Right: AnalysisPanelWidget (Clean workstation placeholder)
    - Middle/Bottom section: Cascading Multi-Track Waveforms
    - Bottom Dock: Phase 2 Spotify Playback Bar Format
    - Auto-launches Maximized
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

        # Workspace (Sidebar + Vertical Splitter)
        main_workspace = QHBoxLayout()
        main_workspace.setContentsMargins(0, 0, 0, 0)
        main_workspace.setSpacing(0)

        self.sidebar = ControlSidebarWidget(self.data_loader, self.engine, self)
        main_workspace.addWidget(self.sidebar)

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setHandleWidth(4)

        # Top Row (Horizontal Splitter: Topomap Viewport + Analysis Panel)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setHandleWidth(4)

        # Top-Left: Toggleable Topomap Container (2D Map / 3D Brain)
        self.topomap_container = TopomapContainerWidget(self.data_loader, self)
        # Top-Right: Analysis Workstation Panel
        self.analysis_panel = AnalysisPanelWidget(self)

        self.top_splitter.addWidget(self.topomap_container)
        self.top_splitter.addWidget(self.analysis_panel)
        self.top_splitter.setSizes([960, 960])
        self.main_splitter.addWidget(self.top_splitter)

        # Middle/Bottom Section: Cascading Waveforms
        self.waveforms_widget = WaveformsWidget(self.data_loader, self)
        self.main_splitter.addWidget(self.waveforms_widget)
        self.main_splitter.setSizes([560, 480])

        main_workspace.addWidget(self.main_splitter, stretch=1)
        root_layout.addLayout(main_workspace, stretch=1)

        # Bottom Dock: Spotify Playback Bar (Phase 2 Format)
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

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.engine.toggle_play)
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self._toggle_fullscreen)

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

        # Connect 2D Topomap channel toggles
        self.topomap_container.widget_2d.channel_toggled.connect(self._on_channel_toggled)

        self.sidebar.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.sidebar.win_slider.valueChanged.connect(self._on_window_changed)
        self.sidebar.topo_slider.valueChanged.connect(self._on_topo_scale_changed)

    def _on_frame_update(self, current_sample: int, current_time: float):
        # 1. Update only the currently active topomap mode (zero 3D overhead when in 2D)
        if self.topomap_container.current_mode == "2D":
            self.topomap_container.widget_2d.update_voltage_frame(current_sample)
        else:
            voltages_64 = self.data_loader.filtered_data[:len(self.topomap_container.widget_2d.valid_ch_names), current_sample]
            self.topomap_container.widget_3d.update_voltage_frame(voltages_64, self.topomap_container.widget_2d.v_scale)

        # 2. Update Cascading Waveforms
        self.waveforms_widget.update_frame(current_sample)

        # 3. Update Spotify Playback Bar (Time & Stimulus status)
        self.playback_bar.update_frame(current_sample, current_time)

    def _on_channel_toggled(self, ch_name: str, is_active: bool):
        selected_list = sorted(list(self.topomap_container.widget_2d.selected_channels))
        self.waveforms_widget.set_active_channels(selected_list)
        self.topomap_container.widget_3d.set_selected_channels(self.topomap_container.widget_2d.selected_channels)
        self.topomap_container.status_ch_lbl.setText(f"{len(selected_list)} of 64 Active")
        self.status_label.setText(f"Status: {len(selected_list)} of 64 channels selected ({', '.join(selected_list[:6]) if selected_list else 'None'})")

    def _on_gain_changed(self, val: int):
        self.waveforms_widget.gain = float(val)

    def _on_window_changed(self, val: int):
        self.waveforms_widget.update_window(float(val))

    def _on_topo_scale_changed(self, val: int):
        self.topomap_container.widget_2d.v_scale = float(val)
        self.topomap_container.widget_3d.set_clim(float(val))

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