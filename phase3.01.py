"""
neuromap - Professional Closed-Loop EEG Visualization Dashboard
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Stack: PyQt5, PyQtGraph, PyVista / PyVistaQt, MNE-Python, NumPy, SciPy, Scikit-learn
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
    QPolygonF, QIcon, QKeySequence
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QLabel, QPushButton, QSlider, QLineEdit,
    QCheckBox, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QStatusBar, QProgressBar, QFileDialog, QShortcut,
    QFrame, QSizePolicy
)

# PyQtGraph High-Performance Scientific Graphics
import pyqtgraph as pg

# CRITICAL ENGINE SAFETY: Never set useOpenGL=True when using PyVistaQt on Windows.
# It causes fatal OpenGL context collisions (0xC0000005) between QOpenGLWidget and VTK.
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

# Scikit-learn FastICA
try:
    from sklearn.decomposition import FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# DATA STRUCTURES & CONSTANTS
# ==============================================================================

STANDARD_64_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]

PALETTE_COLORS = [
    '#1DB954', '#00E5FF', '#E040FB', '#FFD600', '#FF5252', '#69F0AE',
    '#448AFF', '#FF6E40', '#EEFF41', '#B388FF', '#18FFFF', '#FF4081',
    '#64FFDA', '#B2FF59', '#FFAB40', '#7C4DFF', '#00B0FF', '#FF5722',
    '#40C4FF', '#A7FFEB', '#FFD180', '#FF80AB', '#EA80FC', '#82B1FF'
]

EVENT_COLOR_MAP = {
    'T0': {'name': 'Rest', 'bg': '#1E2530', 'border': '#00E5FF', 'text': '#00E5FF'},
    'T1': {'name': 'Left Fist', 'bg': '#0D332D', 'border': '#00FFA3', 'text': '#00FFA3'},
    'T2': {'name': 'Right Fist', 'bg': '#331238', 'border': '#FF3DF0', 'text': '#FF3DF0'}
}

@dataclass
class StimulusEvent:
    event_id: str
    label: str
    start_time: float
    end_time: float
    start_sample: int
    end_sample: int

@dataclass
class Bookmark:
    id: int
    timestamp: float
    stimulus: str
    note: str

MONTAGE_2D_COORDS = {
    'Fp1': (-0.30, 0.85), 'Fpz': (0.00, 0.88), 'Fp2': (0.30, 0.85),
    'AF7': (-0.60, 0.70), 'AF3': (-0.35, 0.68), 'AFz': (0.00, 0.70), 'AF4': (0.35, 0.68), 'AF8': (0.60, 0.70),
    'F7': (-0.75, 0.50), 'F5': (-0.52, 0.48), 'F3': (-0.32, 0.47), 'F1': (-0.12, 0.46),
    'Fz': (0.00, 0.46), 'F2': (0.12, 0.46), 'F4': (0.32, 0.47), 'F6': (0.52, 0.48), 'F8': (0.75, 0.50),
    'FT7': (-0.85, 0.25), 'FC5': (-0.62, 0.24), 'FC3': (-0.38, 0.24), 'FC1': (-0.15, 0.24),
    'FCz': (0.00, 0.24), 'FC2': (0.15, 0.24), 'FC4': (0.38, 0.24), 'FC6': (0.62, 0.24), 'FT8': (0.85, 0.25),
    'T7': (-0.90, 0.00), 'C5': (-0.68, 0.00), 'C3': (-0.45, 0.00), 'C1': (-0.20, 0.00),
    'Cz': (0.00, 0.00), 'C2': (0.20, 0.00), 'C4': (0.45, 0.00), 'C6': (0.68, 0.00), 'T8': (0.90, 0.00),
    'TP7': (-0.85, -0.25), 'CP5': (-0.62, -0.24), 'CP3': (-0.38, -0.24), 'CP1': (-0.15, -0.24),
    'CPz': (0.00, -0.24), 'CP2': (0.15, -0.24), 'CP4': (0.38, -0.24), 'CP6': (0.62, -0.24), 'TP8': (0.85, -0.25),
    'P7': (-0.75, -0.50), 'P5': (-0.52, -0.48), 'P3': (-0.32, -0.47), 'P1': (-0.12, -0.46),
    'Pz': (0.00, -0.46), 'P2': (0.12, -0.46), 'P4': (0.32, -0.47), 'P6': (0.52, -0.48), 'P8': (0.75, -0.50),
    'PO7': (-0.60, -0.70), 'PO3': (-0.35, -0.68), 'POz': (0.00, -0.70), 'PO4': (0.35, -0.68), 'PO8': (0.60, -0.70),
    'O1': (-0.30, -0.85), 'Oz': (0.00, -0.88), 'O2': (0.30, -0.85), 'Iz': (0.00, -0.96)
}


# ==============================================================================
# DATA LOADER & SIGNAL PROCESSING ENGINE
# ==============================================================================

class EEGDataLoader:
    def __init__(self):
        self.sfreq = 160.0
        self.channel_names: List[str] = STANDARD_64_CHANNELS[:64]
        self.raw_data: np.ndarray = np.zeros((64, 20000), dtype=np.float32)
        self.filtered_data: np.ndarray = np.zeros((64, 20000), dtype=np.float32)
        self.cleaned_data: np.ndarray = np.zeros((64, 20000), dtype=np.float32)
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

        self.ica_components: Optional[np.ndarray] = None
        self.ica_mixing: Optional[np.ndarray] = None
        self.suppress_oc0 = False
        self.suppress_oc1 = False
        self.suppress_emg = False

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
                
                try:
                    montage = mne.channels.make_standard_montage('colin27_1005')
                except Exception:
                    montage = mne.channels.make_standard_montage('standard_1005')

                raw.rename_channels(lambda s: s.strip('.').upper())
                self.sfreq = float(raw.info['sfreq'])
                
                data = raw.get_data()
                n_ch, n_pts = data.shape
                if n_ch < 64:
                    pad = np.random.randn(64 - n_ch, n_pts) * 5e-6
                    data = np.vstack([data, pad])
                elif n_ch > 64:
                    data = data[:64, :]
                
                self.raw_data = (data[:64, :] * 1e6).astype(np.float32)
                self.n_samples = self.raw_data.shape[1]
                self.duration = self.n_samples / self.sfreq
                self.subject_id = "S001"
                self.file_type = "PhysioNet EDF+"
                self.loaded_status = "Online MNE"

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
        self._compute_fast_ica()
        return True

    def _generate_synthetic_eeg(self):
        self.sfreq = 160.0
        self.duration = 125.0
        self.n_samples = int(self.duration * self.sfreq)
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

        blink_times = np.arange(3.0, self.duration, 4.5)
        for bt in blink_times:
            b_start = int(bt * self.sfreq)
            b_len = int(0.25 * self.sfreq)
            if b_start + b_len < self.n_samples:
                blink_shape = np.hanning(b_len) * 75.0
                for fp in ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF8']:
                    if fp in self.channel_names:
                        idx = self.channel_names.index(fp)
                        self.raw_data[idx, b_start:b_start+b_len] += blink_shape

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
        self.apply_ica_cleaning()

    def _compute_fast_ica(self):
        if not SKLEARN_AVAILABLE:
            self.ica_components = None
            return
        try:
            n_comp = min(16, self.filtered_data.shape[0])
            ica = FastICA(n_components=n_comp, random_state=42, max_iter=200)
            sub_data = self.filtered_data[:, :min(8000, self.n_samples)].T
            ica.fit(sub_data)
            self.ica_mixing = ica.mixing_
            self.ica_components = ica.transform(self.filtered_data.T).T
            print("[INFO] FastICA computed successfully (16 components).")
        except Exception as e:
            print(f"[WARN] FastICA initialization: {e}")
            self.ica_components = None

    def apply_ica_cleaning(self):
        if self.ica_components is None or self.ica_mixing is None:
            self.cleaned_data = self.filtered_data.copy()
            return

        comps = self.ica_components.copy()
        if self.suppress_oc0 and comps.shape[0] > 0:
            comps[0, :] = 0.0
        if self.suppress_oc1 and comps.shape[0] > 1:
            comps[1, :] = 0.0
        if self.suppress_emg and comps.shape[0] > 2:
            comps[2, :] = 0.0

        reconstructed = np.dot(self.ica_mixing, comps)
        self.cleaned_data = reconstructed.astype(np.float32)

    def get_window_data(self, current_sample: int, window_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        win_samples = int(window_sec * self.sfreq)
        start_idx = current_sample - win_samples
        end_idx = current_sample

        t_axis = np.linspace(-window_sec, 0.0, win_samples, endpoint=False)
        
        if start_idx < 0:
            n_pad = -start_idx
            valid_slice = self.cleaned_data[:, 0:max(1, end_idx)]
            pad_block = np.repeat(self.cleaned_data[:, [0]], n_pad, axis=1)
            full_block = np.hstack([pad_block, valid_slice])
            if full_block.shape[1] > win_samples:
                full_block = full_block[:, -win_samples:]
            return t_axis, full_block
        else:
            return t_axis, self.cleaned_data[:, start_idx:end_idx]

    def compute_welch_psd(self, channel_indices: List[int], current_sample: int, window_sec: float = 4.0) -> Tuple[np.ndarray, np.ndarray, float]:
        win_samples = int(window_sec * self.sfreq)
        start_idx = max(0, current_sample - win_samples)
        segment = self.cleaned_data[:, start_idx:current_sample]

        if segment.shape[1] < int(self.sfreq * 0.5):
            freqs = np.linspace(1.0, 45.0, 50)
            psd = np.ones(50) * 0.1
            return freqs, psd, 10.0

        if len(channel_indices) == 0:
            target_data = np.mean(segment, axis=0)
        else:
            target_data = np.mean(segment[channel_indices, :], axis=0)

        nperseg = min(len(target_data), int(self.sfreq * 2.0))
        freqs, psd = scipy.signal.welch(target_data, fs=self.sfreq, nperseg=nperseg)

        mask = (freqs >= 1.0) & (freqs <= 45.0)
        freqs_sub = freqs[mask]
        psd_sub = psd[mask]

        alpha_mask = (freqs_sub >= 8.0) & (freqs_sub <= 13.0)
        paf = float(freqs_sub[alpha_mask][np.argmax(psd_sub[alpha_mask])]) if np.any(alpha_mask) else 10.0

        return freqs_sub, psd_sub, paf

    def compute_erp(self, channel_idx: int, t_min: float = -0.5, t_max: float = 2.5) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        results = {}
        n_pre = int(abs(t_min) * self.sfreq)
        n_post = int(t_max * self.sfreq)
        n_epoch = n_pre + n_post
        time_axis = np.linspace(t_min, t_max, n_epoch)

        for eid in ['T0', 'T1', 'T2']:
            matching_events = [ev for ev in self.events if ev.event_id == eid]
            epochs = []
            for ev in matching_events:
                s_idx = ev.start_sample
                ep_start = s_idx - n_pre
                ep_end = s_idx + n_post
                if ep_start >= 0 and ep_end <= self.n_samples:
                    epoch = self.cleaned_data[channel_idx, ep_start:ep_end]
                    baseline = np.mean(epoch[:n_pre])
                    epochs.append(epoch - baseline)

            if len(epochs) > 0:
                epochs_arr = np.array(epochs)
                mean_curve = np.mean(epochs_arr, axis=0)
                sem_curve = np.std(epochs_arr, axis=0) / max(1.0, math.sqrt(len(epochs)))
            else:
                mean_curve = np.zeros(n_epoch)
                sem_curve = np.zeros(n_epoch)

            results[eid] = (time_axis, mean_curve, sem_curve)

        return results


# ==============================================================================
# PLAYBACK ENGINE WITH WALL-CLOCK DELTA-TIME SYNC
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
        self.timer.setInterval(30)
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
# 2D TOPOMAP WIDGET
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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        self.plot_widget.setRange(xRange=[-1.25, 1.25], yRange=[-1.25, 1.25])

        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.img_item.setZValue(-10)

        self._draw_head_schematic()

        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array([
            [30, 136, 229, 235],
            [129, 212, 250, 210],
            [255, 255, 255, 180],
            [255, 138, 101, 210],
            [229, 57, 53, 235]
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(self.lut)

        self.scatter = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen('#1DB954', width=1.5), brush=pg.mkBrush('#14171E'), hoverable=True
        )
        self.scatter.sigClicked.connect(self._on_electrode_clicked)
        self.plot_widget.addItem(self.scatter)

        self.label_items: Dict[str, pg.TextItem] = {}
        for ch in self.data_loader.channel_names:
            if ch in MONTAGE_2D_COORDS:
                x, y = MONTAGE_2D_COORDS[ch]
                lbl = pg.TextItem(ch, color='#7A889B', anchor=(0.5, 1.3))
                lbl.setFont(QFont("Segoe UI", 7, QFont.Bold))
                self.plot_widget.addItem(lbl)
                lbl.setPos(x, y)
                self.label_items[ch] = lbl

        layout.addWidget(self.plot_widget)

    def _draw_head_schematic(self):
        theta = np.linspace(0, 2 * np.pi, 200)
        r = 1.0
        head_curve = pg.PlotCurveItem(r * np.cos(theta), r * np.sin(theta), pen=pg.mkPen('#303846', width=2.5))
        self.plot_widget.addItem(head_curve)

        nose_curve = pg.PlotCurveItem(np.array([-0.14, 0.0, 0.14]), np.array([0.98, 1.15, 0.98]), pen=pg.mkPen('#303846', width=2.5))
        self.plot_widget.addItem(nose_curve)

        self.plot_widget.addItem(pg.PlotCurveItem(np.array([-0.99, -1.08, -1.08, -0.99]), np.array([0.15, 0.08, -0.08, -0.15]), pen=pg.mkPen('#303846', width=2.0)))
        self.plot_widget.addItem(pg.PlotCurveItem(np.array([0.99, 1.08, 1.08, 0.99]), np.array([0.15, 0.08, -0.08, -0.15]), pen=pg.mkPen('#303846', width=2.0)))

    def _precompute_idw_matrix(self):
        self.grid_res = 64
        x = np.linspace(-1.05, 1.05, self.grid_res)
        y = np.linspace(-1.05, 1.05, self.grid_res)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.mask_circle = (self.grid_x**2 + self.grid_y**2) <= 1.02

        self.node_positions = []
        self.valid_ch_names = []
        for ch in self.data_loader.channel_names:
            if ch in MONTAGE_2D_COORDS:
                self.node_positions.append(MONTAGE_2D_COORDS[ch])
                self.valid_ch_names.append(ch)

        coords_arr = np.array(self.node_positions)
        grid_pts = np.vstack([self.grid_x.ravel(), self.grid_y.ravel()]).T

        diff = grid_pts[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-4
        weights = 1.0 / (dist**2.5)
        weights /= np.sum(weights, axis=1, keepdims=True)
        self.idw_matrix = weights.astype(np.float32)

        spots = []
        for i, (nx, ny) in enumerate(coords_arr):
            spots.append({'pos': (nx, ny), 'data': self.valid_ch_names[i]})
        self.scatter.setData(spots)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

    def update_voltage_frame(self, sample_idx: int):
        voltages = self.data_loader.cleaned_data[:len(self.valid_ch_names), sample_idx]
        grid_flat = np.dot(self.idw_matrix, voltages)
        grid_2d = grid_flat.reshape((self.grid_res, self.grid_res))
        grid_2d[~self.mask_circle] = np.nan

        v_norm = np.clip((grid_2d + self.v_scale) / (2.0 * self.v_scale), 0.0, 1.0)
        img_bytes = (v_norm * 255.0).astype(np.uint8)
        self.img_item.setImage(img_bytes, levels=(0, 255), autoLevels=False)

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
# 3D BRAIN VIEWPORT WIDGET (PyVistaQt)
# ==============================================================================

class Brain3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if PYVISTA_AVAILABLE:
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter)
            self._load_3d_model()
            self._set_camera_view()
        else:
            lbl = QLabel("PyVista / PyVistaQt not available.\n3D Brain Viewport Disabled.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #7A889B; font-size: 13px; background: #0E0F14;")
            layout.addWidget(lbl)

    def _load_3d_model(self):
        self.plotter.set_background('#0E0F14')
        model_path = os.path.join("data", "human-brain.glb")
        
        mesh = None
        if os.path.exists(model_path):
            try:
                print(f"[INFO] Loading 3D model from {model_path}...")
                mesh = pv.read(model_path)
            except Exception as e:
                print(f"[WARN] Failed to read {model_path}: {e}")

        if mesh is None:
            mesh = pv.ParametricEllipsoid(xradius=1.8, yradius=1.4, zradius=1.3)
            mesh.points += 0.05 * np.sin(mesh.points * 8.0)

        self.brain_mesh = mesh
        self.plotter.add_mesh(
            self.brain_mesh,
            color='#5C6B7D',
            smooth_shading=True,
            specular=0.4,
            opacity=0.92
        )
        self._add_3d_electrodes()

    def _add_3d_electrodes(self):
        pts = []
        for ch in STANDARD_64_CHANNELS[:64]:
            if ch in MONTAGE_2D_COORDS:
                x2, y2 = MONTAGE_2D_COORDS[ch]
                r_sq = x2**2 + y2**2
                z3 = math.sqrt(max(0.01, 1.0 - min(0.95, r_sq))) * 1.2
                pts.append([y2 * 1.6, z3 * 1.3, x2 * 1.3])

        if len(pts) > 0:
            cloud = pv.PolyData(np.array(pts))
            spheres = cloud.glyph(geom=pv.Sphere(radius=0.06), orient=False)
            self.plotter.add_mesh(spheres, color='#1DB954', specular=0.8)

    def _set_camera_view(self):
        self.plotter.camera_position = [(0.0, 5.6, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.plotter.reset_camera_clipping_range()


# ==============================================================================
# WORKSTATION TAB 1: CASCADING WAVEFORMS
# ==============================================================================

class WaveformsWidget(QWidget):
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.active_channels: List[str] = []
        self.gain = 100.0
        self.win_sec = 6.0

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel('bottom', "Time (seconds relative to playhead)", **{'color': '#7A889B', 'font-size': '10pt'})
        self.plot_widget.getAxis('left').setStyle(showValues=False)
        self.plot_widget.setMouseEnabled(x=False, y=False)

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

            lbl = pg.TextItem(ch, color=c_hex, anchor=(1.0, 0.5))
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setPos(-self.win_sec + 0.1, i)
            self.plot_widget.addItem(lbl)
            self.channel_labels[ch] = lbl

    def update_frame(self, current_sample: int):
        if len(self.active_channels) == 0:
            return

        t_axis, block = self.data_loader.get_window_data(current_sample, self.win_sec)
        
        for i, ch in enumerate(self.active_channels):
            ch_idx = self.data_loader.channel_names.index(ch)
            raw_sig = block[ch_idx]
            norm_sig = (raw_sig / self.gain) + i
            self.curves[ch].setData(t_axis, norm_sig)
            self.channel_labels[ch].setPos(-self.win_sec + 0.15, i)


# ==============================================================================
# WORKSTATION TAB 2: WELCH PERIODOGRAM & LIVE FFT SPECTROGRAM
# ==============================================================================

class SpectralWidget(QWidget):
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.active_channels: List[str] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        self.paf_badge = QLabel("Peak Alpha Frequency: -- Hz (0.0 µV²/Hz)")
        self.paf_badge.setStyleSheet("""
            background: #18202A; color: #00E5FF; border: 1px solid #00E5FF;
            border-radius: 4px; padding: 4px 10px; font-weight: bold; font-size: 11px;
        """)
        top_row.addWidget(QLabel("WELCH PERIODOGRAM (1 - 45 Hz)"))
        top_row.addStretch()
        top_row.addWidget(self.paf_badge)
        layout.addLayout(top_row)

        self.psd_plot = pg.PlotWidget()
        self.psd_plot.setBackground('#0E0F14')
        self.psd_plot.showGrid(x=True, y=True, alpha=0.15)
        self.psd_plot.setLabel('left', "PSD (µV²/Hz)", **{'color': '#7A889B'})
        self.psd_plot.setLabel('bottom', "Frequency (Hz)", **{'color': '#7A889B'})
        self.psd_plot.setXRange(1.0, 45.0)

        bands = [
            (1.0, 4.0, '#311B92', 'Delta'),
            (4.0, 8.0, '#0D47A1', 'Theta'),
            (8.0, 13.0, '#004D40', 'Alpha'),
            (13.0, 30.0, '#E65100', 'Beta'),
            (30.0, 45.0, '#880E4F', 'Gamma')
        ]
        for f_low, f_high, c_hex, bname in bands:
            region = pg.LinearRegionItem(
                [f_low, f_high], movable=False,
                brush=QBrush(QColor(c_hex + '40')), pen=pg.mkPen(QColor(c_hex), width=0.8)
            )
            self.psd_plot.addItem(region)

        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen('#00FFA3', width=2.0))
        layout.addWidget(self.psd_plot, stretch=3)

        layout.addWidget(QLabel("LIVE ROLLING SPECTROGRAM (0 - 45 Hz)"))
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setBackground('#0E0F14')
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        self.spec_plot.setLabel('left', "Freq (Hz)", **{'color': '#7A889B'})
        self.spec_plot.setLabel('bottom', "Recent Time (sec)", **{'color': '#7A889B'})

        self.spec_history = np.zeros((100, 45), dtype=np.float32)
        pos = np.array([0.0, 0.3, 0.7, 1.0])
        colors = np.array([
            [14, 15, 20, 255],
            [0, 150, 136, 255],
            [255, 179, 0, 255],
            [255, 61, 0, 255]
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, colors)
        self.spec_img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

        layout.addWidget(self.spec_plot, stretch=2)

    def update_frame(self, current_sample: int, selected_channels: List[str]):
        ch_indices = [
            self.data_loader.channel_names.index(ch)
            for ch in selected_channels if ch in self.data_loader.channel_names
        ]
        freqs, psd, paf = self.data_loader.compute_welch_psd(ch_indices, current_sample)

        self.psd_curve.setData(freqs, psd)
        max_psd = np.max(psd) if len(psd) > 0 else 0.0
        self.paf_badge.setText(f"Peak Alpha Frequency: {paf:.1f} Hz ({max_psd:.1f} µV²/Hz)")

        fft_slice = np.interp(np.linspace(1, 45, 45), freqs, psd)
        self.spec_history = np.roll(self.spec_history, -1, axis=0)
        self.spec_history[-1, :] = np.log10(np.maximum(1e-4, fft_slice))
        self.spec_img.setImage(self.spec_history.T, autoLevels=True)


# ==============================================================================
# WORKSTATION TAB 3: EVENT-RELATED POTENTIAL (ERP) AVERAGING
# ==============================================================================

class ERPWidget(QWidget):
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Channel Selection:"))
        self.ch_combo = QComboBox()
        self.ch_combo.addItems(['C3', 'Cz', 'C4', 'Fz', 'Pz', 'Fp1', 'O1'])
        self.ch_combo.currentTextChanged.connect(self.recompute_erp)
        top_bar.addWidget(self.ch_combo)

        top_bar.addSpacing(20)
        recompute_btn = QPushButton("↻ Recalculate ERP Epochs")
        recompute_btn.clicked.connect(self.recompute_erp)
        top_bar.addWidget(recompute_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel('bottom', "Time relative to stimulus onset (seconds)", **{'color': '#7A889B'})
        self.plot_widget.setLabel('left', "Potential (µV)", **{'color': '#7A889B'})
        self.plot_widget.addLegend(offset=(10, 10))

        onset_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen('#FF5252', width=1.5, style=Qt.DashLine))
        self.plot_widget.addItem(onset_line)

        self.curve_t1 = self.plot_widget.plot(name="T1: Left Fist", pen=pg.mkPen('#00FFA3', width=2.5))
        self.curve_t2 = self.plot_widget.plot(name="T2: Right Fist", pen=pg.mkPen('#FF3DF0', width=2.5))
        self.curve_t0 = self.plot_widget.plot(name="T0: Rest Baseline", pen=pg.mkPen('#7A889B', width=1.8, style=Qt.DotLine))

        layout.addWidget(self.plot_widget)
        self.recompute_erp()

    def recompute_erp(self):
        ch_name = self.ch_combo.currentText()
        if ch_name not in self.data_loader.channel_names:
            return
        ch_idx = self.data_loader.channel_names.index(ch_name)
        results = self.data_loader.compute_erp(ch_idx)

        t_axis, t1_mean, _ = results['T1']
        _, t2_mean, _ = results['T2']
        _, t0_mean, _ = results['T0']

        self.curve_t1.setData(t_axis, t1_mean)
        self.curve_t2.setData(t_axis, t2_mean)
        self.curve_t0.setData(t_axis, t0_mean)


# ==============================================================================
# WORKSTATION TAB 4: SESSION BOOKMARKS & ANNOTATION TOOL
# ==============================================================================

class BookmarksWidget(QWidget):
    seek_requested = pyqtSignal(float)
    bookmark_added = pyqtSignal(Bookmark)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.bookmarks: List[Bookmark] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        bar = QHBoxLayout()
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Enter annotation note (or press 'M' during playback)...")
        bar.addWidget(self.note_edit, stretch=3)

        add_btn = QPushButton("+ Add Bookmark (M)")
        add_btn.clicked.connect(self._on_add_clicked)
        bar.addWidget(add_btn)

        export_btn = QPushButton("⭳ Export CSV")
        export_btn.clicked.connect(self.export_csv)
        bar.addWidget(export_btn)
        layout.addLayout(bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Timestamp", "Stimulus", "Annotation Note"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)

    def add_bookmark(self, current_time: float, current_stimulus: str, note: str = ""):
        if not note.strip():
            note = f"Session marker at {current_time:.2f}s"
        b_id = len(self.bookmarks) + 1
        bm = Bookmark(b_id, current_time, current_stimulus, note)
        self.bookmarks.append(bm)

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(bm.id)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{bm.timestamp:06.2f}s"))
        self.table.setItem(row, 2, QTableWidgetItem(bm.stimulus))
        self.table.setItem(row, 3, QTableWidgetItem(bm.note))

        self.note_edit.clear()
        self.bookmark_added.emit(bm)

    def _on_add_clicked(self):
        pass

    def _on_row_double_clicked(self, row, col):
        if row < len(self.bookmarks):
            self.seek_requested.emit(self.bookmarks[row].timestamp)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Bookmarks", "session_bookmarks.csv", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Timestamp_Sec", "Active_Stimulus", "Note"])
                for bm in self.bookmarks:
                    writer.writerow([bm.id, f"{bm.timestamp:.3f}", bm.stimulus, bm.note])
            print(f"[INFO] Exported {len(self.bookmarks)} bookmarks to {path}")


# ==============================================================================
# WORKSTATION TABS CONTAINER
# ==============================================================================

class WorkstationTabsWidget(QTabWidget):
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader

        self.waveforms_tab = WaveformsWidget(data_loader, self)
        self.spectral_tab = SpectralWidget(data_loader, self)
        self.erp_tab = ERPWidget(data_loader, self)
        self.bookmarks_tab = BookmarksWidget(data_loader, self)

        self.addTab(self.waveforms_tab, "📈 Cascading Waveforms")
        self.addTab(self.spectral_tab, "📊 Welch Periodogram & Live FFT")
        self.addTab(self.erp_tab, "🧠 Event-Related Potentials (ERP)")
        self.addTab(self.bookmarks_tab, "🔖 Session Bookmarks")


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
        self.setFixedHeight(38)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(x=False, y=False)
        self.setXRange(0.0, self.data_loader.duration, padding=0.01)
        self.setYRange(-0.5, 0.5)

        rail = pg.PlotCurveItem([0.0, self.data_loader.duration], [0.0, 0.0], pen=pg.mkPen('#2A313E', width=4.0))
        self.addItem(rail)

        for ev in self.data_loader.events:
            cfg = EVENT_COLOR_MAP.get(ev.event_id, {'bg': '#1E2530', 'border': '#7A889B'})
            brush_c = QColor(cfg['bg'])
            brush_c.setAlpha(190)
            region = pg.LinearRegionItem(
                [ev.start_time, ev.end_time], movable=False,
                brush=QBrush(brush_c), pen=pg.mkPen(cfg['border'], width=1.0)
            )
            self.addItem(region)

        self.playhead_line = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False,
            pen=pg.mkPen('#1DB954', width=2.5)
        )
        self.addItem(self.playhead_line)

        self.playhead_dot = pg.ScatterPlotItem(
            [0.0], [0.0], size=16,
            brush=pg.mkBrush('#1DB954'), pen=pg.mkPen('#FFFFFF', width=2.0)
        )
        self.addItem(self.playhead_dot)

    def update_playhead(self, time_sec: float):
        if not self.is_dragging:
            self.playhead_line.setValue(time_sec)
            self.playhead_dot.setData([time_sec], [0.0])

    def add_bookmark_marker(self, time_sec: float):
        mark = pg.InfiniteLine(
            pos=time_sec, angle=90, movable=False,
            pen=pg.mkPen('#FFD600', width=1.5, style=Qt.DashLine)
        )
        self.addItem(mark)

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
            self.playhead_dot.setData([t_sec], [0.0])
            self.seek_requested.emit(t_sec)


# ==============================================================================
# SPOTIFY PLAYBACK BAR (WITH GLOWING STIMULUS BADGES)
# ==============================================================================

class SpotifyPlaybackBar(QFrame):
    def __init__(self, data_loader: EEGDataLoader, engine: PlaybackEngine, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.engine = engine

        self._init_ui()

    def _init_ui(self):
        self.setFixedHeight(68)
        self.setStyleSheet("""
            QFrame {
                background: #08080C;
                border-top: 1px solid #1A1F2B;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(14)

        self.step_back_btn = QPushButton("⏮")
        self.step_back_btn.setFixedSize(36, 36)
        self.step_back_btn.setToolTip("Step Back (1 frame / 0.1s)")
        self.step_back_btn.clicked.connect(lambda: self.engine.step(-int(self.data_loader.sfreq * 0.1)))

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(48, 48)
        self.play_btn.setToolTip("Play / Pause (Spacebar)")
        self.play_btn.setStyleSheet("""
            QPushButton {
                background: #1DB954;
                color: #000000;
                font-size: 20px;
                font-weight: bold;
                border-radius: 24px;
            }
            QPushButton:hover {
                background: #1ED760;
            }
            QPushButton:pressed {
                background: #169C46;
            }
        """)
        self.play_btn.clicked.connect(self.engine.toggle_play)

        self.step_fwd_btn = QPushButton("⏭")
        self.step_fwd_btn.setFixedSize(36, 36)
        self.step_fwd_btn.setToolTip("Step Forward (1 frame / 0.1s)")
        self.step_fwd_btn.clicked.connect(lambda: self.engine.step(int(self.data_loader.sfreq * 0.1)))

        layout.addWidget(self.step_back_btn)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.step_fwd_btn)

        self.time_label = QLabel("00:00.0 / 02:05.0")
        self.time_label.setFixedWidth(120)
        self.time_label.setStyleSheet("color: #9AA7B7; font-family: 'Consolas', monospace; font-size: 11px;")
        layout.addWidget(self.time_label)

        # Center: Spotify Timeline Widget (Defined above this class)
        self.timeline = SpotifyTimelineWidget(self.data_loader, parent=self, engine=self.engine)
        self.timeline.seek_requested.connect(self.engine.seek_time)
        layout.addWidget(self.timeline, stretch=1)

        # Right: Glowing Active Stimulus Badges
        stim_layout = QHBoxLayout()
        stim_layout.setSpacing(8)

        self.badge_t0 = QLabel("T0: REST")
        self.badge_t1 = QLabel("T1: LEFT FIST")
        self.badge_t2 = QLabel("T2: RIGHT FIST")

        self.stim_badges = {
            'T0': self.badge_t0,
            'T1': self.badge_t1,
            'T2': self.badge_t2
        }

        for badge in self.stim_badges.values():
            badge.setFixedHeight(28)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(self._get_badge_style(active=False))
            stim_layout.addWidget(badge)

        layout.addLayout(stim_layout)

    def _get_badge_style(self, active: bool, color_key: str = 'T0') -> str:
        if not active:
            return """
                background: #14171E;
                color: #4A5668;
                border: 1px solid #202632;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 10px;
                font-weight: bold;
            """
        cfg = EVENT_COLOR_MAP.get(color_key, {'border': '#00FFA3', 'bg': '#0D332D'})
        return f"""
            background: {cfg['bg']};
            color: #FFFFFF;
            border: 2px solid {cfg['border']};
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: bold;
        """

    def update_playback_state(self, is_playing: bool):
        self.play_btn.setText("⏸" if is_playing else "▶")

    def update_frame(self, current_sample: int, current_time: float):
        total_time = self.data_loader.duration
        cur_min = int(current_time // 60)
        cur_sec = current_time % 60
        tot_min = int(total_time // 60)
        tot_sec = total_time % 60
        self.time_label.setText(f"{cur_min:02d}:{cur_sec:04.1f} / {tot_min:02d}:{tot_sec:04.1f}")
        self.timeline.update_playhead(current_time)

        active_eid: Optional[str] = None
        for ev in self.data_loader.events:
            if ev.start_time <= current_time <= ev.end_time:
                active_eid = ev.event_id
                break

        for eid, badge in self.stim_badges.items():
            is_active = (eid == active_eid)
            badge.setStyleSheet(self._get_badge_style(is_active, eid))


# ==============================================================================
# TOP HEADER & METADATA PILLS
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

        lbl_ica = QLabel("FASTICA ARTIFACT REMOVAL")
        lbl_ica.setStyleSheet("color: #7A889B; font-size: 10px; font-weight: bold; margin-top: 6px;")
        layout.addWidget(lbl_ica)

        self.chk_ic0 = QCheckBox("Ocular IC0 (Blinks)")
        self.chk_ic0.toggled.connect(self._on_ica_toggled)
        layout.addWidget(self.chk_ic0)

        self.chk_ic1 = QCheckBox("Saccadic IC1 (Eyes)")
        self.chk_ic1.toggled.connect(self._on_ica_toggled)
        layout.addWidget(self.chk_ic1)

        self.chk_emg = QCheckBox("Myographic IC2 (Muscle)")
        self.chk_emg.toggled.connect(self._on_ica_toggled)
        layout.addWidget(self.chk_emg)

        layout.addSpacing(10)

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

    def _on_ica_toggled(self):
        self.data_loader.suppress_oc0 = self.chk_ic0.isChecked()
        self.data_loader.suppress_oc1 = self.chk_ic1.isChecked()
        self.data_loader.suppress_emg = self.chk_emg.isChecked()
        self.data_loader.apply_ica_cleaning()
        self.settings_changed.emit()

    def _on_speed_changed(self, text: str):
        val = float(text.replace('x', ''))
        self.engine.set_speed(val)


# ==============================================================================
# MAIN APPLICATION WINDOW
# ==============================================================================

class NeuromapMainWindow(QMainWindow):
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

        self.header = HeaderWidget(self.data_loader, self)
        self.header.export_requested.connect(self._on_export_data)
        root_layout.addWidget(self.header)

        main_workspace = QHBoxLayout()
        main_workspace.setContentsMargins(0, 0, 0, 0)
        main_workspace.setSpacing(0)

        self.sidebar = ControlSidebarWidget(self.data_loader, self.engine, self)
        main_workspace.addWidget(self.sidebar)

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setHandleWidth(4)

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setHandleWidth(4)

        self.topomap_2d = Topomap2DWidget(self.data_loader, self)
        self.brain_3d = Brain3DWidget(self)

        self.top_splitter.addWidget(self.topomap_2d)
        self.top_splitter.addWidget(self.brain_3d)
        self.top_splitter.setSizes([960, 960])

        self.main_splitter.addWidget(self.top_splitter)

        self.workstation_tabs = WorkstationTabsWidget(self.data_loader, self)
        self.main_splitter.addWidget(self.workstation_tabs)
        self.main_splitter.setSizes([550, 480])

        main_workspace.addWidget(self.main_splitter, stretch=1)
        root_layout.addLayout(main_workspace, stretch=1)

        self.playback_bar = SpotifyPlaybackBar(self.data_loader, self.engine, self)
        root_layout.addWidget(self.playback_bar)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background: #060709; color: #7A889B; border-top: 1px solid #14171E;")
        self.status_label = QLabel("System Ready • All 64 EEG Channels Deselected • Real-Time Loop Active")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setValue(100)
        self.progress_bar.setTextVisible(False)
        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.engine.toggle_play)
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_M), self, activated=self._quick_bookmark)

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
            QTabWidget::pane {
                border: 1px solid #1A1F2C;
                background: #0E0F14;
            }
            QTabBar::tab {
                background: #12151D;
                color: #8C9BAE;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 600;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #1B212D;
                color: #00FFA3;
                border-bottom: 2px solid #00FFA3;
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
            QTableWidget {
                background: #0E0F14;
                gridline-color: #1A1F2C;
                border: none;
                font-size: 11px;
            }
            QHeaderView::section {
                background: #141720;
                color: #7A889B;
                padding: 4px;
                border: 1px solid #1A1F2C;
                font-weight: bold;
                font-size: 10px;
            }
        """)

    def _connect_signals(self):
        self.engine.frame_changed.connect(self._on_frame_update)
        self.engine.state_changed.connect(self.playback_bar.update_playback_state)

        self.topomap_2d.channel_toggled.connect(self._on_channel_toggled)

        self.sidebar.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.sidebar.win_slider.valueChanged.connect(self._on_window_changed)
        self.sidebar.topo_slider.valueChanged.connect(self._on_topo_scale_changed)

        self.workstation_tabs.bookmarks_tab.seek_requested.connect(self.engine.seek_time)
        self.workstation_tabs.bookmarks_tab.bookmark_added.connect(
            lambda bm: self.playback_bar.timeline.add_bookmark_marker(bm.timestamp)
        )

    def _on_frame_update(self, current_sample: int, current_time: float):
        self.topomap_2d.update_voltage_frame(current_sample)
        self.workstation_tabs.waveforms_tab.update_frame(current_sample)

        if self.workstation_tabs.currentIndex() == 1:
            self.workstation_tabs.spectral_tab.update_frame(current_sample, list(self.topomap_2d.selected_channels))

        self.playback_bar.update_frame(current_sample, current_time)

    def _on_channel_toggled(self, ch_name: str, is_active: bool):
        selected_list = sorted(list(self.topomap_2d.selected_channels))
        self.workstation_tabs.waveforms_tab.set_active_channels(selected_list)
        self.status_label.setText(f"Active Channels: {len(selected_list)} of 64 selected ({', '.join(selected_list[:6])}...)")

    def _on_gain_changed(self, val: int):
        self.workstation_tabs.waveforms_tab.gain = float(val)

    def _on_window_changed(self, val: int):
        self.workstation_tabs.waveforms_tab.win_sec = float(val)
        selected_list = sorted(list(self.topomap_2d.selected_channels))
        self.workstation_tabs.waveforms_tab.set_active_channels(selected_list)

    def _on_topo_scale_changed(self, val: int):
        self.topomap_2d.v_scale = float(val)

    def _quick_bookmark(self):
        cur_t = self.engine.current_sample / self.data_loader.sfreq
        active_eid = "Rest"
        for ev in self.data_loader.events:
            if ev.start_time <= cur_t <= ev.end_time:
                active_eid = ev.label
                break
        self.workstation_tabs.bookmarks_tab.add_bookmark(cur_t, active_eid, "Quick marker (M)")
        self.status_label.setText(f"Added bookmark at {cur_t:.2f}s [{active_eid}]")

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
                    row = [i, f"{t_val:.4f}"] + [f"{self.data_loader.cleaned_data[c, i]:.2f}" for c in range(64)]
                    writer.writerow(row)
            self.status_label.setText(f"Export completed: {path}")


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