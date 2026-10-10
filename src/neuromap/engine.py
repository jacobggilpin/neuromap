"""
neuromap - Core Engine Architecture (Phase 4.03)
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Components:
- CentralizedChannelState: Single source of truth for channel multi-selection across all views.
- SharedInterpolationCache: Precomputed 128x128 2D regularized multiquadric IDW matrix & anti-aliased edge mask.
- EEGDataLoader: Dataset ingestion, MNE-Python / procedural EEG, zero-phase DSP filters,
  and dynamic FastICA artifact removal engine.
- PlaybackEngine: High-precision wall-clock ticker.
"""

import os
import time
import math
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import scipy.signal

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

try:
    import mne
    from mne.datasets import eegbci
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# ==============================================================================
# AUTOMATIC PATH BOOTSTRAP (Supports root execution, src/neuromap layout & VS Code)
# ==============================================================================
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent
for _cand in [
    _root / 'src' / 'neuromap',
    _root / 'src',
    _root,
    _root.parent / 'src' / 'neuromap',
    _root.parent / 'src',
    _root.parent,
]:
    _cand_str = str(_cand)
    if _cand.is_dir() and _cand_str not in sys.path:
        sys.path.insert(0, _cand_str)

try:
    from neuromap.config import (
        STANDARD_64_CHANNELS, PALETTE_COLORS, EVENT_COLOR_MAP,
        MONTAGE_2D_COORDS, StimulusEvent
    )
except (ImportError, ModuleNotFoundError):
    from config import (
        STANDARD_64_CHANNELS, PALETTE_COLORS, EVENT_COLOR_MAP,
        MONTAGE_2D_COORDS, StimulusEvent
    )

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
# DATA LOADER & SIGNAL PROCESSING (WITH FASTICA ARTIFACT REMOVAL)
# ==============================================================================

class EEGDataLoader:
    """
    Loads MNE PhysioNet Motor Movement/Imagery dataset or generates high-fidelity
    synthetic 64-channel EEG with alpha rhythms and motor imagery desynchronization.
    Applies zero-phase Butterworth bandpass & notch filters.
    Includes FastICA spatial filtering engine for real-time artifact removal.
    """
    def __init__(self):
        self.sfreq: float = 160.0
        self.channel_names: List[str] = STANDARD_64_CHANNELS.copy()
        self.raw_data: Optional[np.ndarray] = None
        self.filtered_data: Optional[np.ndarray] = None
        self.base_filtered_data: Optional[np.ndarray] = None
        self.clean_data: Optional[np.ndarray] = None
        self.n_channels: int = 64
        self.n_samples: int = 0
        self.duration: float = 0.0
        self.events: List[StimulusEvent] = []
        self.channel_colors: Dict[str, str] = {}

        # FastICA Artifact Removal Attributes
        self.ica_enabled: bool = False
        self.ica_preset: str = "Ocular / Blinks (IC0)"
        self.ica_sources: Optional[np.ndarray] = None
        self.ica_mixing: Optional[np.ndarray] = None
        self.ica_unmixing: Optional[np.ndarray] = None
        self.ica_mean: Optional[np.ndarray] = None
        self.spatial_proj: Optional[np.ndarray] = None
        self.n_ica_components: int = 16

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
                    return True
            except Exception as e:
                print(f"[WARN] MNE loading failed ({e}). Falling back to procedural 64-channel EEG.")

        self._generate_procedural_eeg()
        self.apply_dsp_filters()
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

        # Embed realistic ocular blink artifacts onto frontal channels (Fp1, Fpz, Fp2)
        blink_train = np.zeros(self.n_samples, dtype=np.float32)
        for b_time in np.arange(3.0, self.duration - 2.0, 6.5):
            b_idx = int(b_time * self.sfreq)
            dur_pts = int(0.35 * self.sfreq)
            if b_idx + dur_pts < self.n_samples:
                b_wave = np.hanning(dur_pts) * 75.0
                blink_train[b_idx:b_idx + dur_pts] += b_wave

        self.raw_data[0, :] += blink_train * 1.0  # Fp1
        self.raw_data[1, :] += blink_train * 1.15 # Fpz
        self.raw_data[2, :] += blink_train * 1.0  # Fp2

        # Embed lateral saccade artifacts onto F7 / F8
        saccade_train = np.zeros(self.n_samples, dtype=np.float32)
        for s_time in np.arange(5.0, self.duration - 3.0, 9.0):
            s_idx = int(s_time * self.sfreq)
            dur_pts = int(0.25 * self.sfreq)
            if s_idx + dur_pts < self.n_samples:
                s_wave = np.sin(np.linspace(0, np.pi, dur_pts)) * 35.0
                self.raw_data[8, s_idx:s_idx + dur_pts] += s_wave   # F7
                self.raw_data[16, s_idx:s_idx + dur_pts] -= s_wave  # F8

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

        # Vectorized filtering along axis=-1 (compiled SciPy C routine across all 64 channels)
        s_bp = scipy.signal.filtfilt(b_band, a_band, self.raw_data, axis=-1)
        self.filtered_data = scipy.signal.filtfilt(b_notch, a_notch, s_bp, axis=-1).astype(np.float32)

        self.base_filtered_data = self.filtered_data.copy()
        self._compute_gfp_envelope()
        
        # Decompose FastICA
        self.decompose_fastica()

    def decompose_fastica(self, n_components: int = 16):
        """
        Dynamic FastICA Spatial Filter Engine:
        Decomposes the 64-channel EEG array into statistically independent components
        using fixed-point iteration. Automatically ranks components:
        - IC0: Frontal vertical EOG / blinks (highest absolute frontal loading)
        - IC1: Lateral anterior saccades (highest horizontal differential |F7 - F8|)
        - IC2: Temporal high-frequency EMG muscle activity
        """
        print(f"[INFO] Running FastICA Spatial Decomposition ({n_components} components)...")
        t0 = time.perf_counter()

        data = self.base_filtered_data if self.base_filtered_data is not None else self.raw_data
        n_channels, n_samples = data.shape
        mean_X = np.mean(data, axis=1, keepdims=True)
        Xc = data - mean_X

        # Subsample if long for sub-second decomposition
        max_pts = min(n_samples, int(self.sfreq * 60))
        step = max(1, n_samples // max_pts)
        Xc_sub = Xc[:, ::step]
        n_sub = Xc_sub.shape[1]

        # 1. PCA Whitening
        cov = np.dot(Xc_sub, Xc_sub.T) / (n_sub - 1)
        d, E = np.linalg.eigh(cov)
        idx = np.argsort(d)[::-1][:n_components]
        d = np.maximum(d[idx], 1e-6)
        E = E[:, idx]

        K = np.dot(np.diag(1.0 / np.sqrt(d)), E.T)
        Z = np.dot(K, Xc_sub)

        # 2. FastICA Fixed-Point Iteration (Hyvärinen 1999)
        W = np.zeros((n_components, n_components), dtype=np.float32)
        np.random.seed(42)
        for j in range(n_components):
            w = np.random.randn(n_components)
            w /= np.linalg.norm(w)
            for it in range(40):
                w_old = w.copy()
                u = np.dot(w, Z)
                tanh_u = np.tanh(u)
                w1 = np.mean(Z * tanh_u, axis=1) - np.mean(1.0 - tanh_u**2) * w
                if j > 0:
                    w1 -= np.dot(W[:j, :].T, np.dot(W[:j, :], w1))
                w1_norm = np.linalg.norm(w1)
                if w1_norm > 1e-8:
                    w = w1 / w1_norm
                if abs(abs(np.dot(w, w_old)) - 1.0) < 1e-3:
                    break
            W[j, :] = w

        W_unmix = np.dot(W, K)
        A_mix = np.linalg.pinv(W_unmix)
        S = np.dot(W_unmix, Xc)

        # 3. Automated Component Classification & Ranking
        # IC0: Frontal blink / vertical EOG (max absolute weight on Fp1, Fpz, Fp2)
        frontal_scores = np.sum(np.abs(A_mix[:3, :]), axis=0)
        ic0_idx = int(np.argmax(frontal_scores))

        # IC1: Horizontal saccades (max difference between F7 [index 8] and F8 [index 16])
        lat_scores = np.abs(A_mix[8, :] - A_mix[16, :])
        lat_scores[ic0_idx] = -1.0
        ic1_idx = int(np.argmax(lat_scores))

        # IC2: High-frequency temporal muscle EMG (T7, T8, T9, T10: indices 26, 34, 35, 36)
        temp_indices = [26, 34, 35, 36]
        temp_scores = np.sum(np.abs(A_mix[temp_indices, :]), axis=0)
        temp_scores[ic0_idx] = -1.0
        temp_scores[ic1_idx] = -1.0
        ic2_idx = int(np.argmax(temp_scores))

        # Re-order components so IC0, IC1, IC2 are first
        order = [ic0_idx, ic1_idx, ic2_idx] + [i for i in range(n_components) if i not in (ic0_idx, ic1_idx, ic2_idx)]
        self.ica_mixing = A_mix[:, order].astype(np.float32)
        self.ica_unmixing = W_unmix[order, :].astype(np.float32)
        self.ica_sources = S[order, :].astype(np.float32)
        self.ica_mean = mean_X.astype(np.float32)

        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[SUCCESS] FastICA completed in {dt:.1f}ms. Ranked: IC0=Blink, IC1=Saccade, IC2=EMG.")
        
        self.apply_ica_rejection(self.ica_preset)

    def apply_ica_rejection(self, preset: str = "Ocular / Blinks (IC0)"):
        """
        Reconstructs EEG with rejected artifact ICs removed:
        - 'Ocular / Blinks (IC0)': Reconstructs without IC0.
        - 'Ocular + Saccades (IC0, IC1)': Reconstructs without IC0 and IC1.
        - 'Aggressive EOG + EMG (IC0, IC1, IC2)': Reconstructs without IC0, IC1, IC2.
        """
        self.ica_preset = preset
        if self.ica_sources is None or self.ica_mixing is None:
            return

        if preset == "Ocular / Blinks (IC0)":
            reject_indices = [0]
        elif preset == "Ocular + Saccades (IC0, IC1)":
            reject_indices = [0, 1]
        elif preset == "Aggressive EOG + EMG (IC0, IC1, IC2)":
            reject_indices = [0, 1, 2]
        else:
            reject_indices = [0]

        A_clean = self.ica_mixing.copy()
        for idx in reject_indices:
            if idx < A_clean.shape[1]:
                A_clean[:, idx] = 0.0

        # Closed-loop spatial projection operator: P = A_clean @ W (64x64 matrix)
        if self.ica_unmixing is not None:
            self.spatial_proj = np.dot(A_clean, self.ica_unmixing).astype(np.float32)

        S_clean = self.ica_sources.copy()
        for idx in reject_indices:
            if idx < S_clean.shape[0]:
                S_clean[idx, :] = 0.0

        # Reconstruct cleaned 64-channel array
        self.clean_data = (np.dot(self.ica_mixing, S_clean) + self.ica_mean).astype(np.float32)

        # Set active visualization buffer
        if self.ica_enabled:
            self.filtered_data = self.clean_data
        else:
            self.filtered_data = self.base_filtered_data

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
        self.playback_speed = max(0.05, min(1.0, speed))

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

    def step_frame_forward(self, delta_samples: int = 1):
        """Frame-by-Frame forward step (1 sample = 6.25ms at 160Hz)."""
        if self.is_playing:
            self.pause()
        self.seek_sample(self.current_sample + delta_samples)

    def step_frame_backward(self, delta_samples: int = 1):
        """Frame-by-Frame backward step (1 sample = 6.25ms at 160Hz)."""
        if self.is_playing:
            self.pause()
        self.seek_sample(self.current_sample - delta_samples)

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


