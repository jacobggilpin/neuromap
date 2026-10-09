"""
neuromap - Core Engine Architecture
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Components:
- CentralizedChannelState: Single source of truth for channel multi-selection across all views.
- SharedInterpolationCache: Precomputed 128x128 2D regularized multiquadric IDW matrix & anti-aliased edge mask.
- EEGDataLoader: Closed-loop streaming data loader with real-time FastICA & ASR spatial projection (P = A_clean @ W),
  Phase 4.03 standard sliding window slicing with np.hstack edge padding, and integrated Phase 5 Analytical Engines.
- LSLReceiverThread: LabStreamingLayer (pylsl) live hardware stream receiver inlet for plug-and-play EEG headsets.
- PlaybackEngine: High-precision wall-clock ticker with frame-by-frame stepping (1 sample = 6.25ms @ 160Hz).
"""

import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import scipy.signal

# PyQt5 Core
try:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
except ImportError:
    # Graceful dummy classes for headless/compile testing on environments without PyQt5
    class QObject:
        def __init__(self, *args, **kwargs): pass
    class QThread(QObject):
        def __init__(self, *args, **kwargs): super().__init__()
        def start(self): pass
        def wait(self, *args): pass
    def pyqtSignal(*args, **kwargs):
        class _Signal:
            def connect(self, slot): pass
            def emit(self, *args, **kwargs): pass
        return _Signal()
    class _QtMock:
        pass
    Qt = _QtMock()
    class QTimer:
        def __init__(self): self.timeout = pyqtSignal()
        def setInterval(self, val): pass
        def start(self): pass
        def stop(self): pass

# MNE-Python for real neurophysiology data
try:
    import mne
    from mne.datasets import eegbci
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# LabStreamingLayer for live EEG hardware streaming
try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

from config import (
    STANDARD_64_CHANNELS, PALETTE_COLORS, EVENT_COLOR_MAP,
    MONTAGE_2D_COORDS, StimulusEvent, SENSORIMOTOR_CHANNELS,
    FREQUENCY_BANDS
)
import analytics


# ==============================================================================
# CENTRALIZED CHANNEL STATE (SINGLE SOURCE OF TRUTH)
# ==============================================================================

class CentralizedChannelState(QObject):
    """
    Centralized Channel State:
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


# ==============================================================================
# SHARED INTERPOLATION CACHE (2D & 3D GEOMETRIC CACHES)
# ==============================================================================

class SharedInterpolationCache:
    """
    Shared Interpolation Matrix Cache:
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
# DATA STRUCTURES FOR FASTICA ARTIFACT WORKSTATION
# ==============================================================================

@dataclass
class IndependentComponentInfo:
    idx: int
    name: str
    tag: str
    tag_color: str
    kurtosis: float
    variance_pct: float
    trace_snippet: np.ndarray
    is_rejected: bool = False


# ==============================================================================
# CLOSED-LOOP EEG DATA LOADER
# ==============================================================================

class EEGDataLoader(QObject):
    """
    Closed-Loop EEG Data Engine:
    - Designed for live streaming input and real-time closed-loop neurofeedback/analysis.
    - Eliminates offline pre-computations on startup; algorithms operate on the active stream.
    - Dynamic FastICA Spatial Decomposition & Online Rejection Projection Engine (P = A_clean @ W).
    - Artifact Subspace Reconstruction (ASR) burst filter.
    - Phase 4.03 standard array window slicing with edge padding (no pre-padded zero-copy circular ring buffer).
    - Integrated Phase 5 Advanced Analytical Engines (PSD, ERSP, Connectivity, BCI Decoder).
    """
    data_reconstructed = pyqtSignal()
    live_chunk_ingested = pyqtSignal()

    def __init__(self):
        super().__init__()
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

        # FastICA Spatial Filtering Attributes (Calculated on demand, NOT at startup)
        self.ica_enabled: bool = False
        self.ica_preset: str = "Ocular / Blinks (IC0)"
        self.ica_sources: Optional[np.ndarray] = None
        self.ica_mixing: Optional[np.ndarray] = None
        self.ica_unmixing: Optional[np.ndarray] = None
        self.ica_mean: Optional[np.ndarray] = None
        self.spatial_projection_matrix: Optional[np.ndarray] = None
        self.n_ica_components: int = 16
        self.ic_components: List[IndependentComponentInfo] = []
        self.rejected_ic_indices: Set[int] = {0}

        # Artifact Subspace Reconstruction (ASR) Attributes
        self.asr_enabled: bool = False
        self.asr_cutoff_sd: float = 5.0
        self.asr_cleaned_windows_pct: float = 0.0

        for i, ch in enumerate(self.channel_names):
            self.channel_colors[ch] = PALETTE_COLORS[i % len(PALETTE_COLORS)]

        # Centralized State and Caches
        self.channel_state = CentralizedChannelState(self.channel_names)
        self.interpolation_cache = SharedInterpolationCache(self.channel_names)
        self.gfp_envelope: Optional[np.ndarray] = None

        # Phase 5: BCI Motor Imagery Decoder Pipeline
        self.bci_pipeline = analytics.BCIDecoderPipeline(sfreq=self.sfreq)
        
        # Real-time connectivity cache (to avoid recomputing expensive matrices every single tick)
        self._cached_conn_sample: int = -1
        self._cached_plv_matrix: Optional[np.ndarray] = None
        self._cached_coherence_matrix: Optional[np.ndarray] = None

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
                    
                    self.raw_data = data[:64, :].astype(np.float32) * 1e6
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
                    self.bci_pipeline.calibrate(self.filtered_data, self.events)
                    return True
            except Exception as e:
                print(f"[WARN] MNE loading failed ({e}). Falling back to procedural 64-channel EEG.")

        self._generate_procedural_eeg()
        self.apply_dsp_filters()
        self.bci_pipeline.calibrate(self.filtered_data, self.events)
        return True

    def _generate_procedural_eeg(self):
        self.sfreq = 160.0
        self.duration = 125.0
        self.n_samples = int(self.sfreq * self.duration)
        self.n_channels = 64
        self.channel_names = STANDARD_64_CHANNELS.copy()

        t = np.linspace(0, self.duration, self.n_samples, endpoint=False, dtype=np.float32)
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

        # 1. Embed realistic ocular blink artifacts onto frontal channels (Fp1, Fpz, Fp2)
        blink_train = np.zeros(self.n_samples, dtype=np.float32)
        for b_time in np.arange(3.0, self.duration - 2.0, 6.0):
            b_idx = int(b_time * self.sfreq)
            dur_pts = int(0.35 * self.sfreq)
            if b_idx + dur_pts < self.n_samples:
                b_wave = np.hanning(dur_pts) * 85.0
                blink_train[b_idx:b_idx + dur_pts] += b_wave

        self.raw_data[0, :] += blink_train * 1.0   # Fp1
        self.raw_data[1, :] += blink_train * 1.25  # Fpz
        self.raw_data[2, :] += blink_train * 1.0   # Fp2

        # 2. Embed lateral saccade artifacts onto F7 / F8
        for s_time in np.arange(5.0, self.duration - 3.0, 8.5):
            s_idx = int(s_time * self.sfreq)
            dur_pts = int(0.25 * self.sfreq)
            if s_idx + dur_pts < self.n_samples:
                s_wave = np.sin(np.linspace(0, np.pi, dur_pts)) * 38.0
                self.raw_data[8, s_idx:s_idx + dur_pts] += s_wave   # F7
                self.raw_data[16, s_idx:s_idx + dur_pts] -= s_wave  # F8

        # 3. Embed cardiac ECG QRS complexes (rhythmic ~72 BPM pulse)
        ecg_indices = np.arange(int(0.6 * self.sfreq), self.n_samples - 40, int(0.833 * self.sfreq))
        for p in ecg_indices:
            qrs = np.array([-4.0, 28.0, 48.0, -10.0, -2.0], dtype=np.float32)
            self.raw_data[26, p-2:p+3] += qrs * 0.7  # T7
            self.raw_data[34, p-2:p+3] += qrs * 0.7  # T8
            self.raw_data[61, p-2:p+3] += qrs * 0.9  # Oz
            self.raw_data[63, p-2:p+3] += qrs * 1.0  # Iz

        # 4. Embed transient high-variance motion bursts for ASR
        for m_time in [22.0, 65.0, 98.0]:
            m_idx = int(m_time * self.sfreq)
            m_pts = int(1.2 * self.sfreq)
            if m_idx + m_pts < self.n_samples:
                self.raw_data[:, m_idx:m_idx+m_pts] += np.random.randn(64, m_pts) * 55.0

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
        print(f"[INFO] Initializing Stream DSP Filters: Bandpass [{l_freq}-{h_freq} Hz], Notch [{notch_freq} Hz]...")
        self.filtered_data = np.zeros_like(self.raw_data)
        nyq = 0.5 * self.sfreq

        low = max(0.001, l_freq / nyq)
        high = min(0.999, h_freq / nyq)
        b_band, a_band = scipy.signal.butter(3, [low, high], btype='bandpass')

        w0 = notch_freq / nyq
        b_notch, a_notch = scipy.signal.iirnotch(w0, 30.0)

        s_bp = scipy.signal.filtfilt(b_band, a_band, self.raw_data, axis=-1)
        self.filtered_data = scipy.signal.filtfilt(b_notch, a_notch, s_bp, axis=-1).astype(np.float32)
        self.base_filtered_data = self.filtered_data.copy()

        self._compute_gfp_envelope()

    def get_active_data(self) -> np.ndarray:
        if (self.ica_enabled or self.asr_enabled) and self.clean_data is not None:
            return self.clean_data
        if self.filtered_data is not None:
            return self.filtered_data
        return self.raw_data

    def get_window_data(self, channel_idx: int, current_sample: int, n_samples: int) -> np.ndarray:
        data = self.get_active_data()
        if data is None or current_sample <= 0:
            return np.zeros(n_samples, dtype=np.float32)

        start_idx = current_sample - n_samples
        if start_idx < 0:
            pad_len = -start_idx
            chunk = data[channel_idx, 0:max(0, current_sample)]
            first_val = chunk[0] if len(chunk) > 0 else 0.0
            return np.hstack([np.full(pad_len, first_val, dtype=np.float32), chunk])
        else:
            return data[channel_idx, start_idx:current_sample]

    def get_channel_voltages(self, sample_idx: int) -> np.ndarray:
        data = self.get_active_data()
        if data is None or sample_idx < 0 or sample_idx >= self.n_samples:
            return np.zeros(self.n_channels, dtype=np.float32)
        return data[:, sample_idx]

    def ingest_live_chunk(self, chunk_64: np.ndarray):
        c = chunk_64.astype(np.float32)
        if self.raw_data is None:
            self.raw_data = c
            self.filtered_data = c.copy()
            self.base_filtered_data = c.copy()
        else:
            self.raw_data = np.hstack([self.raw_data, c])
            if self.base_filtered_data is not None:
                self.base_filtered_data = np.hstack([self.base_filtered_data, c])
                if self.ica_enabled and self.spatial_projection_matrix is not None:
                    c_clean = (np.dot(self.spatial_projection_matrix, c - self.ica_mean) + self.ica_mean).astype(np.float32)
                    self.clean_data = np.hstack([self.clean_data, c_clean]) if self.clean_data is not None else c_clean
                    self.filtered_data = self.clean_data
                else:
                    self.filtered_data = self.base_filtered_data
        self.n_channels, self.n_samples = self.raw_data.shape
        self.duration = self.n_samples / self.sfreq
        self.live_chunk_ingested.emit()

    def process_live_frame(self, raw_sample_64: np.ndarray) -> np.ndarray:
        x = raw_sample_64.astype(np.float32)
        if self.ica_enabled and self.spatial_projection_matrix is not None and self.ica_mean is not None:
            mu = self.ica_mean.ravel()
            x = np.dot(self.spatial_projection_matrix, (x - mu)) + mu
        return x

    def decompose_fastica(self, n_components: int = 16):
        print(f"[INFO] Running FastICA Spatial Decomposition ({n_components} components)...")
        t0 = time.perf_counter()

        data = self.base_filtered_data if self.base_filtered_data is not None else self.raw_data
        n_channels, n_samples = data.shape
        mean_X = np.mean(data, axis=1, keepdims=True)
        Xc = data - mean_X

        max_pts = min(n_samples, int(self.sfreq * 60))
        step = max(1, n_samples // max_pts)
        Xc_sub = Xc[:, ::step]
        n_sub = Xc_sub.shape[1]

        cov = np.dot(Xc_sub, Xc_sub.T) / (n_sub - 1)
        d, E = np.linalg.eigh(cov)
        idx = np.argsort(d)[::-1][:n_components]
        d = np.maximum(d[idx], 1e-6)
        E = E[:, idx]

        K = np.dot(np.diag(1.0 / np.sqrt(d)), E.T)
        Z = np.dot(K, Xc_sub)

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

        frontal_scores = np.sum(np.abs(A_mix[:3, :]), axis=0)
        ic0_idx = int(np.argmax(frontal_scores))

        lat_scores = np.abs(A_mix[8, :] - A_mix[16, :])
        lat_scores[ic0_idx] = -1.0
        ic1_idx = int(np.argmax(lat_scores))

        temp_indices = [26, 34, 35, 36]
        ecg_scores = np.zeros(n_components, dtype=np.float32)
        for c in range(n_components):
            if c not in (ic0_idx, ic1_idx):
                s_c = S[c, :]
                peaks, _ = scipy.signal.find_peaks(np.abs(s_c), distance=int(self.sfreq * 0.6), height=np.std(s_c)*2.0)
                if len(peaks) > 10:
                    diffs = np.diff(peaks)
                    if np.std(diffs) < (0.25 * np.mean(diffs)):
                        ecg_scores[c] = float(len(peaks))
        ic2_idx = int(np.argmax(ecg_scores)) if np.max(ecg_scores) > 0 else (2 if 2 not in (ic0_idx, ic1_idx) else 3)

        temp_scores = np.sum(np.abs(A_mix[temp_indices, :]), axis=0)
        temp_scores[ic0_idx] = -1.0
        temp_scores[ic1_idx] = -1.0
        temp_scores[ic2_idx] = -1.0
        ic3_idx = int(np.argmax(temp_scores))

        order = [ic0_idx, ic1_idx, ic2_idx, ic3_idx] + [i for i in range(n_components) if i not in (ic0_idx, ic1_idx, ic2_idx, ic3_idx)]
        self.ica_mixing = A_mix[:, order].astype(np.float32)
        self.ica_unmixing = W_unmix[order, :].astype(np.float32)
        self.ica_sources = S[order, :].astype(np.float32)
        self.ica_mean = mean_X.astype(np.float32)

        total_data_var = np.sum(np.var(data, axis=1)) + 1e-6
        self.ic_components.clear()

        snip_len = min(600, n_samples)
        for i in range(n_components):
            s_i = self.ica_sources[i, :]
            m4 = np.mean((s_i - np.mean(s_i))**4)
            m2 = np.var(s_i)
            kurt = float(m4 / (m2**2 + 1e-8) - 3.0)
            
            back_proj_var = np.sum(np.var(np.outer(self.ica_mixing[:, i], s_i), axis=1))
            var_pct = float(back_proj_var / total_data_var * 100.0)
            
            if i == 0:
                tag, color = "OCULAR BLINK", "#FF5252"
            elif i == 1:
                tag, color = "OCULAR SACCADE", "#00E5FF"
            elif i == 2:
                tag, color = "CARDIAC ECG", "#E040FB"
            elif i == 3:
                tag, color = "TEMPORAL EMG", "#FFD600"
            else:
                tag, color = "NEURAL CORTICAL", "#69F0AE"

            snip = s_i[:snip_len].copy()
            snip_norm = (snip - np.mean(snip)) / (np.std(snip) + 1e-6)
            
            self.ic_components.append(IndependentComponentInfo(
                idx=i,
                name=f"IC{i:02d}",
                tag=tag,
                tag_color=color,
                kurtosis=kurt,
                variance_pct=var_pct,
                trace_snippet=snip_norm,
                is_rejected=(i in self.rejected_ic_indices)
            ))

        self._update_spatial_projection()
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[SUCCESS] FastICA Spatial Decomposition completed in {dt:.1f}ms. Extracted {n_components} components.")

        if self.ica_enabled:
            self.apply_ica_rejection(self.ica_preset)

    def apply_ica_rejection(self, preset: str = "Ocular / Blinks (IC0)"):
        self.ica_preset = preset
        if preset == "Ocular / Blinks (IC0)":
            self.rejected_ic_indices = {0}
        elif preset == "Ocular + Saccades (IC0, IC1)":
            self.rejected_ic_indices = {0, 1}
        elif preset == "Aggressive EOG + EMG (IC0, IC1, IC2)":
            self.rejected_ic_indices = {0, 1, 2}
        else:
            self.rejected_ic_indices = {0}

        for comp in self.ic_components:
            comp.is_rejected = (comp.idx in self.rejected_ic_indices)

        self._reconstruct_signal()

    def toggle_ic_component(self, ic_idx: int, reject: bool):
        if reject:
            self.rejected_ic_indices.add(ic_idx)
        else:
            self.rejected_ic_indices.discard(ic_idx)

        for comp in self.ic_components:
            if comp.idx == ic_idx:
                comp.is_rejected = reject

        self._reconstruct_signal()

    def apply_asr(self, cutoff_sd: float = 5.0, win_len_sec: float = 0.5):
        data = self.clean_data if self.clean_data is not None else self.base_filtered_data
        n_channels, n_samples = data.shape
        win_len = int(win_len_sec * self.sfreq)
        step = win_len // 2

        covs = []
        for s in range(0, n_samples - win_len, win_len):
            chunk = data[:, s:s+win_len]
            c = np.dot(chunk, chunk.T) / (win_len - 1)
            covs.append(c)

        cov_clean = np.median(np.stack(covs, axis=0), axis=0)
        d_clean, V_clean = np.linalg.eigh(cov_clean)
        idx = np.argsort(d_clean)[::-1]
        d_clean = np.maximum(d_clean[idx], 1e-6)
        V_clean = V_clean[:, idx]
        thresholds = cutoff_sd * np.sqrt(d_clean)

        cleaned = data.copy()
        n_rej = 0
        tot = 0

        for s in range(0, n_samples - win_len, step):
            tot += 1
            chunk = data[:, s:s+win_len]
            proj = np.dot(V_clean.T, chunk)
            sd = np.std(proj, axis=1)
            bad_dims = sd > thresholds
            if np.any(bad_dims):
                n_rej += 1
                proj[bad_dims, :] *= (thresholds[bad_dims, np.newaxis] / (sd[bad_dims, np.newaxis] + 1e-6))
                cleaned[:, s:s+win_len] = np.dot(V_clean, proj)

        self.asr_cleaned_windows_pct = float(n_rej / max(1, tot) * 100.0)
        print(f"[SUCCESS] ASR Subspace Filtering: Cleaned {n_rej} of {tot} windows ({self.asr_cleaned_windows_pct:.1f}%).")
        return cleaned

    def _update_spatial_projection(self):
        if self.ica_mixing is None or self.ica_unmixing is None:
            self.spatial_projection_matrix = None
            return

        n_comp = self.ica_mixing.shape[1]
        keep = [i for i in range(n_comp) if i not in self.rejected_ic_indices]
        
        if len(keep) == 0:
            self.spatial_projection_matrix = np.zeros((self.n_channels, self.n_channels), dtype=np.float32)
        elif len(keep) == n_comp:
            self.spatial_projection_matrix = np.eye(self.n_channels, dtype=np.float32)
        else:
            A_k = self.ica_mixing[:, keep]
            W_k = self.ica_unmixing[keep, :]
            self.spatial_projection_matrix = np.dot(A_k, W_k).astype(np.float32)

    def _reconstruct_signal(self):
        if self.ica_sources is None or self.ica_mixing is None:
            if self.asr_enabled:
                self.clean_data = self.apply_asr(self.asr_cutoff_sd)
                self.filtered_data = self.clean_data
            else:
                self.filtered_data = self.base_filtered_data
            self._compute_gfp_envelope()
            self.data_reconstructed.emit()
            return

        self._update_spatial_projection()

        if self.spatial_projection_matrix is not None and self.base_filtered_data is not None:
            Xc = self.base_filtered_data - self.ica_mean
            reconstructed = (np.dot(self.spatial_projection_matrix, Xc) + self.ica_mean).astype(np.float32)
        else:
            S_clean = self.ica_sources.copy()
            for idx in self.rejected_ic_indices:
                if idx < S_clean.shape[0]:
                    S_clean[idx, :] = 0.0
            reconstructed = (np.dot(self.ica_mixing, S_clean) + self.ica_mean).astype(np.float32)

        if self.asr_enabled:
            reconstructed = self.apply_asr(self.asr_cutoff_sd)

        self.clean_data = reconstructed

        if self.ica_enabled or self.asr_enabled:
            self.filtered_data = self.clean_data
        else:
            self.filtered_data = self.base_filtered_data

        self._compute_gfp_envelope()
        self.data_reconstructed.emit()

    def _compute_gfp_envelope(self):
        if self.filtered_data is None:
            return
        mean_v = np.mean(self.filtered_data, axis=0)
        gfp = np.sqrt(np.mean((self.filtered_data - mean_v)**2, axis=0))
        target_pts = 1200
        step = max(1, self.n_samples // target_pts)
        gfp_sub = gfp[::step][:target_pts]
        denom = (np.max(gfp_sub) - np.min(gfp_sub)) + 1e-6
        self.gfp_envelope = ((gfp_sub - np.min(gfp_sub)) / denom) * 0.7 - 0.35

    # ==========================================================================
    # PHASE 5 ANALYTICAL RETRIEVAL WRAPPERS
    # ==========================================================================

    def get_sliding_window_64(self, current_sample: int, window_sec: float = 2.0) -> np.ndarray:
        data = self.get_active_data()
        n_pts = int(window_sec * self.sfreq)
        if data is None or current_sample <= 0:
            return np.zeros((self.n_channels, n_pts), dtype=np.float32)
        s0 = current_sample - n_pts
        if s0 < 0:
            pad_len = -s0
            chunk = data[:, 0:max(0, current_sample)]
            first_cols = chunk[:, [0]] if chunk.shape[1] > 0 else np.zeros((self.n_channels, 1), dtype=np.float32)
            pad = np.repeat(first_cols, pad_len, axis=1)
            return np.hstack([pad, chunk])
        return data[:, s0:current_sample]

    def get_psd_for_selection(
        self,
        current_sample: int,
        window_sec: float = 2.0,
        selected_channels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
        win_data = self.get_sliding_window_64(current_sample, window_sec)
        if selected_channels and len(selected_channels) > 0:
            indices = [self.channel_names.index(ch) for ch in selected_channels if ch in self.channel_names]
            if len(indices) > 0:
                win_data = win_data[indices, :]
        freqs, psd = analytics.compute_welch_psd(win_data, self.sfreq)
        bands = analytics.compute_band_powers(freqs, psd)
        return freqs, psd, bands

    def get_ersp_map(
        self,
        channel_name: str,
        event_id: str = 'T1'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.get_active_data()
        if data is None or channel_name not in self.channel_names:
            t = np.linspace(-1.0, 3.5, 72, dtype=np.float32)
            f = np.linspace(6.0, 32.0, 26, dtype=np.float32)
            return t, f, np.zeros((len(f), len(t)), dtype=np.float32)

        ch_idx = self.channel_names.index(channel_name)
        return analytics.compute_event_ersp(data, ch_idx, self.events, event_id, self.sfreq)

    def get_connectivity_matrix(
        self,
        current_sample: int,
        metric: str = "PLV",
        band_name: str = "Alpha",
        window_sec: float = 2.0
    ) -> Tuple[np.ndarray, List[Tuple[str, str, float, Tuple[float, float], Tuple[float, float]]]]:
        band = FREQUENCY_BANDS.get(band_name, (8.0, 12.0, ''))[:2]
        win_data = self.get_sliding_window_64(current_sample, window_sec)

        if metric == "PLV":
            conn_mat = analytics.compute_plv_matrix(win_data, self.sfreq, band=band)
        else:
            conn_mat = analytics.compute_spectral_coherence_matrix(win_data, self.sfreq, band=band)

        edges = analytics.get_top_connectivity_edges(conn_mat, self.channel_names, threshold=0.62, top_k=24)
        return conn_mat, edges

    def decode_motor_intent(self, current_sample: int, window_sec: float = 2.0) -> Tuple[Dict[str, float], str, float]:
        win_data = self.get_sliding_window_64(current_sample, window_sec)
        return self.bci_pipeline.decode_active_window(win_data)


# ==============================================================================
# LIVE STREAM INGESTION (LABSTREAMINGLAYER / PYLSL INLET THREAD)
# ==============================================================================

class LSLReceiverThread(QThread):
    """
    LabStreamingLayer (pylsl) Stream Receiver:
    - Resolves real-time EEG streams over the local network (timeout=1.5s).
    - Ingests streaming chunks into EEGDataLoader.ingest_live_chunk().
    - High-fidelity synthetic fallback simulator for standalone testing without hardware.
    """
    chunk_received = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(str, str)  # (text, color)
    metrics_updated = pyqtSignal(float, float)  # (effective_fs, latency_ms)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.is_running = False
        self.is_simulated = False
        self.inlet = None
        self.stream_name = "EEG"

    def connect_stream(self, use_simulation: bool = False):
        self.is_simulated = use_simulation
        self.is_running = True
        self.start()

    def disconnect_stream(self):
        self.is_running = False
        self.wait(1000)
        self.status_changed.emit("LSL: Disconnected", "#8C9BAE")

    def run(self):
        if not self.is_simulated and LSL_AVAILABLE:
            self.status_changed.emit("LSL: Resolving Streams...", "#FFB300")
            try:
                streams = pylsl.resolve_byprop('type', 'EEG', timeout=2.0)
                if len(streams) > 0:
                    info = streams[0]
                    self.inlet = pylsl.StreamInlet(info, max_buflen=360, max_chunklen=32)
                    self.stream_name = info.name()
                    self.status_changed.emit(f"LSL: Connected ({self.stream_name})", "#00FFA3")
                else:
                    self.status_changed.emit("LSL: No Stream Found (Running Simulator)", "#00E5FF")
                    self.is_simulated = True
            except Exception as e:
                print(f"[WARN] LSL resolve exception: {e}. Falling back to simulation.")
                self.is_simulated = True

        if self.is_simulated or not LSL_AVAILABLE:
            self.status_changed.emit("LSL Simulator: Active (160 Hz)", "#00FFA3")

        t_last = time.perf_counter()
        samples_count = 0

        while self.is_running:
            t0 = time.perf_counter()
            if self.inlet is not None and not self.is_simulated:
                try:
                    samples, timestamps = self.inlet.pull_chunk(timeout=0.04, max_samples=32)
                    if samples:
                        arr = np.array(samples, dtype=np.float32).T
                        if arr.shape[0] < 64:
                            pad = np.zeros((64 - arr.shape[0], arr.shape[1]), dtype=np.float32)
                            arr = np.vstack([arr, pad])
                        chunk_64 = arr[:64, :]
                        self.data_loader.ingest_live_chunk(chunk_64)
                        self.chunk_received.emit(chunk_64)
                        samples_count += chunk_64.shape[1]
                except Exception as e:
                    self.status_changed.emit(f"LSL Error: {e}", "#FF5252")
                    break
            else:
                time.sleep(0.04)
                k_samples = 6
                chunk_sim = np.random.randn(64, k_samples).astype(np.float32) * 4.0
                t_arr = np.linspace(0, 0.04, k_samples)
                chunk_sim[60:64, :] += 18.0 * np.sin(2 * np.pi * 10.0 * t_arr)
                self.data_loader.ingest_live_chunk(chunk_sim)
                self.chunk_received.emit(chunk_sim)
                samples_count += k_samples

            now = time.perf_counter()
            if now - t_last >= 1.0:
                eff_fs = samples_count / (now - t_last)
                latency = (time.perf_counter() - t0) * 1000.0
                self.metrics_updated.emit(eff_fs, latency)
                samples_count = 0
                t_last = now


# ==============================================================================
# PLAYBACK ENGINE (HIGH PRECISION TICKER WITH FRAME-BY-FRAME ANALYSIS)
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
        self.timer.setInterval(30)
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

    def step_frame_forward(self, delta_samples: int = 1):
        """Frame-by-Frame forward step (1 sample = 6.25ms at 160Hz)."""
        self.seek_sample(self.current_sample + delta_samples)

    def step_frame_backward(self, delta_samples: int = 1):
        """Frame-by-Frame backward step (1 sample = 6.25ms at 160Hz)."""
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
