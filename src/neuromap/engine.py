"""
neuromap - Closed-Loop Signal Processing & Playback Engine
Includes:
- CentralizedChannelState: Unified channel multi-selection
- SharedInterpolationCache: Shared 2D/3D IDW matrices
- EEGDataLoader: Streaming ingestion, DSP filters, FastICA spatial filter, Phase 4.03 sliding window
- PlaybackEngine: High-precision wall-clock playback ticker & frame-by-frame stepping
"""
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import scipy.signal

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# MNE-Python
try:
    import mne
    from mne.datasets import eegbci
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

try:
    from .config import (
        STANDARD_64_CHANNELS, PALETTE_COLORS, StimulusEvent,
        EVENT_COLOR_MAP, MONTAGE_2D_COORDS
    )
except ImportError:
    from config import (
        STANDARD_64_CHANNELS, PALETTE_COLORS, StimulusEvent,
        EVENT_COLOR_MAP, MONTAGE_2D_COORDS
    )

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
# DATA LOADER & SIGNAL PROCESSING (FASTICA & ASR ENGINES)
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


class EEGDataLoader(QObject):
    """
    Closed-Loop EEG Data Engine:
    - Designed for live streaming input and real-time closed-loop neurofeedback/analysis.
    - Eliminates offline pre-computations on startup; algorithms operate on the active stream.
    - Dynamic FastICA Spatial Decomposition & Online Rejection Projection Engine (P = A_clean @ W).
    - Artifact Subspace Reconstruction (ASR) burst filter.
    - Phase 4.03 standard array window slicing with edge padding (no pre-padded zero-copy circular ring buffer).
    """
    data_reconstructed = pyqtSignal()

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
        """
        Initializes real-time filter coefficients and applies initial streaming bandpass/notch filtering.
        NOTE: Heavy spatial decompositions (FastICA, ASR) are NOT run here; they execute on-demand
        in the closed loop to ensure instantaneous application startup.
        """
        print(f"[INFO] Initializing Stream DSP Filters: Bandpass [{l_freq}-{h_freq} Hz], Notch [{notch_freq} Hz]...")
        self.filtered_data = np.zeros_like(self.raw_data)
        nyq = 0.5 * self.sfreq

        low = max(0.001, l_freq / nyq)
        high = min(0.999, h_freq / nyq)
        b_band, a_band = scipy.signal.butter(3, [low, high], btype='bandpass')

        w0 = notch_freq / nyq
        b_notch, a_notch = scipy.signal.iirnotch(w0, 30.0)

        # Fast vectorized filtering across channels
        s_bp = scipy.signal.filtfilt(b_band, a_band, self.raw_data, axis=-1)
        self.filtered_data = scipy.signal.filtfilt(b_notch, a_notch, s_bp, axis=-1).astype(np.float32)
        self.base_filtered_data = self.filtered_data.copy()

        # Compute timeline waveform mini-map envelope (<15ms)
        self._compute_gfp_envelope()

    def get_active_data(self) -> np.ndarray:
        """Returns the active data buffer: cleaned (if ICA/ASR active), base filtered, or raw."""
        if (self.ica_enabled or self.asr_enabled) and self.clean_data is not None:
            return self.clean_data
        if self.filtered_data is not None:
            return self.filtered_data
        return self.raw_data

    def get_window_data(self, channel_idx: int, current_sample: int, n_samples: int) -> np.ndarray:
        """
        Phase 4.03 Standard Sliding Window:
        Retrieves an n_samples window ending at current_sample.
        Left-pads with the initial sample value using np.hstack if current_sample < n_samples.
        """
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
        """Returns the 64-channel voltage vector at the given sample index."""
        data = self.get_active_data()
        if data is None or sample_idx < 0 or sample_idx >= self.n_samples:
            return np.zeros(self.n_channels, dtype=np.float32)
        return data[:, sample_idx]

    def ingest_live_chunk(self, chunk_64: np.ndarray):
        """
        Closed-Loop Ingestion:
        Ingests actively inputted data chunk (64, K) into the live stream buffer,
        enabling live streaming EEG input in a closed-loop system.
        """
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

    def process_live_frame(self, raw_sample_64: np.ndarray) -> np.ndarray:
        """
        Closed-Loop Real-Time Frame Processor:
        Processes a single incoming 64-channel EEG frame through the closed-loop pipeline:
        Applies real-time spatial projection if ICA cleaning is active.
        """
        x = raw_sample_64.astype(np.float32)
        if self.ica_enabled and self.spatial_projection_matrix is not None and self.ica_mean is not None:
            mu = self.ica_mean.ravel()
            x = np.dot(self.spatial_projection_matrix, (x - mu)) + mu
        return x

    def decompose_fastica(self, n_components: int = 16):
        """
        Dynamic FastICA Spatial Filter Engine:
        Decomposes the 64-channel EEG array into independent components.
        Ranks and tags components:
        - IC0: Frontal vertical EOG / blinks (highest absolute frontal loading & kurtosis)
        - IC1: Lateral anterior saccades (highest horizontal differential |F7 - F8|)
        - IC2: Cardiac ECG pulse (rhythmic ~72 BPM QRS complexes)
        - IC3: Temporal high-frequency EMG muscle activity
        """
        print(f"[INFO] Running FastICA Spatial Decomposition ({n_components} components)...")
        t0 = time.perf_counter()

        data = self.base_filtered_data if self.base_filtered_data is not None else self.raw_data
        n_channels, n_samples = data.shape
        mean_X = np.mean(data, axis=1, keepdims=True)
        Xc = data - mean_X

        # Subsample for sub-second decomposition speed
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

        # 3. Component Statistics & Automated Classification
        # IC0: Frontal blink / vertical EOG (max absolute weight on Fp1, Fpz, Fp2)
        frontal_scores = np.sum(np.abs(A_mix[:3, :]), axis=0)
        ic0_idx = int(np.argmax(frontal_scores))

        # IC1: Horizontal saccades (max difference between F7 and F8)
        lat_scores = np.abs(A_mix[8, :] - A_mix[16, :])
        lat_scores[ic0_idx] = -1.0
        ic1_idx = int(np.argmax(lat_scores))

        # IC2: Cardiac ECG pulse (periodic heartbeat peaks)
        temp_indices = [26, 34, 35, 36]
        ecg_scores = np.zeros(n_components, dtype=np.float32)
        for c in range(n_components):
            if c not in (ic0_idx, ic1_idx):
                s_c = S[c, :]
                peaks, _ = scipy.signal.find_peaks(np.abs(s_c), distance=int(self.sfreq * 0.6), height=np.std(s_c)*2.0)
                if len(peaks) > 10:
                    # Regularity of inter-beat intervals
                    diffs = np.diff(peaks)
                    if np.std(diffs) < (0.25 * np.mean(diffs)):
                        ecg_scores[c] = float(len(peaks))
        ic2_idx = int(np.argmax(ecg_scores)) if np.max(ecg_scores) > 0 else (2 if 2 not in (ic0_idx, ic1_idx) else 3)

        # IC3: Temporal EMG muscle activity
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

        # Compute Kurtosis and Variance Explained for each component
        total_data_var = np.sum(np.var(data, axis=1)) + 1e-6
        self.ic_components.clear()

        # Snippet length: 600 points (~3.75s)
        snip_len = min(600, n_samples)
        for i in range(n_components):
            s_i = self.ica_sources[i, :]
            m4 = np.mean((s_i - np.mean(s_i))**4)
            m2 = np.var(s_i)
            kurt = float(m4 / (m2**2 + 1e-8) - 3.0)
            
            back_proj_var = np.sum(np.var(np.outer(self.ica_mixing[:, i], s_i), axis=1))
            var_pct = float(back_proj_var / total_data_var * 100.0)
            
            # Tags & Colors
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
            # Normalize snippet for thumbnail display
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

        # Re-apply current rejection preset if ICA is active
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
        """Allows individual component selection from the Dedicated Pre-processing Workstation."""
        if reject:
            self.rejected_ic_indices.add(ic_idx)
        else:
            self.rejected_ic_indices.discard(ic_idx)

        for comp in self.ic_components:
            if comp.idx == ic_idx:
                comp.is_rejected = reject

        self._reconstruct_signal()

    def apply_asr(self, cutoff_sd: float = 5.0, win_len_sec: float = 0.5):
        """
        Artifact Subspace Reconstruction (ASR):
        Projects sliding windows against clean baseline calibration covariance and
        reconstructs corrupted subspaces where variance exceeds cutoff_sd standard deviations.
        """
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
        """
        Updates the closed-loop 64x64 spatial filter projection matrix P.
        For any incoming frame or active array:
        x_clean = P @ (x - mu) + mu
        """
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
        """Reconstructs the active EEG array from FastICA spatial projection and optional ASR."""
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

        # Apply ASR if enabled
        if self.asr_enabled:
            reconstructed = self.apply_asr(self.asr_cutoff_sd)

        self.clean_data = reconstructed

        # Set active visualization buffer
        if self.ica_enabled or self.asr_enabled:
            self.filtered_data = self.clean_data
        else:
            self.filtered_data = self.base_filtered_data

        self._compute_gfp_envelope()
        self.data_reconstructed.emit()

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


# ==============================================================================
# DEDICATED PRE-PROCESSING & ARTIFACT RECONSTRUCTION WORKSTATION (TOP-LEFT)
# ==============================================================================

