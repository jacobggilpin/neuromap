# ==============================================================================
# SECTION 1: CLOSED-LOOP DATA ENGINE & STREAMING PLAYBACK
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
        for b_time in np.arange(3.0, self.duration - 2.0, 6.0):
            b_idx = int(b_time * self.sfreq)
            dur_pts = int(0.35 * self.sfreq)
            if b_idx + dur_pts < self.n_samples:
                b_wave = np.hanning(dur_pts) * 85.0
                blink_train[b_idx:b_idx + dur_pts] += b_wave

        self.raw_data[0, :] += blink_train * 1.0   # Fp1
        self.raw_data[1, :] += blink_train * 1.25  # Fpz
        self.raw_data[2, :] += blink_train * 1.0   # Fp2

        # Embed lateral saccade artifacts onto F7 / F8
        for s_time in np.arange(5.0, self.duration - 3.0, 8.5):
            s_idx = int(s_time * self.sfreq)
            dur_pts = int(0.25 * self.sfreq)
            if s_idx + dur_pts < self.n_samples:
                s_wave = np.sin(np.linspace(0, np.pi, dur_pts)) * 38.0
                self.raw_data[8, s_idx:s_idx + dur_pts] += s_wave   # F7
                self.raw_data[16, s_idx:s_idx + dur_pts] -= s_wave  # F8

        # Embed cardiac ECG QRS complexes
        ecg_indices = np.arange(int(0.6 * self.sfreq), self.n_samples - 40, int(0.833 * self.sfreq))
        for p in ecg_indices:
            qrs = np.array([-4.0, 28.0, 48.0, -10.0, -2.0], dtype=np.float32)
            self.raw_data[26, p-2:p+3] += qrs * 0.7  # T7
            self.raw_data[34, p-2:p+3] += qrs * 0.7  # T8
            self.raw_data[61, p-2:p+3] += qrs * 0.9  # Oz
            self.raw_data[63, p-2:p+3] += qrs * 1.0  # Iz

        # Embed transient high-variance bursts for ASR
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
        Decomposes the 64-channel EEG array into independent components on demand.
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

        # Compute Kurtosis and Variance Explained for each component
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
# SECTION 2: DEDICATED PRE-PROCESSING WORKSTATION (TOP-LEFT)
# ==============================================================================

class ICTraceThumbnailWidget(QWidget):
    """Clean mini-waveform thumbnail preview for an Independent Component."""
    def __init__(self, trace: np.ndarray, color: str = "#00FFA3", parent=None):
        super().__init__(parent)
        self.trace = trace
        self.color = QColor(color)
        self.setFixedHeight(24)
        self.setMinimumWidth(80)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        mid_y = h / 2.0

        # Background rail
        painter.fillRect(self.rect(), QColor("#090C12"))
        painter.setPen(QPen(QColor("#1A2230"), 1, Qt.DashLine))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        if self.trace is None or len(self.trace) < 2:
            return

        poly = QPolygonF()
        n_pts = len(self.trace)
        step = max(1, n_pts // w)
        sub_trace = self.trace[::step]
        n_sub = len(sub_trace)

        for i, val in enumerate(sub_trace):
            x = (i / max(1, n_sub - 1)) * w
            y = mid_y - (val * (h * 0.38))
            poly.append(QPointF(x, max(2.0, min(h - 2.0, y))))

        pen = QPen(self.color, 1.2)
        painter.setPen(pen)
        painter.drawPolyline(poly)


class AnalysisPanelWidget(QWidget):
    """
    Dedicated Pre-Processing & Artifact Reconstruction Workstation (Top-Left):
    - Tab 1: FastICA Spatial Decomposition (component explorer, badges, thumbnails, rejection checkboxes).
    - Tab 2: Artifact Subspace Reconstruction (ASR calibration, thresholding, subspace cleaning).
    - Standby state on launch; decomposes on-demand without freezing application startup.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self._init_ui()
        self.data_loader.data_reconstructed.connect(self._on_data_reconstructed)

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

        title_lbl = QLabel("<span style='color: #00E5FF;'>●</span> PRE-PROCESSING WORKSTATION")
        title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(title_lbl)

        h_layout.addStretch()

        self.reconstruct_status_lbl = QLabel("FastICA: Standby (Active Stream Filter)")
        self.reconstruct_status_lbl.setStyleSheet("color: #7A889B; font-size: 10px;")
        h_layout.addWidget(self.reconstruct_status_lbl)

        layout.addWidget(header)

        # Tabs for Pre-Processing Modes
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: #0E0F14;
            }
            QTabBar::tab {
                background: #12151D;
                color: #8C9BAE;
                padding: 5px 12px;
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
        self.tab_widget.addTab(self.tab_ica, "FastICA Decomposition")

        # Tab 2: Artifact Subspace Reconstruction (ASR)
        self.tab_asr = QWidget()
        self._init_asr_tab()
        self.tab_widget.addTab(self.tab_asr, "Artifact Subspace (ASR)")

        layout.addWidget(self.tab_widget, stretch=1)

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

    def _init_asr_tab(self):
        tab_layout = QVBoxLayout(self.tab_asr)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(12)

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

    def rebuild_component_cards(self):
        """Builds interactive cards for all decomposed FastICA components."""
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
        self.data_loader.ica_enabled = False
        self.data_loader._reconstruct_signal()
        self.rebuild_component_cards()
        self.reconstruct_status_lbl.setText("FastICA: All Components Restored (Raw Filtered)")

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
# ==============================================================================
# SECTION 3: WAVEFORMS & SPOTIFY PLAYBACK BAR (FRAME-BY-FRAME STEPPING)
# ==============================================================================

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

        # Prominent Thicker Playhead Scrubber Line (Z=5, width=3.5)
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
        super().mouseReleaseEvent(ev)

    def _handle_scrub(self, mouse_x: float):
        w = max(1.0, float(self.width()))
        frac = max(0.0, min(1.0, mouse_x / w))
        t_target = frac * self.data_loader.duration
        self.playhead_line.setValue(t_target)
        self.seek_requested.emit(t_target)


class SpotifyPlaybackBar(QFrame):
    """
    Spotify-Style Transport Bar:
    - Frame-by-frame backward (⏮) and forward (⏭) stepping.
    - Speeds strictly limited to [0.1x, 0.25x, 0.5x, 1.0x].
    - No mouse focus highlight borders on buttons.
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

        # Frame-by-Frame Backward (1 sample / 6.25ms) with NoFocus
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
        self.btn_play.setFixedSize(36, 36)
        self.btn_play.setToolTip("Play / Pause (Space)")
        self.btn_play.setFocusPolicy(Qt.NoFocus)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                color: #000000;
                font-size: 15px;
                font-weight: bold;
                border-radius: 18px;
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

        # Frame-by-Frame Forward (1 sample / 6.25ms) with NoFocus
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

        mins = int(self.data_loader.duration // 60)
        secs = self.data_loader.duration % 60
        self.time_total_lbl = QLabel(f"{mins}:{secs:06.3f}")
        self.time_total_lbl.setFixedWidth(52)
        self.time_total_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.time_total_lbl.setStyleSheet("color: #8C9BAE; font-family: 'Consolas', monospace; font-size: 10px;")

        time_row.addWidget(self.time_cur_lbl)
        time_row.addWidget(self.timeline, stretch=1)
        time_row.addWidget(self.time_total_lbl)
        center_layout.addLayout(time_row)

        layout.addLayout(center_layout, stretch=1)
        layout.addSpacing(16)

        # Right: Stimulus Legend & Current State Badge
        right_layout = QVBoxLayout()
        right_layout.setSpacing(4)
        right_layout.setAlignment(Qt.AlignVCenter)

        legend_box = QHBoxLayout()
        legend_box.setSpacing(10)
        for eid in ['T0', 'T1', 'T2']:
            cfg = EVENT_COLOR_MAP[eid]
            dot = QLabel(f"<span style='color: {cfg['color']}; font-size: 14px;'>●</span> {cfg['name']}")
            dot.setStyleSheet("color: #8C9BAE; font-size: 10px; font-weight: bold;")
            legend_box.addWidget(dot)
        right_layout.addLayout(legend_box)

        self.stimulus_badge = QLabel("Active Stimulus: Rest")
        self.stimulus_badge.setAlignment(Qt.AlignCenter)
        self.stimulus_badge.setStyleSheet("""
            background: #14171E;
            color: #8C9BAE;
            border: 1px solid #232B3B;
            border-radius: 3px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: bold;
        """)
        right_layout.addWidget(self.stimulus_badge)

        layout.addLayout(right_layout, stretch=0)

        # Connect engine signals
        self.engine.state_changed.connect(self._on_play_state_changed)

    def _on_play_state_changed(self, is_playing: bool):
        self.btn_play.setText("⏸" if is_playing else "▶")
        self.btn_play.setStyleSheet("""
            QPushButton {
                background: #1DB954;
                color: #000000;
                font-size: 15px;
                font-weight: bold;
                border-radius: 18px;
                padding-left: 1px;
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background: #1ED760;
            }
            QPushButton:focus {
                outline: none;
                border: none;
            }
        """ if is_playing else """
            QPushButton {
                background: #FFFFFF;
                color: #000000;
                font-size: 15px;
                font-weight: bold;
                border-radius: 18px;
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

    def update_frame(self, current_sample: int, current_time: float):
        mins = int(current_time // 60)
        secs = current_time % 60
        self.time_cur_lbl.setText(f"{mins}:{secs:06.3f}")
        self.timeline.update_playhead(current_time)

        # Update active stimulus event readout
        active_ev = None
        for ev in self.data_loader.events:
            if ev.start_time <= current_time <= ev.end_time:
                active_ev = ev
                break

        if active_ev is not None:
            cfg = EVENT_COLOR_MAP.get(active_ev.event_id, {'color': '#00FFA3', 'bg': '#14171E', 'border': '#232B3B'})
            self.stimulus_badge.setText(f"Active Stimulus: {active_ev.label}")
            self.stimulus_badge.setStyleSheet(f"""
                background: {cfg['bg']};
                color: {cfg['color']};
                border: 1px solid {cfg['border']};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            """)
        else:
            self.stimulus_badge.setText("Active Stimulus: Rest")
            self.stimulus_badge.setStyleSheet("""
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            """)
# ==============================================================================
# SECTION 3: WAVEFORMS & SPOTIFY PLAYBACK BAR (FRAME-BY-FRAME STEPPING)
# ==============================================================================

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

        # Prominent Thicker Playhead Scrubber Line (Z=5, width=3.5)
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
        super().mouseReleaseEvent(ev)

    def _handle_scrub(self, mouse_x: float):
        w = max(1.0, float(self.width()))
        frac = max(0.0, min(1.0, mouse_x / w))
        t_target = frac * self.data_loader.duration
        self.playhead_line.setValue(t_target)
        self.seek_requested.emit(t_target)


class SpotifyPlaybackBar(QFrame):
    """
    Spotify-Style Transport Bar:
    - Frame-by-frame backward (⏮) and forward (⏭) stepping.
    - Speeds strictly limited to [0.1x, 0.25x, 0.5x, 1.0x].
    - No mouse focus highlight borders on buttons.
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

        # Frame-by-Frame Backward (1 sample / 6.25ms) with NoFocus
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
        self.btn_play.setFixedSize(36, 36)
        self.btn_play.setToolTip("Play / Pause (Space)")
        self.btn_play.setFocusPolicy(Qt.NoFocus)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background: #FFFFFF;
                color: #000000;
                font-size: 15px;
                font-weight: bold;
                border-radius: 18px;
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

        # Frame-by-Frame Forward (1 sample / 6.25ms) with NoFocus
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

        mins = int(self.data_loader.duration // 60)
        secs = self.data_loader.duration % 60
        self.time_total_lbl = QLabel(f"{mins}:{secs:06.3f}")
        self.time_total_lbl.setFixedWidth(52)
        self.time_total_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.time_total_lbl.setStyleSheet("color: #8C9BAE; font-family: 'Consolas', monospace; font-size: 10px;")

        time_row.addWidget(self.time_cur_lbl)
        time_row.addWidget(self.timeline, stretch=1)
        time_row.addWidget(self.time_total_lbl)
        center_layout.addLayout(time_row)

        layout.addLayout(center_layout, stretch=1)
        layout.addSpacing(16)

        # Right: Stimulus Legend & Current State Badge
        right_layout = QVBoxLayout()
        right_layout.setSpacing(4)
        right_layout.setAlignment(Qt.AlignVCenter)

        legend_box = QHBoxLayout()
        legend_box.setSpacing(10)
        for eid in ['T0', 'T1', 'T2']:
            cfg = EVENT_COLOR_MAP[eid]
            dot = QLabel(f"<span style='color: {cfg['color']}; font-size: 14px;'>●</span> {cfg['name']}")
            dot.setStyleSheet("color: #8C9BAE; font-size: 10px; font-weight: bold;")
            legend_box.addWidget(dot)
        right_layout.addLayout(legend_box)

        self.stimulus_badge = QLabel("Active Stimulus: Rest")
        self.stimulus_badge.setAlignment(Qt.AlignCenter)
        self.stimulus_badge.setStyleSheet("""
            background: #14171E;
            color: #8C9BAE;
            border: 1px solid #232B3B;
            border-radius: 3px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: bold;
        """)
        right_layout.addWidget(self.stimulus_badge)

        layout.addLayout(right_layout, stretch=0)

        # Connect engine signals
        self.engine.state_changed.connect(self._on_play_state_changed)

    def _on_play_state_changed(self, is_playing: bool):
        self.btn_play.setText("⏸" if is_playing else "▶")
        self.btn_play.setStyleSheet("""
            QPushButton {
                background: #1DB954;
                color: #000000;
                font-size: 15px;
                font-weight: bold;
                border-radius: 18px;
                padding-left: 1px;
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background: #1ED760;
            }
            QPushButton:focus {
                outline: none;
                border: none;
            }
        """ if is_playing else """
            QPushButton {
                background: #FFFFFF;
                color: #000000;
                font-size: 15px;
                font-weight: bold;
                border-radius: 18px;
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

    def update_frame(self, current_sample: int, current_time: float):
        mins = int(current_time // 60)
        secs = current_time % 60
        self.time_cur_lbl.setText(f"{mins}:{secs:06.3f}")
        self.timeline.update_playhead(current_time)

        # Update active stimulus event readout
        active_ev = None
        for ev in self.data_loader.events:
            if ev.start_time <= current_time <= ev.end_time:
                active_ev = ev
                break

        if active_ev is not None:
            cfg = EVENT_COLOR_MAP.get(active_ev.event_id, {'color': '#00FFA3', 'bg': '#14171E', 'border': '#232B3B'})
            self.stimulus_badge.setText(f"Active Stimulus: {active_ev.label}")
            self.stimulus_badge.setStyleSheet(f"""
                background: {cfg['bg']};
                color: {cfg['color']};
                border: 1px solid {cfg['border']};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            """)
        else:
            self.stimulus_badge.setText("Active Stimulus: Rest")
            self.stimulus_badge.setStyleSheet("""
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            """)