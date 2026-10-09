"""
neuromap - Professional Closed-Loop 64-Channel EEG Visualization & Analytical Workstation
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Phase 5 Architecture:
- Top-Left: Dedicated Pre-Processing & Advanced Analytical Workstation:
  * Tab 1: FastICA Spatial Decomposition & Online Rejection
  * Tab 2: Artifact Subspace Reconstruction (ASR)
  * Tab 3: Power Spectral Density (PSD) & Canonical Band Power (Delta, Theta, Alpha, Beta, Gamma)
  * Tab 4: Time-Frequency ERSP & Sensorimotor ERD (C3, Cz, C4 synced to T1/T2 motor imagery)
  * Tab 5: Functional Connectivity Graph Networks (PLV & Spectral Coherence)
  * Tab 6: Real-Time BCI Motor Imagery Decoder (CSP + Online LDA, <50µs inference)
  * Live Stream Ingestion: LabStreamingLayer (pylsl) inlet plugged directly into EEGDataLoader.ingest_live_chunk
- Top-Right: Dual Topomap Container with Mode Toggle [2D Topomap] | [3D Brain]:
  * Dynamic Neural Connectivity Arcs (PLV & Coherence) rendered in both 2D and 3D cortical space!
  * 3D Cold-Standby Optimization (zero overhead when in 2D mode or collapsed).
- Middle/Bottom: Cascading Raw Waveforms with Phase 4.03 standard sliding window slicing.
- Bottom Dock: Spotify Playback Bar with Frame-by-Frame Stepping & GFP Mini-Map (speeds [0.1x, 0.25x, 0.5x, 1.0x]).
"""

import sys
import os
import csv
from typing import List, Set

import numpy as np

# PyQt5 GUI Framework
try:
    from PyQt5.QtCore import Qt, QKeySequence
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QLabel, QFileDialog, QShortcut, QStatusBar, QProgressBar
    )
    import pyqtgraph as pg

    # CRITICAL ENGINE SAFETY: Keep useOpenGL=False when running PyVistaQt on Windows
    # to prevent fatal OpenGL context collisions (0xC0000005) between QOpenGLWidget and VTK.
    pg.setConfigOptions(useOpenGL=False, antialias=True)
except ImportError:
    pass

# ==============================================================================
# AUTOMATIC PATH BOOTSTRAP (Supports root execution, src/neuromap layout & VS Code)
# ==============================================================================
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
    from neuromap.config import STANDARD_64_CHANNELS
    from neuromap.engine import EEGDataLoader, PlaybackEngine
    from neuromap.workstation import AnalysisPanelWidget
    from neuromap.viewports import TopomapContainerWidget
    from neuromap.ui_components import (
        WaveformsWidget, SpotifyPlaybackBar, HeaderWidget, ControlSidebarWidget
    )
except (ImportError, ModuleNotFoundError):
    from config import STANDARD_64_CHANNELS
    from engine import EEGDataLoader, PlaybackEngine
    from workstation import AnalysisPanelWidget
    from viewports import TopomapContainerWidget
    from ui_components import (
        WaveformsWidget, SpotifyPlaybackBar, HeaderWidget, ControlSidebarWidget
    )



class NeuromapMainWindow(QMainWindow):
    """
    Neuromap Main Dashboard Window (Phase 5):
    - Layout:
      * Top-Left: Dedicated Analytical Workstation (FastICA, ASR, PSD, ERSP, Connectivity, BCI Decoder, LSL)
      * Top-Right: Topomap Viewport (Toggle between 2D Topomap and 3D Brain with Dynamic Neural Connectivity Arcs)
    - Middle/Bottom: Cascading Raw Waveforms (Phase 4.03 Sliding Window)
    - Bottom Dock: Spotify Playback Bar with Frame-by-Frame Stepping & GFP Mini-Map
    - Zero-Overhead Dissolved Section Culling & Cold-Standby 3D VTK Engine
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("neuromap - Professional Closed-Loop EEG Workstation (Phase 5)")
        self.resize(1920, 1200)

        self.data_loader = EEGDataLoader()
        self.data_loader.load_dataset()
        self.engine = PlaybackEngine(self.data_loader)

        self._last_conn_sync = 0.0

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

        # Top Row (Horizontal Splitter: Analytical Workstation on LEFT, Topomap on RIGHT)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setHandleWidth(4)

        # Top-Left: Dedicated Advanced Analytical Workstation
        self.analysis_panel = AnalysisPanelWidget(self.data_loader, self)
        self.top_splitter.addWidget(self.analysis_panel)

        # Top-Right: Toggleable Topomap Container (2D Map / 3D Brain + Dynamic Neural Connectivity Arcs)
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

        # Bottom Dock: Spotify Playback Bar with Frame-by-Frame Stepping
        self.playback_bar = SpotifyPlaybackBar(self.data_loader, self.engine, self)
        root_layout.addWidget(self.playback_bar)

        # Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background: #060709; color: #7A889B; border-top: 1px solid #14171E;")
        self.status_label = QLabel("Status: System Ready | Closed-Loop Active | 60+ FPS Performance Baseline")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Phase 5: Online (100%)")
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

        # Keyboard Shortcuts
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.engine.toggle_play)
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_F4), self, activated=self.sidebar.toggle_collapse)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.engine.step_frame_backward(1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.engine.step_frame_forward(1))

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

        # FastICA Artifact Removal Listeners
        self.sidebar.chk_enable_ica.toggled.connect(self._on_toggle_ica)
        self.sidebar.combo_ica_preset.currentTextChanged.connect(self._on_ica_preset_changed)
        self.sidebar.btn_recompute_ica.clicked.connect(self._on_recompute_ica)

        # Data reconstruction signal
        self.data_loader.data_reconstructed.connect(self._on_live_reconstruction_update)

        # Splitter motion listeners for dissolved section culling
        self.main_splitter.splitterMoved.connect(self._on_splitter_adjusted)
        self.top_splitter.splitterMoved.connect(self._on_splitter_adjusted)

    def _is_widget_active(self, widget: QWidget) -> bool:
        if not widget.isVisible():
            return False
        return widget.width() >= 25 and widget.height() >= 25

    def _on_splitter_adjusted(self, pos: int, index: int):
        topomap_active = self._is_widget_active(self.topomap_container)
        if not topomap_active:
            if self.topomap_container.widget_3d is not None:
                self.topomap_container.widget_3d.pause_engine()
        else:
            if self.topomap_container.current_mode == "3D" and self.topomap_container.widget_3d is not None:
                self.topomap_container.widget_3d.resume_engine()

    def _on_frame_update(self, current_sample: int, current_time: float):
        # 1. Update 2D / 3D Topomap Viewport
        if self._is_widget_active(self.topomap_container):
            if self.topomap_container.current_mode == "2D":
                self.topomap_container.widget_2d.update_voltage_frame(current_sample)
            elif self.topomap_container.current_mode == "3D" and self.topomap_container.widget_3d is not None:
                voltages = self.data_loader.get_channel_voltages(current_sample)
                self.topomap_container.widget_3d.update_voltage_frame(voltages, self.topomap_container.widget_2d.v_scale)

        # 2. Update Dynamic Neural Connectivity Arcs (Throttled at ~5 Hz)
        if self.topomap_container.chk_conn_arcs.isChecked():
            if current_time - self._last_conn_sync >= 0.20:
                metric = "PLV" if "PLV" in self.analysis_panel.combo_conn_metric.currentText() else "Coherence"
                band = self.analysis_panel.combo_conn_band.currentText()
                _, edges = self.data_loader.get_connectivity_matrix(current_sample, metric, band)
                self.topomap_container.update_connectivity_arcs(edges)
                self._last_conn_sync = current_time

        # 3. Update Cascading Scrolling Waveforms
        if self._is_widget_active(self.waveforms_widget):
            self.waveforms_widget.update_frame(current_sample)

        # 4. Update Spotify Playback Bar & Scrubber
        self.playback_bar.update_frame(current_sample, current_time)

        # 5. Update Advanced Analytical Workstation Modules
        active_chs = list(self.data_loader.channel_state.selected_channels)
        self.analysis_panel.update_analytics_frame(current_sample, current_time, active_chs)

        # 6. Update Sidebar Quick BCI Intent Readout
        probs, label, conf = self.data_loader.decode_motor_intent(current_sample)
        conf_pct = int(round(conf * 100.0))
        self.sidebar.lbl_bci_quick_status.setText(f"{label} ({conf_pct}%)")
        if label == "Left Fist":
            self.sidebar.lbl_bci_quick_status.setStyleSheet("color: #00FFA3; font-size: 11px; font-weight: bold;")
        elif label == "Right Fist":
            self.sidebar.lbl_bci_quick_status.setStyleSheet("color: #BA68C8; font-size: 11px; font-weight: bold;")
        else:
            self.sidebar.lbl_bci_quick_status.setStyleSheet("color: #00E5FF; font-size: 11px; font-weight: bold;")

    def _on_central_channel_selection_changed(self, selected_set: Set[str]):
        sel_list = list(selected_set)
        self.waveforms_widget.set_active_channels(sel_list)
        self.topomap_container.update_active_badge(len(selected_set))
        self.topomap_container.widget_2d._refresh_node_styles()

        if self.topomap_container.widget_3d is not None:
            self.topomap_container.widget_3d.set_selected_channels(selected_set)

    def _on_toggle_ica(self, checked: bool):
        self.data_loader.ica_enabled = checked
        if checked and len(self.data_loader.ic_components) == 0:
            self.data_loader.decompose_fastica()
            self.analysis_panel.rebuild_component_cards()
        self.data_loader._reconstruct_signal()

    def _on_ica_preset_changed(self, preset: str):
        if len(self.data_loader.ic_components) == 0:
            self.data_loader.decompose_fastica()
            self.analysis_panel.rebuild_component_cards()
        self.data_loader.apply_ica_rejection(preset)
        self.analysis_panel.rebuild_component_cards()

    def _on_recompute_ica(self):
        self.data_loader.decompose_fastica()
        self.analysis_panel.rebuild_component_cards()

    def _on_live_reconstruction_update(self):
        curr_sample = self.engine.current_sample
        t_sec = curr_sample / self.data_loader.sfreq
        self._on_frame_update(curr_sample, t_sec)

    def _on_gain_changed(self, val: int):
        self.waveforms_widget.gain = val / 20.0
        self.waveforms_widget.update_frame(self.engine.current_sample)

    def _on_window_changed(self, val: int):
        self.waveforms_widget.update_window(float(val))
        self.waveforms_widget.update_frame(self.engine.current_sample)

    def _on_topo_scale_changed(self, val: int):
        self.topomap_container.widget_2d.set_v_scale(float(val))
        if self.topomap_container.widget_3d is not None:
            self.topomap_container.widget_3d.set_clim(float(val))
        self.topomap_container.widget_2d.update_voltage_frame(self.engine.current_sample)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _on_export_data(self):
        data = self.data_loader.get_active_data()
        if data is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Filtered EEG CSV", "neuromap_filtered_eeg.csv", "CSV Files (*.csv)")
        if not path:
            return

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Time_s'] + self.data_loader.channel_names
                writer.writerow(header)

                step = max(1, int(self.data_loader.sfreq / 40.0))
                for s in range(0, self.data_loader.n_samples, step):
                    t = s / self.data_loader.sfreq
                    row = [f"{t:.4f}"] + [f"{v:.3f}" for v in data[:, s]]
                    writer.writerow(row)

            self.status_label.setText(f"Status: Exported CSV successfully to {path}")
        except Exception as e:
            self.status_label.setText(f"Status: CSV Export failed ({e})")


def main():
    app = QApplication(sys.argv)
    window = NeuromapMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
