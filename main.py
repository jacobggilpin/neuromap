"""
neuromap - Application Entry Point & Main Dashboard
Integrates Pre-Processing Workstation, Dual Topomap Container, Cascading Waveforms,
Spotify Playback Bar, and Control Sidebar into a unified 1920x1200 dark theme layout.
"""
import sys
import csv
from typing import Set

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QFileDialog, QShortcut
)
import pyqtgraph as pg

# OpenGL safety on Windows
pg.setConfigOptions(useOpenGL=False, antialias=True)

try:
    from .engine import EEGDataLoader, PlaybackEngine
    from .workstation import AnalysisPanelWidget
    from .viewports import TopomapContainerWidget
    from .ui_components import (
        WaveformsWidget, SpotifyPlaybackBar, HeaderWidget, ControlSidebarWidget
    )
except ImportError:
    from engine import EEGDataLoader, PlaybackEngine
    from workstation import AnalysisPanelWidget
    from viewports import TopomapContainerWidget
    from ui_components import (
        WaveformsWidget, SpotifyPlaybackBar, HeaderWidget, ControlSidebarWidget
    )

class NeuromapMainWindow(QMainWindow):
    """
    Neuromap Main Dashboard Window:
    - Layout:
      * Top-Left: Dedicated Pre-Processing Workstation (FastICA + ASR)
      * Top-Right: Topomap Viewport (Toggle between 2D Topomap and 3D Brain)
    - Middle/Bottom: Cascading Raw Waveforms (Phase 4.03 Sliding Window)
    - Bottom Dock: Spotify Playback Bar with Frame-by-Frame Stepping & Mini-Map
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

        # Top Row (Horizontal Splitter: Pre-Processing Workstation on LEFT, Topomap Viewport on RIGHT)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setHandleWidth(4)

        # Top-Left: Dedicated Pre-Processing & Artifact Reconstruction Workstation
        self.analysis_panel = AnalysisPanelWidget(self.data_loader, self)
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

        # Phase 4.04 Shortcuts: Frame-by-Frame navigation with Left/Right arrows
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

    def _on_toggle_ica(self, checked: bool):
        self.data_loader.ica_enabled = checked
        if checked and len(self.data_loader.ic_components) == 0:
            self.status_label.setText("Status: Computing FastICA Spatial Filter on active stream...")
            QApplication.processEvents()
            self.data_loader.decompose_fastica()
            self.analysis_panel.rebuild_component_cards()
            self.status_label.setText("Status: FastICA Spatial Filter Ready")
        self.data_loader.apply_ica_rejection(self.sidebar.combo_ica_preset.currentText())
        self.sidebar.ica_status_lbl.setText(f"ICA: {self.data_loader.ica_preset.split('(')[-1].strip(')')} rejected" if checked else "ICA: Disabled (Raw)")

    def _on_ica_preset_changed(self, preset: str):
        if self.data_loader.ica_enabled and len(self.data_loader.ic_components) == 0:
            self.data_loader.decompose_fastica()
            self.analysis_panel.rebuild_component_cards()
        self.data_loader.apply_ica_rejection(preset)
        self.sidebar.ica_status_lbl.setText(f"ICA: {preset.split('(')[-1].strip(')')} rejected")

    def _on_recompute_ica(self):
        self.status_label.setText("Status: Re-computing FastICA Spatial Filter...")
        QApplication.processEvents()
        self.data_loader.decompose_fastica()
        self.analysis_panel.rebuild_component_cards()
        self.sidebar.ica_status_lbl.setText(f"ICA: Re-computed ({self.data_loader.ica_preset})")
        self.status_label.setText("Status: FastICA Decomposition Completed Successfully")

    def _on_live_reconstruction_update(self):
        """Immediately refreshes waveforms, topomaps, and timeline mini-map when FastICA/ASR updates."""
        self.playback_bar.timeline.refresh_minimap()
        cur_samp = self.engine.current_sample
        cur_t = cur_samp / self.data_loader.sfreq
        self._on_frame_update(cur_samp, cur_t)

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
