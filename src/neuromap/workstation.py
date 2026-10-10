"""
neuromap - Analysis Workstation Panel (Phase 4.03)
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Components:
- AnalysisPanelWidget: Clean, modern workstation placeholder panel (Top-Left section).
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame

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


