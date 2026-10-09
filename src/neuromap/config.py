"""
neuromap - Configuration & Constants
Defines standard 64-channel 10-05 montage, coordinate maps, stimulus events, and colormaps.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


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

