"""
neuromap - Topographic Viewports Architecture (2D Topomap & 3D Cortical Brain)
Targeted for 1920x1200 Display | High-DPI | Dark Modern Theme

Components:
- TopomapColorbarWidget: Horizontal voltage colorbar (µV) with unified coolwarm palette.
- Topomap2DWidget: 128x128 regularized multiquadric interpolation, anti-aliased smoothstep boundary mask,
  Phase 3 aesthetics, and Phase 5 Dynamic Neural Connectivity Arcs (PLV & Coherence).
- Brain3DWidget: Anatomical cortical surface mesh (data/human-brain.glb), strict bilateral symmetry (Z <-> -Z),
  midline longitudinal fissure bridge (Z = 0.0), outward surface normal offset (+0.055),
  Phase 5 3D Cortical Connectivity Arcs, and cold-standby zero-overhead pause.
- TopomapContainerWidget: QStackedWidget switching between 2D Topomap and 3D Brain with connectivity overlay controls.
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# PyQt5 GUI Framework
try:
    from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
    from PyQt5.QtGui import QColor, QFont, QPen, QBrush, QPainter, QPolygonF
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QCheckBox, QFrame, QStackedWidget, QSizePolicy
    )
    import pyqtgraph as pg
except ImportError:
    pass

# PyVista 3D Visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from config import (
    STANDARD_64_CHANNELS, MONTAGE_2D_COORDS, UNIFIED_LUT_256, PYVISTA_CMAP
)
from engine import EEGDataLoader


# ==============================================================================
# VOLTAGE COLORBAR DOCK WIDGET (-V to +V µV)
# ==============================================================================

class TopomapColorbarWidget(QWidget):
    """
    Dedicated Horizontal Voltage Colorbar:
    - Sits cleanly below 2D topomap.
    - Accurately indicates voltage range (-V to +V µV).
    """
    def __init__(self, v_scale: float = 50.0, parent=None):
        super().__init__(parent)
        self.v_scale = v_scale
        self.setFixedHeight(26)
        self.setMinimumWidth(220)

    def set_v_scale(self, v_scale: float):
        self.v_scale = v_scale
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_w = int(w * 0.75)
        bar_h = 8
        bar_x = (w - bar_w) // 2
        bar_y = 4

        # Draw segmented color blocks from UNIFIED_LUT_256
        n_steps = 64
        step_w = bar_w / float(n_steps)
        for i in range(n_steps):
            lut_idx = int((i / float(n_steps)) * 255)
            r, g, b, a = UNIFIED_LUT_256[lut_idx]
            painter.fillRect(
                QRectF(bar_x + i * step_w, bar_y, step_w + 1.0, bar_h),
                QColor(int(r), int(g), int(b))
            )

        painter.setPen(QPen(QColor('#323C4E'), 1))
        painter.drawRect(bar_x, bar_y, bar_w, bar_h)

        # Labels
        painter.setPen(QPen(QColor('#8C9BAE'), 1))
        painter.setFont(QFont("Consolas", 8, QFont.Bold))

        painter.drawText(bar_x - 38, bar_y + bar_h, f"-{self.v_scale:.0f}µV")
        painter.drawText(bar_x + bar_w + 6, bar_y + bar_h, f"+{self.v_scale:.0f}µV")
        painter.drawText(bar_x + (bar_w // 2) - 8, bar_y + bar_h + 10, "0")


# ==============================================================================
# 2D VOLTAGE TOPOMAP (WITH PHASE 5 NEURAL CONNECTIVITY ARCS)
# ==============================================================================

class Topomap2DWidget(QWidget):
    """
    2D Voltage Topographic Map:
    - High-Resolution (128x128) Regularized Multiquadric Interpolation.
    - Anti-aliased smoothstep circular alpha mask eliminating pixel staircase edges.
    - Clean Phase 3 aesthetics:
      * NO filled-in grey background (100% transparent outside & inside).
      * NO circular borders/dots around unselected electrode points (crisp white text labels only).
      * Selected channels feature prominent circular badges with white border.
      * Upward-pointing green triangle nose and crisp white head/ear outlines.
    - Phase 5: Dynamic Neural Connectivity Arcs (PLV & Spectral Coherence).
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.v_scale = 50.0
        self.show_connectivity = False

        # Retrieve cached interpolation structures
        self.cache = self.data_loader.interpolation_cache
        self.grid_res = self.cache.grid_res_2d
        self.idw_matrix = self.cache.idw_2d_matrix
        self.alpha_mask = self.cache.alpha_mask_2d
        self.valid_ch_names = self.cache.valid_ch_names_2d

        self.connectivity_curve_items: List[pg.PlotCurveItem] = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(0)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0E0F14')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        self.plot_widget.setRange(xRange=[-1.20, 1.20], yRange=[-1.20, 1.20])

        # Continuous RGBA heatmap item (Z=1)
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.img_item.setZValue(1)

        self._draw_head_schematic()

        # Scatter item for interactive clicking & selected channel badges (Z=10)
        self.scatter = pg.ScatterPlotItem(hoverable=True)
        self.scatter.sigClicked.connect(self._on_electrode_clicked)
        self.scatter.setZValue(10)
        self.plot_widget.addItem(self.scatter)

        # Crisp centered text labels (Z=15)
        self.label_items: Dict[str, pg.TextItem] = {}
        for ch in self.valid_ch_names:
            x, y = MONTAGE_2D_COORDS[ch]
            lbl = pg.TextItem(ch, color='#FFFFFF', anchor=(0.5, 0.5))
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setZValue(15)
            self.plot_widget.addItem(lbl)
            lbl.setPos(x, y)
            self.label_items[ch] = lbl

        layout.addWidget(self.plot_widget, stretch=1)

        # Bottom Colorbar Dock
        cbar_container = QWidget()
        cbar_container.setFixedHeight(30)
        cb_layout = QHBoxLayout(cbar_container)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setAlignment(Qt.AlignCenter)
        self.colorbar = TopomapColorbarWidget(self.v_scale, self)
        cb_layout.addWidget(self.colorbar)
        layout.addWidget(cbar_container)

        self._refresh_node_styles()

        # Initialize baseline image as 100% transparent (no filled-in grey disk)
        init_rgba = np.zeros((self.grid_res, self.grid_res, 4), dtype=np.uint8)
        self.img_item.setImage(init_rgba, autoLevels=False)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

    def _draw_head_schematic(self):
        theta = np.linspace(0, 2 * np.pi, 250)
        r = 1.0

        # White Circular Head Outline (Z=5)
        head_curve = pg.PlotCurveItem(r * np.cos(theta), r * np.sin(theta), pen=pg.mkPen('#FFFFFF', width=2.0))
        head_curve.setZValue(5)
        self.plot_widget.addItem(head_curve)

        # Upward-pointing Green Nose Triangle (Z=6) matching reference
        nose_x = np.array([-0.12, 0.0, 0.12, -0.12])
        nose_y = np.array([0.98, 1.15, 0.98, 0.98])
        nose_curve = pg.PlotCurveItem(nose_x, nose_y, pen=pg.mkPen('#00FFA3', width=2.5))
        nose_curve.setZValue(6)
        self.plot_widget.addItem(nose_curve)

        # White Left and Right Ears (Z=5)
        ear_l = pg.PlotCurveItem(
            np.array([-0.99, -1.08, -1.08, -0.99]),
            np.array([0.15, 0.08, -0.08, -0.15]),
            pen=pg.mkPen('#FFFFFF', width=2.0)
        )
        ear_l.setZValue(5)
        self.plot_widget.addItem(ear_l)

        ear_r = pg.PlotCurveItem(
            np.array([0.99, 1.08, 1.08, 0.99]),
            np.array([0.15, 0.08, -0.08, -0.15]),
            pen=pg.mkPen('#FFFFFF', width=2.0)
        )
        ear_r.setZValue(5)
        self.plot_widget.addItem(ear_r)

    def set_v_scale(self, v_scale: float):
        self.v_scale = v_scale
        self.colorbar.set_v_scale(v_scale)

    def update_voltage_frame(self, sample_idx: int):
        voltages = self.data_loader.get_channel_voltages(sample_idx)[:len(self.valid_ch_names)]
        grid_flat = np.dot(self.idw_matrix, voltages)
        v_norm = np.clip((grid_flat + self.v_scale) / (2.0 * self.v_scale), 0.0, 1.0)
        idx = (v_norm * 255.0).astype(np.uint8)

        # Map to RGBA and apply anti-aliased smoothstep alpha mask
        rgba = UNIFIED_LUT_256[idx].reshape((self.grid_res, self.grid_res, 4))
        rgba[:, :, 3] = (rgba[:, :, 3].astype(np.float32) * self.alpha_mask).astype(np.uint8)

        self.img_item.setImage(rgba, autoLevels=False)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

    # --------------------------------------------------------------------------
    # PHASE 5: DYNAMIC NEURAL CONNECTIVITY ARCS (2D)
    # --------------------------------------------------------------------------
    def set_connectivity_edges(self, edges: List[Tuple[str, str, float, Tuple[float, float], Tuple[float, float]]]):
        """Renders curved neural connectivity arcs across the 2D topomap array."""
        # Clear previous curves
        for item in self.connectivity_curve_items:
            self.plot_widget.removeItem(item)
        self.connectivity_curve_items.clear()

        if not self.show_connectivity or len(edges) == 0:
            return

        for ch1, ch2, strength, p1, p2 in edges:
            x1, y1 = p1
            x2, y2 = p2

            # Compute quadratic Bezier arc curved slightly towards center or outward
            t = np.linspace(0.0, 1.0, 30)
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            # Arc bend normal
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx**2 + dy**2)
            ctrl_x = mid_x - dy * 0.22
            ctrl_y = mid_y + dx * 0.22

            arc_x = (1 - t)**2 * x1 + 2 * (1 - t) * t * ctrl_x + t**2 * x2
            arc_y = (1 - t)**2 * y1 + 2 * (1 - t) * t * ctrl_y + t**2 * y2

            # Color and Alpha proportional to strength (Cyan to Gold)
            alpha_val = int(min(240, max(60, strength * 255)))
            if strength > 0.80:
                color = QColor(255, 214, 0, alpha_val)  # Gold
                width = 2.4
            else:
                color = QColor(0, 229, 255, alpha_val)  # Cyan
                width = 1.6

            pen = pg.mkPen(color, width=width)
            curve = pg.PlotCurveItem(arc_x, arc_y, pen=pen)
            curve.setZValue(8)  # Under electrodes, above heatmap
            self.plot_widget.addItem(curve)
            self.connectivity_curve_items.append(curve)

    def set_connectivity_visible(self, visible: bool):
        self.show_connectivity = visible
        if not visible:
            for item in self.connectivity_curve_items:
                self.plot_widget.removeItem(item)
            self.connectivity_curve_items.clear()

    def _on_electrode_clicked(self, item, points):
        if len(points) == 0:
            return
        ch_name = points[0].data()
        self.data_loader.channel_state.toggle_channel(ch_name)

    def _refresh_node_styles(self):
        spots = []
        for ch in self.valid_ch_names:
            nx, ny = MONTAGE_2D_COORDS[ch]
            is_on = self.data_loader.channel_state.is_selected(ch)
            
            if is_on:
                c_hex = self.data_loader.channel_colors.get(ch, '#00A3FF')
                spots.append({
                    'pos': (nx, ny),
                    'data': ch,
                    'size': 26,
                    'brush': pg.mkBrush(c_hex),
                    'pen': pg.mkPen('#FFFFFF', width=2.5)
                })
            else:
                spots.append({
                    'pos': (nx, ny),
                    'data': ch,
                    'size': 22,
                    'brush': pg.mkBrush(0, 0, 0, 1),
                    'pen': pg.mkPen(0, 0, 0, 0)
                })

            if ch in self.label_items:
                self.label_items[ch].setColor('#FFFFFF')

        self.scatter.setData(spots)


# ==============================================================================
# 3D CORTICAL TOPOMAP (WITH PHASE 5 3D NEURAL CONNECTIVITY ARCS)
# ==============================================================================

class Brain3DWidget(QWidget):
    """
    3D Topographic Heat Map Engine:
    - Renders data/human-brain.glb (or procedural cortical surface).
    - Decimates mesh (~3,500 vertices) for 60+ FPS rendering.
    - True Anatomical Electrode Placement:
      * Ellipsoid angular projection covering full frontal, temporal, parietal, and occipital lobes.
      * Strict bilateral symmetry (Z <-> -Z) for all 27 pairs.
      * Strict midline pinning (Z = 0.0) bridging the longitudinal fissure.
      * Surface normal / radial outward offset (+0.055) ensuring spheres are pinned to the exterior.
    - Phase 5: 3D Cortical Functional Connectivity Arcs (PLV / Coherence).
    - Cold-Standby Optimization:
      * Fully stops VTK rendering and IDW math when unselected or dissolved.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.is_active = False

        self.brain_mesh = None
        self.elec_mesh = None
        self.mesh_actor = None
        self.elec_actor = None
        self.highlight_actor = None
        self.scalar_bar_actor = None
        self.conn_3d_actor = None
        self.pts_per_sphere = 1

        self.show_heatmap = True
        self.show_electrodes = True
        self.show_scalar_bar = False
        self.show_connectivity_3d = False
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

    def pause_engine(self):
        """Halts all VTK rendering and calculations when 3D is unselected or dissolved."""
        self.is_active = False

    def resume_engine(self):
        """Resumes 3D engine updates upon switching to 3D mode."""
        self.is_active = True
        self.reset_camera_view()

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
            raw_mesh = pv.ParametricEllipsoid(xradius=1.85, yradius=1.25, zradius=1.45)
            raw_mesh.points += 0.05 * np.sin(raw_mesh.points * 8.0)

        center = raw_mesh.center
        raw_mesh.points -= center

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
            cmap=PYVISTA_CMAP,
            clim=self.clim,
            smooth_shading=True,
            specular=0.35,
            ambient=0.25,
            diffuse=0.85,
            show_scalar_bar=False,
            lighting=True
        )

    def _precompute_3d_idw(self):
        cache = self.data_loader.interpolation_cache
        if cache.idw_3d_matrix is not None:
            self.idw_3d_matrix = cache.idw_3d_matrix
            self.elec_coord_dict = cache.elec_coords_3d
            self.valid_ch_names = cache.valid_ch_names_3d
            return

        bounds = self.brain_mesh.bounds
        Lx = (bounds[1] - bounds[0]) / 2.0
        Ly = (bounds[3] - bounds[2]) / 2.0
        Lz = (bounds[5] - bounds[4]) / 2.0

        mesh_pts = np.array(self.brain_mesh.points, dtype=np.float32)

        PAIRS = [
            ('Fp1', 'Fp2'), ('AF7', 'AF8'), ('AF3', 'AF4'), ('F7', 'F8'), ('F5', 'F6'),
            ('F3', 'F4'), ('F1', 'F2'), ('FT7', 'FT8'), ('FC5', 'FC6'), ('FC3', 'FC4'),
            ('FC1', 'FC2'), ('T7', 'T8'), ('C5', 'C6'), ('C3', 'C4'), ('C1', 'C2'),
            ('T9', 'T10'), ('TP7', 'TP8'), ('CP5', 'CP6'), ('CP3', 'CP4'), ('CP1', 'CP2'),
            ('P7', 'P8'), ('P5', 'P6'), ('P3', 'P4'), ('P1', 'P2'), ('PO7', 'PO8'),
            ('PO3', 'PO4'), ('O1', 'O2')
        ]
        MIDLINE = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'Iz']

        self.elec_coord_dict = {}
        self.valid_ch_names = []

        def get_ellipsoid_target(ch_name: str) -> np.ndarray:
            x2, y2 = MONTAGE_2D_COORDS[ch_name]
            r = math.sqrt(x2**2 + y2**2)
            theta = min(math.pi * 0.49, r * (math.pi * 0.49 / 0.94))
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            ux = sin_t * (y2 / r) if r > 1e-4 else 0.0
            uz = sin_t * (x2 / r) if r > 1e-4 else 0.0
            uy = cos_t
            return np.array([Lx * ux * 1.05, Ly * uy * 1.05, Lz * uz * 1.05], dtype=np.float32)

        # 1. Place 27 Bilateral Symmetric Pairs
        for l_ch, r_ch in PAIRS:
            target_l = get_ellipsoid_target(l_ch)
            target_r = get_ellipsoid_target(r_ch)

            id_l = self.brain_mesh.find_closest_point(target_l)
            id_r = self.brain_mesh.find_closest_point(target_r)

            pt_l = mesh_pts[id_l]
            pt_r = mesh_pts[id_r]

            x_sym = (pt_l[0] + pt_r[0]) / 2.0
            y_sym = (pt_l[1] + pt_r[1]) / 2.0
            z_mag = (abs(pt_l[2]) + abs(pt_r[2])) / 2.0
            z_mag = max(z_mag, abs(MONTAGE_2D_COORDS[r_ch][0]) * Lz * 0.85)

            pos_r = np.array([x_sym, y_sym, +z_mag], dtype=np.float32)
            pos_l = np.array([x_sym, y_sym, -z_mag], dtype=np.float32)

            u_r = pos_r / (np.linalg.norm(pos_r) + 1e-6)
            u_l = pos_l / (np.linalg.norm(pos_l) + 1e-6)

            self.elec_coord_dict[r_ch] = pos_r + u_r * 0.055
            self.elec_coord_dict[l_ch] = pos_l + u_l * 0.055

        # 2. Place 10 Midline Channels bridging the longitudinal fissure
        for ch in MIDLINE:
            target = get_ellipsoid_target(ch)
            target[2] = 0.0
            best_id = self.brain_mesh.find_closest_point(target)
            pt = mesh_pts[best_id].copy()

            near_x = (np.abs(mesh_pts[:, 0] - pt[0]) < (0.08 * Lx)) & (np.abs(mesh_pts[:, 2]) < (0.22 * Lz))
            if np.any(near_x):
                pt[1] = max(pt[1], float(np.max(mesh_pts[near_x, 1])))

            pt[2] = 0.0
            u = pt / (np.linalg.norm(pt) + 1e-6)
            self.elec_coord_dict[ch] = pt + u * 0.055

        elec_pts = []
        for ch in STANDARD_64_CHANNELS:
            if ch in self.elec_coord_dict:
                elec_pts.append(self.elec_coord_dict[ch])
                self.valid_ch_names.append(ch)

        elec_arr = np.array(elec_pts, dtype=np.float32)

        diff = mesh_pts[:, np.newaxis, :] - elec_arr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-4
        weights = 1.0 / (dist ** 2.5)

        inferior_mask = mesh_pts[:, 1] < (-0.25 * Ly)
        weights[inferior_mask, :] *= 0.25

        weights /= np.sum(weights, axis=1, keepdims=True)
        self.idw_3d_matrix = weights.astype(np.float32)

        cache.idw_3d_matrix = self.idw_3d_matrix
        cache.elec_coords_3d = self.elec_coord_dict
        cache.valid_ch_names_3d = self.valid_ch_names
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
            cmap=PYVISTA_CMAP,
            clim=self.clim,
            smooth_shading=True,
            specular=0.8,
            show_scalar_bar=False,
            lighting=True
        )

    def reset_camera_view(self):
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        self.plotter.camera_position = [(0.0, 5.6, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def update_voltage_frame(self, voltages_64: np.ndarray, v_scale: float = 50.0):
        if not self.is_active or not PYVISTA_AVAILABLE or self.brain_mesh is None:
            return

        if self.show_heatmap and self.idw_3d_matrix is not None:
            cortex_scalars = np.dot(self.idw_3d_matrix, voltages_64[:len(self.valid_ch_names)])
            self.brain_mesh['voltage'] = cortex_scalars

        if self.show_electrodes and self.elec_mesh is not None:
            expanded_v = np.repeat(voltages_64[:len(self.valid_ch_names)], self.pts_per_sphere)
            self.elec_mesh['voltage'] = expanded_v

        self.plotter.render()

    # --------------------------------------------------------------------------
    # PHASE 5: 3D DYNAMIC NEURAL CONNECTIVITY ARCS
    # --------------------------------------------------------------------------
    def update_connectivity_3d(self, edges: List[Tuple[str, str, float, Tuple[float, float], Tuple[float, float]]]):
        if not self.is_active or not PYVISTA_AVAILABLE or self.plotter is None:
            return

        if self.conn_3d_actor is not None:
            self.plotter.remove_actor(self.conn_3d_actor)
            self.conn_3d_actor = None

        if not self.show_connectivity_3d or len(edges) == 0:
            return

        lines = []
        for ch1, ch2, strength, _, _ in edges:
            if ch1 in self.elec_coord_dict and ch2 in self.elec_coord_dict:
                p1 = self.elec_coord_dict[ch1]
                p2 = self.elec_coord_dict[ch2]
                # Slight outward bulge to float above cortex
                mid = (p1 + p2) / 2.0 * 1.12
                # Create 3-point spline
                spline = pv.Spline(np.vstack([p1, mid, p2]), n_points=12)
                lines.append(spline)

        if len(lines) > 0:
            multi_lines = lines[0]
            for l in lines[1:]:
                multi_lines = multi_lines.merge(l)
            self.conn_3d_actor = self.plotter.add_mesh(
                multi_lines,
                color='#00FFA3',
                line_width=2.5,
                lighting=False
            )
            self.plotter.render()

    def set_connectivity_visible(self, visible: bool):
        self.show_connectivity_3d = visible
        if not visible and self.conn_3d_actor is not None:
            self.plotter.remove_actor(self.conn_3d_actor)
            self.conn_3d_actor = None
            if self.is_active:
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
        if self.is_active:
            self.plotter.render()

    def set_selected_channels(self, selected_set: Set[str]):
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return

        if self.highlight_actor is not None:
            self.plotter.remove_actor(self.highlight_actor)
            self.highlight_actor = None

        if len(selected_set) > 0 and self.show_electrodes:
            sel_pts = []
            for ch in selected_set:
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
        if self.is_active:
            self.plotter.render()

    def _on_toggle_heatmap(self, checked: bool):
        self.show_heatmap = checked
        if not PYVISTA_AVAILABLE or self.mesh_actor is None:
            return
        mapper = self.mesh_actor.GetMapper()
        if mapper:
            mapper.SetScalarVisibility(checked)
        if self.is_active:
            self.plotter.render()

    def _on_toggle_electrodes(self, checked: bool):
        self.show_electrodes = checked
        if not PYVISTA_AVAILABLE:
            return
        if self.elec_actor is not None:
            self.elec_actor.SetVisibility(checked)
        if self.highlight_actor is not None:
            self.highlight_actor.SetVisibility(checked)
        if self.is_active:
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
        if self.is_active:
            self.plotter.render()


# ==============================================================================
# TOPOMAP CONTAINER (SWITCHABLE 2D / 3D WITH CONNECTIVITY CONTROLS)
# ==============================================================================

class TopomapContainerWidget(QWidget):
    """
    Unified Topomap Viewport:
    - Hosts a QStackedWidget switching between Page 0 (2D Topomap) and Page 1 (3D Brain).
    - Segmented Mode Selector: [ 2D Topomap ] | [ 3D Brain ].
    - Phase 5: Functional Neural Connectivity Arcs Toggle Checkbox.
    - Zero background overhead: When 2D is active, 3D engine is completely stopped.
    """
    mode_changed = pyqtSignal(str)

    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.current_mode = "2D"
        self.widget_3d: Optional[Brain3DWidget] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top Control Bar
        toolbar = QFrame()
        toolbar.setFixedHeight(36)
        toolbar.setStyleSheet("""
            QFrame {
                background: #060709;
                border-bottom: 1px solid #14171E;
            }
        """)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 2, 12, 2)
        tb_layout.setSpacing(10)

        # Segmented Control: [ 2D Topomap ] | [ 3D Brain ]
        seg_frame = QFrame()
        seg_frame.setFixedHeight(26)
        seg_frame.setStyleSheet("""
            QFrame {
                background: #10131B;
                border: 1px solid #1E2533;
                border-radius: 4px;
            }
        """)
        seg_layout = QHBoxLayout(seg_frame)
        seg_layout.setContentsMargins(2, 2, 2, 2)
        seg_layout.setSpacing(2)

        self.btn_2d = QPushButton("2D Topomap")
        self.btn_2d.setFixedHeight(22)
        self.btn_2d.setFocusPolicy(Qt.NoFocus)
        self.btn_2d.clicked.connect(lambda: self.set_mode("2D"))
        seg_layout.addWidget(self.btn_2d)

        self.btn_3d = QPushButton("3D Brain")
        self.btn_3d.setFixedHeight(22)
        self.btn_3d.setFocusPolicy(Qt.NoFocus)
        self.btn_3d.clicked.connect(lambda: self.set_mode("3D"))
        seg_layout.addWidget(self.btn_3d)

        tb_layout.addWidget(seg_frame)

        # Phase 5: Connectivity Arcs Overlay Checkbox
        self.chk_conn_arcs = QCheckBox("Neural Arcs")
        self.chk_conn_arcs.setToolTip("Overlay dynamic functional connectivity arcs (PLV / Coherence)")
        self.chk_conn_arcs.setChecked(False)
        self.chk_conn_arcs.toggled.connect(self._on_toggle_connectivity_arcs)
        self.chk_conn_arcs.setStyleSheet("color: #00FFA3; font-size: 10px; font-weight: bold;")
        tb_layout.addWidget(self.chk_conn_arcs)

        tb_layout.addStretch()

        # Active Channel Badge
        self.active_badge = QLabel("0 Active")
        self.active_badge.setStyleSheet("""
            background: #14221D;
            color: #00FFA3;
            border: 1px solid #00FFA3;
            border-radius: 3px;
            font-size: 9px;
            font-weight: bold;
            padding: 2px 6px;
        """)
        tb_layout.addWidget(self.active_badge)

        btn_select_all = QPushButton("Select All")
        btn_select_all.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #DDE2EB;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1F2737; }
        """)
        btn_select_all.clicked.connect(self._on_select_all)
        tb_layout.addWidget(btn_select_all)

        btn_clear = QPushButton("Clear")
        btn_clear.setStyleSheet("""
            QPushButton {
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #1F2737; color: #FFFFFF; }
        """)
        btn_clear.clicked.connect(self._on_clear)
        tb_layout.addWidget(btn_clear)

        layout.addWidget(toolbar)

        # Stacked Viewports (2D on Page 0, 3D on Page 1)
        self.stack = QStackedWidget()

        self.widget_2d = Topomap2DWidget(self.data_loader, self)
        self.stack.addWidget(self.widget_2d)

        layout.addWidget(self.stack, stretch=1)
        self._refresh_button_styles()

    def set_mode(self, mode: str):
        if mode == self.current_mode:
            return

        self.current_mode = mode
        if mode == "2D":
            self.stack.setCurrentIndex(0)
            if self.widget_3d is not None:
                self.widget_3d.pause_engine()
        elif mode == "3D":
            if self.widget_3d is None:
                print("[INFO] Lazy-initializing 3D Cortical Viewport...")
                self.widget_3d = Brain3DWidget(self.data_loader, self)
                self.stack.addWidget(self.widget_3d)
                self.widget_3d.set_clim(self.widget_2d.v_scale)
                self.widget_3d.set_selected_channels(self.data_loader.channel_state.selected_channels)
                self.widget_3d.set_connectivity_visible(self.chk_conn_arcs.isChecked())

            self.stack.setCurrentIndex(1)
            self.widget_3d.resume_engine()

        self._refresh_button_styles()
        self.mode_changed.emit(mode)

    def _on_toggle_connectivity_arcs(self, checked: bool):
        self.widget_2d.set_connectivity_visible(checked)
        if self.widget_3d is not None:
            self.widget_3d.set_connectivity_visible(checked)

    def update_connectivity_arcs(self, edges: List[Tuple[str, str, float, Tuple[float, float], Tuple[float, float]]]):
        if self.current_mode == "2D":
            self.widget_2d.set_connectivity_edges(edges)
        elif self.current_mode == "3D" and self.widget_3d is not None and self.widget_3d.is_active:
            self.widget_3d.update_connectivity_3d(edges)

    def update_active_badge(self, count: int):
        self.active_badge.setText(f"{count} Active")

    def _on_select_all(self):
        self.data_loader.channel_state.select_all()

    def _on_clear(self):
        self.data_loader.channel_state.clear_all()

    def _refresh_button_styles(self):
        active_style = """
            QPushButton {
                background: #1DB954;
                color: #000000;
                font-weight: bold;
                border-radius: 3px;
                border: none;
                padding: 2px 10px;
                font-size: 10px;
            }
        """
        inactive_style = """
            QPushButton {
                background: transparent;
                color: #8C9BAE;
                font-weight: bold;
                border-radius: 3px;
                border: none;
                padding: 2px 10px;
                font-size: 10px;
            }
            QPushButton:hover {
                color: #FFFFFF;
                background: #181E2B;
            }
        """
        if self.current_mode == "2D":
            self.btn_2d.setStyleSheet(active_style)
            self.btn_3d.setStyleSheet(inactive_style)
        else:
            self.btn_2d.setStyleSheet(inactive_style)
            self.btn_3d.setStyleSheet(active_style)
