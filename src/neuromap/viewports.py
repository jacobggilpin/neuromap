"""
neuromap - Topomap Viewports
Includes:
- TopomapColorbarWidget: Horizontal voltage colorbar
- Topomap2DWidget: 2D scalp voltage topomap with smooth regularized IDW interpolation
- Brain3DWidget: Cortical surface topomap with cold-standby rendering
- TopomapContainerWidget: Viewport switcher with lazy 3D loading
"""
import os
import math
from typing import List, Dict, Optional, Set

import numpy as np
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QLinearGradient
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QCheckBox, QStackedWidget
)
import pyqtgraph as pg

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    from .config import MONTAGE_2D_COORDS, UNIFIED_LUT_256, PYVISTA_CMAP
    from .engine import EEGDataLoader
except ImportError:
    from config import MONTAGE_2D_COORDS, UNIFIED_LUT_256, PYVISTA_CMAP
    from engine import EEGDataLoader

class TopomapColorbarWidget(QWidget):
    """
    Sleek Horizontal Voltage Colorbar for 2D Topomap:
    - Features clean title: 'Voltage (µV)'
    - Renders exact gradient using UNIFIED_COLORMAP stops (-V Blue -> Cyan -> Dark -> Coral -> +V Red)
    - Dynamically updates tick values (-V, 0.0, +V) in sync with Topomap Limit slider.
    """
    def __init__(self, v_scale: float = 50.0, parent=None):
        super().__init__(parent)
        self.v_scale = v_scale
        self.setFixedHeight(30)
        self.setFixedWidth(240)
        self.setStyleSheet("background: transparent;")

    def set_v_scale(self, v_scale: float):
        self.v_scale = v_scale
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Title: Voltage (µV)
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor("#8C9BAE"))
        painter.drawText(QRectF(0, 0, w, 11), Qt.AlignCenter, "Voltage (µV)")

        # Gradient Bar
        bar_x = 10
        bar_y = 12
        bar_w = w - 20
        bar_h = 6

        gradient = QLinearGradient(bar_x, bar_y, bar_x + bar_w, bar_y)
        gradient.setColorAt(0.0, QColor(30, 136, 229))   # -V Blue
        gradient.setColorAt(0.25, QColor(0, 229, 255))   # Cyan
        gradient.setColorAt(0.50, QColor(18, 21, 28))    # 0V Neutral Dark
        gradient.setColorAt(0.75, QColor(255, 110, 64))  # Coral Orange
        gradient.setColorAt(1.0, QColor(229, 57, 53))    # +V Crimson Red

        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor("#252D3C"), 1))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)

        # Tick Labels
        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor("#A0AEC0"))
        painter.drawText(QRectF(bar_x, bar_y + bar_h + 1, 60, 11), Qt.AlignLeft, f"-{self.v_scale:.0f}")
        painter.drawText(QRectF(bar_x + (bar_w // 2) - 25, bar_y + bar_h + 1, 50, 11), Qt.AlignCenter, "0")
        painter.drawText(QRectF(bar_x + bar_w - 60, bar_y + bar_h + 1, 60, 11), Qt.AlignRight, f"+{self.v_scale:.0f}")


class Topomap2DWidget(QWidget):
    """
    2D Voltage Topographic Map:
    - High-Resolution (128x128) Regularized Multiquadric Interpolation for smooth gradients.
    - Anti-aliased smoothstep circular alpha mask eliminating pixel staircase edges.
    - Phase 3 aesthetics restored:
      * NO filled-in grey background (100% transparent outside & inside).
      * NO circular borders/dots around unselected electrode points (crisp white text labels only).
      * Selected channels feature prominent circular badges with white border.
      * Upward-pointing green triangle nose and crisp white head/ear outlines.
    - Integrated dynamic Voltage Colorbar dock at bottom center.
    """
    def __init__(self, data_loader: EEGDataLoader, parent=None):
        super().__init__(parent)
        self.data_loader = data_loader
        self.v_scale = 50.0

        # Retrieve cached interpolation structures
        self.cache = self.data_loader.interpolation_cache
        self.grid_res = self.cache.grid_res_2d
        self.idw_matrix = self.cache.idw_2d_matrix
        self.alpha_mask = self.cache.alpha_mask_2d
        self.valid_ch_names = self.cache.valid_ch_names_2d

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
        voltages = self.data_loader.filtered_data[:len(self.valid_ch_names), sample_idx]
        grid_flat = np.dot(self.idw_matrix, voltages)
        v_norm = np.clip((grid_flat + self.v_scale) / (2.0 * self.v_scale), 0.0, 1.0)
        idx = (v_norm * 255.0).astype(np.uint8)

        # Map to RGBA and apply anti-aliased smoothstep alpha mask
        rgba = UNIFIED_LUT_256[idx].reshape((self.grid_res, self.grid_res, 4))
        rgba[:, :, 3] = (rgba[:, :, 3].astype(np.float32) * self.alpha_mask).astype(np.uint8)

        self.img_item.setImage(rgba, autoLevels=False)
        self.img_item.setRect(QRectF(-1.05, -1.05, 2.1, 2.1))

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
# PHASE 4: 3D CORTICAL TOPOMAP (COLD-STANDBY & RIGOROUS SURFACE PINNING)
# ==============================================================================

class Brain3DWidget(QWidget):
    """
    Phase 4: 3D Topographic Heat Map Engine:
    - Renders data/human-brain.glb (or procedural cortical surface).
    - Decimates mesh (~3,500 vertices) for 60+ FPS rendering.
    - True Anatomical Electrode Placement:
      * Ellipsoid angular projection covering full frontal, temporal, parietal, and occipital lobes.
      * Strict bilateral symmetry (Z <-> -Z) for all 27 pairs.
      * Strict midline pinning (Z = 0.0) bridging the longitudinal fissure.
      * Surface normal / radial outward offset (+0.055) ensuring spheres are pinned to the exterior.
    - Cold-Standby Optimization:
      * Fully stops VTK rendering and IDW math when unselected or dissolved.
      * Disabling electrodes hides both base spheres and selection halos.
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
        self.pts_per_sphere = 1

        self.show_heatmap = True
        self.show_electrodes = True
        self.show_scalar_bar = False
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
        """Precomputes 3D IDW Matrix and caches on SharedInterpolationCache."""
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

        # Store in shared cache
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
        """Top-down superior view looking straight down (+Y down), frontal cortex (+X) up."""
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
# TOPOMAP CONTAINER (TOGGLEABLE 2D MAP / 3D BRAIN WITH ZERO BACKGROUND OVERHEAD)
# ==============================================================================

class TopomapContainerWidget(QWidget):
    """
    Unified Topomap Viewport:
    - Hosts a QStackedWidget switching between Page 0 (2D Topomap) and Page 1 (3D Brain).
    - Segmented Mode Selector: [ 2D Topomap ] | [ 3D Brain ].
    - Lazy 3D Initialization: 3D engine is only allocated upon first click of [3D Brain].
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

        # Header Bar matching reference
        header = QFrame()
        header.setFixedHeight(34)
        header.setStyleSheet("""
            QFrame {
                background: #0B0C10;
                border-bottom: 1px solid #1A1F2C;
            }
            QPushButton {
                background: #14171E;
                color: #8C9BAE;
                border: 1px solid #232B3B;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1E2431;
                color: #E2E8F0;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 2, 10, 2)
        h_layout.setSpacing(8)

        self.title_lbl = QLabel("2D VOLTAGE TOPOMAP (64 CHANNELS)")
        self.title_lbl.setStyleSheet("color: #E2E8F0; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(self.title_lbl)

        # Active channels badge
        self.active_badge = QLabel("0 / 64 Active")
        self.active_badge.setStyleSheet("""
            background: #0D332D;
            color: #00FFA3;
            border: 1px solid #00FFA3;
            border-radius: 9px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: bold;
        """)
        h_layout.addWidget(self.active_badge)

        h_layout.addStretch()

        # [Select All] and [Clear] buttons
        self.btn_select_all = QPushButton("Select All")
        self.btn_clear = QPushButton("Clear")
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_clear.clicked.connect(self._on_clear)
        h_layout.addWidget(self.btn_select_all)
        h_layout.addWidget(self.btn_clear)

        # Segmented Mode Switcher
        self.btn_2d = QPushButton("2D Topomap")
        self.btn_3d = QPushButton("3D Brain")
        self.btn_2d.clicked.connect(lambda: self.set_mode("2D"))
        self.btn_3d.clicked.connect(lambda: self.set_mode("3D"))

        h_layout.addWidget(self.btn_2d)
        h_layout.addWidget(self.btn_3d)

        layout.addWidget(header)

        # Stacked Widget
        self.stack = QStackedWidget()
        self.widget_2d = Topomap2DWidget(self.data_loader, self)
        self.stack.addWidget(self.widget_2d)  # Index 0

        # Placeholder for lazy 3D loading (Index 1)
        self.placeholder_3d = QWidget()
        p_layout = QVBoxLayout(self.placeholder_3d)
        p_lbl = QLabel("Click [3D Brain] to initialize 3D Cortical Engine...")
        p_lbl.setAlignment(Qt.AlignCenter)
        p_lbl.setStyleSheet("color: #4A5568; font-size: 11px;")
        p_layout.addWidget(p_lbl)
        self.stack.addWidget(self.placeholder_3d)

        layout.addWidget(self.stack, stretch=1)
        self._refresh_button_styles()

    def set_mode(self, mode: str):
        if mode == self.current_mode:
            return
        self.current_mode = mode

        if mode == "2D":
            self.stack.setCurrentIndex(0)
            self.title_lbl.setText("2D VOLTAGE TOPOMAP (64 CHANNELS)")
            self.btn_select_all.setVisible(True)
            self.btn_clear.setVisible(True)
            if self.widget_3d is not None:
                self.widget_3d.pause_engine()
        else:
            if self.widget_3d is None:
                print("[INFO] Cold-starting 3D Cortical Engine...")
                self.widget_3d = Brain3DWidget(self.data_loader, self)
                self.stack.removeWidget(self.placeholder_3d)
                self.stack.addWidget(self.widget_3d)
                self.widget_3d.set_selected_channels(self.data_loader.channel_state.selected_channels)
                self.widget_3d.set_clim(self.widget_2d.v_scale)

            self.stack.setCurrentIndex(1)
            self.title_lbl.setText("3D CORTICAL TOPOMAP (PHASE 4 ENGINE)")
            self.btn_select_all.setVisible(False)
            self.btn_clear.setVisible(False)
            self.widget_3d.resume_engine()

        self._refresh_button_styles()
        self.mode_changed.emit(self.current_mode)

    def update_active_badge(self, count: int):
        self.active_badge.setText(f"{count} / 64 Active")

    def _on_select_all(self):
        self.data_loader.channel_state.select_all()

    def _on_clear(self):
        self.data_loader.channel_state.clear_all()

    def _refresh_button_styles(self):
        active_style = """
            background: #1B2B24;
            color: #00FFA3;
            border: 1px solid #00FFA3;
            border-radius: 3px;
            padding: 3px 10px;
            font-size: 10px;
            font-weight: bold;
        """
        inactive_style = """
            background: #14171E;
            color: #8C9BAE;
            border: 1px solid #232B3B;
            border-radius: 3px;
            padding: 3px 10px;
            font-size: 10px;
            font-weight: bold;
        """
        self.btn_2d.setStyleSheet(active_style if self.current_mode == "2D" else inactive_style)
        self.btn_3d.setStyleSheet(active_style if self.current_mode == "3D" else inactive_style)


# ==============================================================================
# CASCADING WAVEFORMS WIDGET (PHASE 4.03 SLIDING WINDOW)
# ==============================================================================

