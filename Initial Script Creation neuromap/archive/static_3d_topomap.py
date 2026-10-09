import mne
import numpy as np
import pyvista as pv

print("Loading EEG data...")
subject = 1
runs = [1]
file_paths = mne.datasets.eegbci.load_data(subject, runs)
raw = mne.io.read_raw_edf(file_paths[0], preload=True)
mne.datasets.eegbci.standardize(raw)
raw.filter(l_freq=1.0, h_freq=40.0)
raw.set_montage('colin27_1005')

# Extract Frame 400 and coordinates
voltages = raw.get_data()[:, 400]
ch_names = raw.ch_names
pos_dict = raw.get_montage().get_positions()['ch_pos']

# Note: We do NOT apply the fsaverage transformation here. 
# We leave the electrodes in their default mathematical space.
electrodes = pv.PolyData(np.array([pos_dict[ch] for ch in ch_names]))
electrodes['Voltage'] = voltages

print("Loading and optimizing custom Sketchfab model...")
# 1. Load the GLB file
# Replace 'custom_head.glb' with your actual file name
head_mesh = pv.read('data/male_head_base_mesh.glb')

# GLB files often import as a "MultiBlock" (a folder of separate parts). 
# We must combine them into a single continuous surface for the color wrapping.
if isinstance(head_mesh, pv.MultiBlock):
    head_mesh = head_mesh.combine()

# 2. Performance Optimization
# Remove 80% of the polygons to ensure smooth rendering and fast interpolation
head_mesh = head_mesh.decimate(0.8)

# 3. Manual Coregistration (You will need to change these numbers!)
# MNE electrode coordinates are tiny (spanning roughly -0.1 to 0.1). 
# You must scale your 3D model up or down to fit inside the electrode cluster.
head_mesh.points *= 0.10  # Try 0.01 if the head is huge, or 10.0 if it's tiny

# Shift the head along the X, Y, or Z axis to center it inside the electrodes
head_mesh.translate([0.0, -0.02, 0.0], inplace=True) 

print("Painting the brainwaves across the scalp...")
# Wrap the colors. You may need to increase the radius (e.g., to 0.1) 
# if your head model is slightly smaller than the electrode radius.
head_surface = head_mesh.interpolate(electrodes, radius=0.08, sharpness=3)

print("Opening 3D viewer...")
plotter = pv.Plotter(title="Custom 3D Head EEG")
plotter.set_background('gray')

plotter.add_mesh(head_surface, scalars='Voltage', cmap='jet', 
                 clim=[-5e-5, 5e-5], show_scalar_bar=True, smooth_shading=True)

plotter.add_points(electrodes, color='lime', point_size=8, render_points_as_spheres=True)

plotter.show()