import mne
import matplotlib.pyplot as plt

# 1. Load and filter the data (same as before)
print("Loading and filtering data...")
subject = 1
runs = [1] 
file_paths = mne.datasets.eegbci.load_data(subject, runs)
raw = mne.io.read_raw_edf(file_paths[0], preload=True)
mne.datasets.eegbci.standardize(raw)
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

# 2. Attach a Montage (3D spatial coordinates)
# PhysioNet uses a standard 64-channel cap layout known as the "10-05" system.
print("Applying 3D sensor locations...")
montage = mne.channels.make_standard_montage('standard_1005')
raw_filtered.set_montage(montage)

# 3. Plot the 3D Sensor Layout
# This opens an interactive 3D window showing where the electrodes sit on the head
print("Opening 3D sensor plot...")
raw_filtered.plot_sensors(kind='3d', ch_type='eeg', show_names=True)

# 4. Plot 2D Color Topomaps
# This calculates the energy of the waves and plots them on 2D head models
print("Calculating and plotting PSD Topomaps...")
# We limit fmax to 40 Hz because we filtered out everything above that
fig = raw_filtered.compute_psd(fmax=40.0).plot_topomap()

# Keep the matplotlib windows open so you can view them
plt.show()