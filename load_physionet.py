import mne

# 1. Download and load the data
subject = 1
runs = [1] 
file_paths = mne.datasets.eegbci.load_data(subject, runs)
raw = mne.io.read_raw_edf(file_paths[0], preload=True)
mne.datasets.eegbci.standardize(raw)

# 2. Plot the original, noisy data
print("Opening original data...")
raw.plot(duration=5.0, n_channels=20, title="1. BEFORE Filtering", block=True)

# 3. Apply the band-pass filter
# We use raw.copy() so we don't permanently overwrite our original raw variable
print("\nApplying filter (1 Hz to 40 Hz)...")
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

# 4. Plot the cleaned data
print("Opening filtered data...")
raw_filtered.plot(duration=5.0, n_channels=20, title="2. AFTER Filtering (1-40 Hz)", block=True)