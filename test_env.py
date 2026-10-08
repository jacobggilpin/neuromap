import mne
import pyqtgraph as pg

print(f"MNE Version: {mne.__version__}")
print(f"PyQtGraph Version: {pg.__version__}")

print("\nDownloading MNE sample dataset (this might take a minute the first time)...")
# This automatically fetches a standard EEG dataset used for testing
sample_data_folder = mne.datasets.sample.data_path()

print(f"\nSuccess! Environment is working and sample data is ready at:\n{sample_data_folder}")