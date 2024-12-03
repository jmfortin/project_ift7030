from os.path import join, dirname, realpath
import numpy as np
import librosa
import matplotlib.pyplot as plt

SCRIPT_DIR = dirname(realpath(__file__))
DATA_FOLDERS = [
    join(SCRIPT_DIR, "..", "data", "sequence1"),
    join(SCRIPT_DIR, "..", "data", "sequence2"),
    join(SCRIPT_DIR, "..", "data", "sequence3"),
    # join(SCRIPT_DIR, "..", "data", "sequence4"),
    join(SCRIPT_DIR, "..", "data", "sequence6"),
    join(SCRIPT_DIR, "..", "data", "sequence5"),
]


fig, axs = plt.subplots(len(DATA_FOLDERS), 1, figsize=(10, 15))

# Initialize min and max values for y-axis
min_val, max_val = float('inf'), float('-inf')

# First pass to find the min and max values
for folder in DATA_FOLDERS:
    file_path = join(folder, "audio.wav")
    signal, sr = librosa.load(file_path, sr=None)
    min_val = min(min_val, np.min(signal))
    max_val = max(max_val, np.max(signal))

# Second pass to plot the signals with the same y-axis limits
for i, folder in enumerate(DATA_FOLDERS):
    print(f"Loading audio from {folder}")
    file_path = join(folder, "audio.wav")
    signal, sr = librosa.load(file_path, sr=None)

    # Plot the signal
    time = np.linspace(0, len(signal) / sr, num=len(signal))
    axs[i].plot(time, signal)
    axs[i].set_ylim(min_val, max_val)
    axs[i].axis('off')
    for spine in axs[i].spines.values():
        spine.set_visible(False)

    # Compute and plot the spectrogram
    # S = librosa.stft(signal)
    # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    # img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=axs[i, 1])
    # fig.colorbar(img, ax=axs[i, 1], format="%+2.0f dB")


plt.tight_layout()
plt.show()