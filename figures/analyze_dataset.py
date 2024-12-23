from os.path import join, dirname, realpath
import numpy as np
import librosa
import matplotlib.pyplot as plt

SCRIPT_DIR = dirname(realpath(__file__))
DATA_FOLDERS = [
    join(SCRIPT_DIR, "..", "data", "sequence1"),
    # join(SCRIPT_DIR, "..", "data", "sequence2"),
    # join(SCRIPT_DIR, "..", "data", "sequence3"),
    # join(SCRIPT_DIR, "..", "data", "sequence4"),
    # join(SCRIPT_DIR, "..", "data", "sequence6"),
    # join(SCRIPT_DIR, "..", "data", "sequence5"),
]


def remove_principal_components(data, nb_remove, nb_total=None):
    
    data_centered = data - np.mean(data, axis=0)   
    cov_matrix = np.cov(data_centered)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals.real)[::-1]
    sorted_eig_vecs = eig_vecs[:, sorted_indices].real

    if nb_total is None:
        nb_total = len(sorted_eig_vecs)

    bases = sorted_eig_vecs[:, nb_remove:nb_total]
    weights = bases.T @ data_centered
    recons_data = bases @ weights
    return recons_data


fig, axs = plt.subplots(len(DATA_FOLDERS), 4, figsize=(10, 15))

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
    signal, sr = librosa.load(file_path, sr=4096)

    # Plot the signal
    time = np.linspace(0, len(signal) / sr, num=len(signal))
    axs[i, 0].plot(time, signal)
    axs[i, 0].set_ylim(min_val, max_val)

    # Compute and plot the spectrogram
    S = librosa.stft(signal, n_fft=4096, hop_length=1024, win_length=4096, window='hann', center=True)
    S_db = librosa.amplitude_to_db(np.abs(S))

    img = librosa.display.specshow(S_db, sr=sr, x_axis='s', y_axis='hz', ax=axs[i, 1], cmap='inferno', vmin=S_db.min(), vmax=S_db.max())

    S_phase = np.angle(S)
    S_pca_db = remove_principal_components(S_db, 1)    
    S_pca = librosa.db_to_amplitude(S_pca_db) * np.exp(1j * S_phase)
    img = librosa.display.specshow(S_pca_db, sr=sr, x_axis='s', y_axis='hz', ax=axs[i, 3], cmap='inferno', vmin=S_db.min(), vmax=S_db.max())

    # Recontruct the signal from the spectrogram
    signal_reconstructed = librosa.istft(S_pca, length=len(signal))
    axs[i, 2].plot(time, signal_reconstructed)
    axs[i, 2].set_ylim(min_val, max_val)



plt.tight_layout()
plt.show()