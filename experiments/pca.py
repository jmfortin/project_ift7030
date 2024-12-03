import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import pandas as pd


############################ PARAMS ############################
  
DATA_FOLDER = "./data"
# POI_FILE = os.path.join(DATA_FOLDER, "trajectory.csv")
# AUDIO_FILE = os.path.join(DATA_FOLDER, "audio.wav")
SAMPLE_FREQ = 4096

# For the spectrogram
N_FFT = 4096
HOP_LENGTH = 1024
WIN_LENGTH = 4096
WINDOW = 'hann'
CENTER = True

# For PCA
N_COMPONENTS = 100

SAVE = False

################################################################

combined_wav = []

for folder in sorted(os.listdir(DATA_FOLDER)):
    if not os.path.isdir(os.path.join(DATA_FOLDER, folder)):
        continue
    print(f"Loading data from folder {folder}")
    audio_file = os.path.join(DATA_FOLDER, folder, "audio.wav")
    traj_file = os.path.join(DATA_FOLDER, folder, "trajectory.csv")
    wav, sample_rate = librosa.load(audio_file, sr=SAMPLE_FREQ)
    combined_wav.append(wav)

combined_wav = np.concatenate(combined_wav)
time = np.arange(0, len(combined_wav)) / sample_rate

print(f"Signal shape: {combined_wav.shape}")
print(f"Sampling rate: {sample_rate} Hz")
print(f"Duration: {len(combined_wav) / sample_rate} s")

print(f"Generating spectrogram...")
spec = librosa.stft(
    combined_wav, 
    n_fft=N_FFT, 
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window=WINDOW, 
    center=CENTER,
)
spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

spec_db_centered = spec_db - np.mean(spec_db, axis=0)   
cov_matrix = np.cov(spec_db_centered)
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eig_vals.real)[::-1]
sorted_eig_vals = eig_vals[sorted_indices]
sorted_eig_vecs = eig_vecs[:, sorted_indices].real

# Reconstruct signal with PCA
recons_only1 = sorted_eig_vecs[:, 0:1] @ (sorted_eig_vecs[:, 0:1].T @ spec_db_centered)
recons_only2 = sorted_eig_vecs[:, 1:2] @ (sorted_eig_vecs[:, 1:2].T @ spec_db_centered)
recons_only3 = sorted_eig_vecs[:, 2:3] @ (sorted_eig_vecs[:, 2:3].T @ spec_db_centered)
# recons_all = sorted_eig_vecs[:, :] @ (sorted_eig_vecs[:, :].T @ spec_db_centered)
recons_minus1 = sorted_eig_vecs[:, 1:] @ (sorted_eig_vecs[:, 1:].T @ spec_db_centered)
recons_minus2 = sorted_eig_vecs[:, 2:] @ (sorted_eig_vecs[:, 2:].T @ spec_db_centered)
recons_minus3 = sorted_eig_vecs[:, 3:] @ (sorted_eig_vecs[:, 3:].T @ spec_db_centered)

print(f"Spectrogram shape: {spec_db.shape}")
print(f"Displaying waveform and spectrogram...")

# Plot waveform and spectrogram
fig, axs = plt.subplots(4, 2, figsize=(16, 9))
axs[0][0].plot(time, combined_wav)
axs[0][0].set_title("Input Waveform")
axs[0][0].set_xlim([0, time[-1]])
axs[0][0].set_ylabel("Amplitude")
axs[0][0].set_xlabel("Time (s)")   

img = librosa.display.specshow(spec_db, ax=axs[0][1], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
axs[0][1].set_title("Spectrogram")
axs[0][1].set_xlabel("Time (s)")
axs[0][1].set_ylabel("Frequency (Hz)")

img = librosa.display.specshow(recons_only1, ax=axs[1][0], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
axs[1][0].set_title(f"PCA Component 1")
axs[1][0].set_xlabel("Time (s)")
axs[1][0].set_ylabel("Frequency (Hz)")

img = librosa.display.specshow(recons_minus1, ax=axs[1][1], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
axs[1][1].set_title(f"Reconstructed Spectrogram (Removed component 1)")
axs[1][1].set_xlabel("Time (s)")
axs[1][1].set_ylabel("Frequency (Hz)")

img = librosa.display.specshow(recons_only2, ax=axs[2][0], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
axs[2][0].set_title(f"PCA Component 2")
axs[2][0].set_xlabel("Time (s)")
axs[2][0].set_ylabel("Frequency (Hz)")

img = librosa.display.specshow(recons_minus2, ax=axs[2][1], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
axs[2][1].set_title(f"Reconstructed Spectrogram (Removed component 2)")
axs[2][1].set_xlabel("Time (s)")
axs[2][1].set_ylabel("Frequency (Hz)")

img = librosa.display.specshow(recons_only3, ax=axs[3][0], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
axs[3][0].set_title(f"PCA Component 3")
axs[3][0].set_xlabel("Time (s)")
axs[3][0].set_ylabel("Frequency (Hz)")

img = librosa.display.specshow(recons_minus3, ax=axs[3][1], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
axs[3][1].set_title(f"Reconstructed Spectrogram (Removed component 3)")
axs[3][1].set_xlabel("Time (s)")
axs[3][1].set_ylabel("Frequency (Hz)")

plt.tight_layout()
plt.savefig(os.path.join(DATA_FOLDER, "pca_reconstruction.png"))
plt.show()

if SAVE:
    print(f"Saving spectrogram and labels...")

    os.makedirs(os.path.join(DATA_FOLDER, "labels"), exist_ok=True)
    spec_file = os.path.join(DATA_FOLDER, "labels/spectrogram.csv")
    labels_file = os.path.join(DATA_FOLDER, "labels/labels.csv")
    labels_pca1_file = os.path.join(DATA_FOLDER, "labels/labels_pca1.csv")
    labels_pca2_file = os.path.join(DATA_FOLDER, "labels/labels_pca2.csv")
    labels_pca3_file = os.path.join(DATA_FOLDER, "labels/labels_pca3.csv")

    # Save spectrogram to CSV
    time_row = np.arange(0, spec_db.shape[1]) * HOP_LENGTH / sample_rate
    save_array = np.vstack((time_row, spec_db))
    pd.DataFrame(save_array).to_csv(spec_file, index=False, header=False)

    # Create labels file
    points_of_interest = pd.read_csv(POI_FILE)
    indices = [np.argmin(np.abs(time_row - timestamp)) for timestamp in points_of_interest["timestamp"]]
    pd.DataFrame(spec_db[:, indices]).to_csv(labels_file, index=False, header=False)
    pd.DataFrame(recons_minus1[:, indices]).to_csv(labels_pca1_file, index=False, header=False)
    pd.DataFrame(recons_minus2[:, indices]).to_csv(labels_pca2_file, index=False, header=False)
    pd.DataFrame(recons_minus3[:, indices]).to_csv(labels_pca3_file, index=False, header=False)

print("All done!")