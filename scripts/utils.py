import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt, welch
import pandas as pd
from torch.nn.functional import unfold


def lowpass_filter(signal, cutoff_freq, sample_freq, order=5):

    nyquist_freq = sample_freq / 2
    cutoff_freq /= nyquist_freq
    b, a = butter(order, cutoff_freq, btype='low', analog=False)
    return filtfilt(b, a, signal)    


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
    

def generate_spectrogram(signal, nfft, hop_length, win_length, window='hann', center=True, db=True):

    spec = librosa.stft(
        signal, 
        n_fft=nfft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )

    if db:
        return librosa.amplitude_to_db(np.abs(spec))
    else:  
        return np.abs(spec)


def load_audio(file_path, sample_freq):

    wav, _ = librosa.load(file_path, sr=sample_freq)
    return wav


def display_spectograms(specs, sample_freq, titles, x_axis='s', y_axis='hz', cmap='inferno', save_path=None):

    fig, axs = plt.subplots(len(specs), 1, figsize=(12, 3*len(specs)))
    
    # Find the global min and max values for the colormap
    min_val = min(spec.min() for spec in specs)
    max_val = max(spec.max() for spec in specs)
    
    for i, spec in enumerate(specs):
        img = librosa.display.specshow(spec, sr=sample_freq, x_axis=x_axis, y_axis=y_axis, cmap=cmap, ax=axs[i], vmin=min_val, vmax=max_val)
        axs[i].set_title(titles[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def compute_vibration(imu_data, freq_range=[1, 30], window_duration=1.0):
    """
    Compute the vibration metric based on z-axis acceleration.
    Based on the method in "How Does It Feel? Self-Supervised Costmap Learning for
    Off-Road Vehicle Traversability". See https://github.com/castacks/learned_cost_map.

    Args:
        freq_range (list): The minimum and maximum frequencies to consider.
        window_duration (float): The duration of the window in seconds.
    """

    z_acc_data = imu_data["acc_z"]

    sensor_freq = 1 / (imu_data["timestamp"].diff().mean() / 1e9)
    window_length = int(window_duration * sensor_freq)
    if window_length % 2 == 0:  # Ensure odd window length
        window_length += 1

    windows = unfold(torch.tensor(z_acc_data).view(1, 1, -1), kernel_size=(1, window_length), stride=1).T
    freqs, psd = welch(windows, sensor_freq, nperseg=window_length)
    idx_band = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    bandpower = simpson(psd[:, idx_band], dx=(freqs[1]-freqs[0]), axis=1)
    metric = np.log(bandpower + 1)
    timestamps = imu_data["timestamp"].iloc[window_length//2:-(window_length//2)].values

    return pd.DataFrame({"timestamp": timestamps, "metric": metric})
