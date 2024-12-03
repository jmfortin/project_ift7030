import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from scipy.signal import butter, filtfilt


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
    plt.show()


def inference_on_datamodule(model, datamodule, device):

    model.eval()
    model.to(device)
    
    all_labels = np.array([])
    all_outputs = np.array([])
    for batch in datamodule.test_dataloader():
        inputs, labels = batch
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        with torch.no_grad():
            outputs = model(inputs)

        # Transpose the outputs and labels
        outputs = outputs.permute(1, 0).cpu().numpy()
        labels = labels.permute(1, 0).cpu().numpy()
        if all_labels.size == 0:
            all_labels = labels
            all_outputs = outputs
        else:
            all_labels = np.hstack((all_labels, labels))
            all_outputs = np.hstack((all_outputs, outputs))

    return all_labels, all_outputs