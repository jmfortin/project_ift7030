import numpy as np
import librosa

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