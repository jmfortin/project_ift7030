import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import pandas as pd

############################ PARAMS ############################
  
DATA_FOLDER = "./data/sequence1"
POI_FILE = os.path.join(DATA_FOLDER, "trajectory.csv")
AUDIO_FILE = os.path.join(DATA_FOLDER, "audio.wav")

SAMPLE_FREQ = 4096
N_FFT = 4096
HOP_LENGTH = 1024
WIN_LENGTH = 4096
SAMPLE_FREQ = 4000
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
WINDOW = 'hann'
CENTER = True

DISPLAY = False
SAVE = True

################################################################

wav, sample_rate = librosa.load(AUDIO_FILE, sr=SAMPLE_FREQ)
time = np.arange(0, len(wav)) / sample_rate

print(f"Signal shape: {wav.shape}")
print(f"Sampling rate: {sample_rate} Hz")
print(f"Duration: {len(wav) / sample_rate} s")

print(f"Generating spectrogram...")
spec = librosa.stft(
    wav, 
    n_fft=N_FFT, 
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window=WINDOW, 
    center=CENTER,
)
spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

print(f"Spectrogram shape: {spec_db.shape}")

if DISPLAY:
    print(f"Displaying waveform and spectrogram...")

    # Plot waveform and spectrogram
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].plot(time, wav)
    axs[0].set_title("Input Waveform")
    axs[0].set_xlim([0, time[-1]])
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlabel("Time (s)")   

    img = librosa.display.specshow(spec_db, ax=axs[1], sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno')
    # fig.colorbar(img, ax=axs[1], format='%+2.0f dB')
    axs[1].set_title("Spectrogram")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

if SAVE:
    print(f"Saving spectrogram and labels...")

    os.makedirs(os.path.join(DATA_FOLDER, "labels"), exist_ok=True)
    spec_file = os.path.join(DATA_FOLDER, "labels/spectrogram.csv")
    labels_file = os.path.join(DATA_FOLDER, "labels/labels.csv")

    # Save spectrogram to CSV
    time_row = np.arange(0, spec_db.shape[1]) * HOP_LENGTH / sample_rate
    save_array = np.vstack((time_row, spec_db))
    pd.DataFrame(save_array).to_csv(spec_file, index=False, header=False)

    # Create labels file
    points_of_interest = pd.read_csv(POI_FILE)
    indices = [np.argmin(np.abs(time_row - timestamp)) for timestamp in points_of_interest["timestamp"]]
    labels = spec_db[:, indices]
    pd.DataFrame(labels).to_csv(labels_file, index=False, header=False)

print("All done!")