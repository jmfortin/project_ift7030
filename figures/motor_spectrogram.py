import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

############################ PARAMS ############################

SAMPLE_FREQ = 4096
N_FFT = 4096
HOP_LENGTH = 1024
WIN_LENGTH = 4096
WINDOW = 'hann'
CENTER = True

FIGSIZE = (12, 3)
DISPLAY = True
SAVE = True

################################################################

# Motor
wav1, sample_rate = librosa.load("./data/motor/audio.wav", sr=SAMPLE_FREQ)
time = np.arange(0, len(wav1)) / sample_rate
spec1 = librosa.stft(
    wav1, 
    n_fft=N_FFT, 
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window=WINDOW, 
    center=CENTER,
)
spec1_db = librosa.amplitude_to_db(np.abs(spec1), ref=1.0)

# Sequence 1
wav2, sample_rate = librosa.load("./data/sequence1/audio.wav", sr=SAMPLE_FREQ)
time = np.arange(0, len(wav2)) / sample_rate
spec2 = librosa.stft(
    wav2, 
    n_fft=N_FFT, 
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window=WINDOW, 
    center=CENTER,
)
spec2_db = librosa.amplitude_to_db(np.abs(spec2), ref=1.0)

# Find the global min and max values for the colormap
min_val = min(spec.min() for spec in [spec1_db, spec2_db])
max_val = max(spec.max() for spec in [spec1_db, spec2_db])

# Display waveform and spectrogram
fig = plt.figure(figsize=FIGSIZE)
ax1 = plt.subplot2grid((1, 10), (0, 0), colspan=1)
ax2 = plt.subplot2grid((1, 10), (0, 1), colspan=9)

img1 = librosa.display.specshow(spec1_db, ax=ax1, sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno', vmin=min_val, vmax=max_val)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")

img2 = librosa.display.specshow(spec2_db, ax=ax2, sr=sample_rate, x_axis='s', y_axis='hz', cmap='inferno', vmin=min_val, vmax=max_val)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("")
ax2.set_yticks([])
fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
plt.tight_layout()

if SAVE:
    print(f"Saving spectrogram and labels...")
    plt.savefig("./output/figures/motor_spectrogram.png")

if DISPLAY:
    print(f"Displaying waveform and spectrogram...")
    plt.show()

print("All done!")