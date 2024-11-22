import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import matplotlib.gridspec as gridspec

audio_path = "./data/sequence1/audio2.wav"
wav, sample_rate = librosa.load(audio_path, sr=22050)
time = np.arange(0, len(wav)) / sample_rate

print(f"Signal shape: {wav.shape}")
print(f"Sampling rate: {sample_rate} Hz")

spec = np.abs(librosa.stft(wav))
spec_db = librosa.amplitude_to_db(spec, ref=np.max)

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
