import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from scipy.io import wavfile

# Fonction pour créer un filtre passe-bas
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Fonction pour créer un filtre passe-haut
def highpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Charger un fichier audio
fs, audio = wavfile.read("./data/sequence1/audio.wav") 
audio = audio.astype(float) 

# Paramètres du filtre
low_cutoff = 300  # Fréquence de coupure du passe-bas (en Hz)
high_cutoff = 300  # Fréquence de coupure du passe-haut (en Hz)

# Appliquer les filtres
lowpassed_signal = lowpass_filter(audio, low_cutoff, fs)
highpassed_signal = highpass_filter(audio, high_cutoff, fs)

# Générer les spectrogrammes
def plot_spectrogram(signal, fs, title, ax):
    f, t, Sxx = spectrogram(signal, fs, nperseg=1024)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='inferno')
    ax.set_title(title)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence (Hz)")
    ax.set_ylim(0, 2000)  # Limiter à 2 kHz pour visualisation (optionnel)

# Visualiser les résultats
time = np.arange(0, len(audio)) / fs

plt.figure(figsize=(12, 12))

# Signal original
plt.subplot(3, 2, 1)
plt.title("Signal original")
plt.plot(time, audio, color='blue')
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 2, 2)
plot_spectrogram(audio, fs, "Spectrogramme Original", plt.gca())

# Signal passe-bas
plt.subplot(3, 2, 3)
plt.title(f"Signal filtré passe-bas ({low_cutoff} Hz)")
plt.plot(time, lowpassed_signal, color='green')
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 2, 4)
plot_spectrogram(lowpassed_signal, fs, "Spectrogramme Passe-Bas", plt.gca())

# Signal passe-haut
plt.subplot(3, 2, 5)
plt.title(f"Signal filtré passe-haut ({high_cutoff} Hz)")
plt.plot(time, highpassed_signal, color='red')
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 2, 6)
plot_spectrogram(highpassed_signal, fs, "Spectrogramme Passe-Haut", plt.gca())

plt.tight_layout()
plt.show()

# Sauvegarder les signaux filtrés (facultatif)
wavfile.write("lowpassed_audio.wav", fs, lowpassed_signal.astype(np.int16))
wavfile.write("highpassed_audio.wav", fs, highpassed_signal.astype(np.int16))
