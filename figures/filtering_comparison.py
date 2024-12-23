import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import load_audio, generate_spectrogram, lowpass_filter, remove_principal_components, display_spectograms

##################### PARAMETERS #######################

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, "..", "data")
SAMPLE_FREQ = 4096

# For the spectrogram
N_FFT = 4096
HOP_LENGTH = 1024
WIN_LENGTH = 4096
WINDOW = 'hann'
CENTER = True

# For low-pass filtering
LOWPASS_FREQ = 300

# For PCA
PCA_DROP = 1

#######################################################

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=16, titlesize=16)

combined_wav = []

for folder in sorted(os.listdir(DATA_FOLDER)):
    if folder == "motor":
        continue
    print(f"Loading data from folder {folder}")
    wav = load_audio(os.path.join(DATA_FOLDER, folder, "audio.wav"), SAMPLE_FREQ)
    combined_wav.append(wav)

combined_wav = np.concatenate(combined_wav)

spec_db = generate_spectrogram(combined_wav, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW, CENTER, db=True)

wav_filtered = lowpass_filter(combined_wav, LOWPASS_FREQ, SAMPLE_FREQ)
spec_db_filtered = generate_spectrogram(wav_filtered, N_FFT, HOP_LENGTH, WIN_LENGTH, WINDOW, CENTER, db=True)

spec_db_pca = remove_principal_components(spec_db, PCA_DROP)

display_spectograms(
    [spec_db, spec_db_filtered, spec_db_pca], 
    SAMPLE_FREQ, 
    ["Original", "Filtre passe-bas (300Hz)", "Retrait d'une composante principale"],
    save_path=os.path.join(SCRIPT_DIR, "..", "output", "figures", "filtering_comparison.png")
)