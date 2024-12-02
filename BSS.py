import os
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile
import torch

# Charger le modèle pré-entraîné
model_name = "mdx_extra"  # Modèle Demucs
model = get_model(model_name)
model.cpu()
model.eval()

# Chemin vers le fichier audio
audio_path = "audio.wav"
output_dir = "separated"
os.makedirs(output_dir, exist_ok=True)

# Charger l'audio
waveform, sr = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
print("Forme initiale du signal :", waveform.shape)

if waveform.ndim == 1:
    waveform = waveform[None, None, :].repeat(1, 2, 1)
elif waveform.ndim == 2:
    waveform = waveform.unsqueeze(0)
print("Forme ajustée du signal :", waveform.shape)

# Appliquer le modèle pour séparer les sources
print("Séparation des sources...")
with torch.no_grad():
    sources = apply_model(model, waveform, device="cpu")

# Sauvegarder les fichiers séparés
print("Sauvegarde des sources séparées...")
sources_names = model.sources
for source, name in zip(sources[0], sources_names):
    output_path = os.path.join(output_dir, f"{name}.wav")
    AudioFile.write(output_path, source, model.samplerate)
    print(f"Sauvegardé : {output_path}")