# Projet IFT-7030
Projet du cours IFT-7030 - Prédiction de spectre audio à partir d'images de terrain

## Sous-projet 1: Séparation de source audio

L'objectif est de séparer le bruit causé par les moteurs du véhicules du son résultant de l'interaction entre les roues et le sol. 
Ça revient à de la séparation de source, mais on n'a pas le ground truth parfait pour chaque séquence.
On peut envisager générer un signal audio des moteurs à vide (donc sans contact entre le roues et le sol) afin de guider la séparation. 

## Sous-projet 2: Prédiction audio à partir d'images

L'objectif est de prédire un spectrogramme audio correspondant à la traversée d'un terrain représenté par une image aérienne. 
L'image suivante représente ce qu'il y a à faire: 

![diagram_ift7030](https://github.com/user-attachments/assets/3b44c1c3-bc4a-43ca-9b6a-457ddae712fb)

Au final, il faut sélectionner un réseau de neurones représenté par les blocs bleus dans le diagramme.
Commençons par une architecture simple et augmentons graduellement vers plus complexe au besoin. 
