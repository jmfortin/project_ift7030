## Projet IFT-7030
Projet du cours IFT-7030 - Prédiction audio-visuelle pour la navigation autonome hors-route
Par Jean-Michel Fortin, Ismaël Baldé et Hamed Soumahoro

### Installer les dépendances
Exécutez la commande suivante dans votre terminal :

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Téléchargement des données
Comme le jeu de données est assez gros, on vous fournit ici une seule séquence pour tester le code. Vous pouvez télécharger la séquence via [ce lien](https://ulavaldti-my.sharepoint.com/:u:/g/personal/jmfor48_ulaval_ca/Ed0nMZp24qxHgQxycfi2DigBg_0p4Aq-iiD6bD0Voky17g?e=hSeIeR). Il faut ensuite l'extraire dans le dossier `data`.

### Pré-entraînement sur le spectrogramme audio
Il suffit d'exécuter le script suivant, en ajustant les paramètres au besoin :

```bash
python3 scripts/train.py --project ift7030 --folds 5 --pca_drop 1 --lowpass_freq 0
```

### Adaptation pour la prédiction de vibrations
L'exécution du script utilisera le dernier checkpoint de pré-entraînement disponible:

```bash
python3 scripts/finetune.py --project ift7030 --folds 5 --latest --freeze --metric vibration
```

### Autres scripts
Le reste du code montre les expériences réalisées sur les données et les scripts permettant de générer les figures présentées dans le rapport. 
