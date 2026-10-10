"""
Lire un fichier de mesures : le module csv, numpy.loadtxt, tpllg.fichiers.
Voir doc/fichiers.md.

Le fichier donnees/mesures.csv contient deux colonnes (temps, position)
séparées par des virgules, avec une ligne de titres. Trois façons de le lire :

- le module csv de la bibliothèque standard, ligne par ligne ;
- numpy.loadtxt, qui rend directement un tableau (delimiter, skiprows…) ;
- tpllg.fichiers.lire_csv, qui reconnaît le délimiteur et la virgule décimale
  des tableurs français, et écarte les lignes d'en-tête.

Le chemin du fichier est construit à partir de celui du script
(Path(__file__).parent) : le script fonctionne quel que soit le dossier
d'où on le lance. Enfin, numpy.save et numpy.load conservent un tableau dans
un fichier binaire .npy, sans perte de précision.
"""

import csv
import tempfile
from pathlib import Path

import numpy as np

from tpllg.fichiers import lire_csv

FICHIER = Path(__file__).parent / "donnees" / "mesures.csv"

# 1. Module csv
temps, position = [], []
with FICHIER.open(newline="", encoding="utf8") as f:
    lecteur = csv.reader(f)  # délimiteur « , » par défaut, sinon delimiter=";"
    next(lecteur)  # la ligne de titres
    for t, x in lecteur:
        temps.append(float(t))
        position.append(float(x))
print("csv :", temps, position)

# 2. numpy
temps, position = np.loadtxt(FICHIER, delimiter=",", skiprows=1, unpack=True)
print("numpy.loadtxt :", temps, position)

# 3. tpllg (entete : nombre de lignes à écarter, 1 par défaut ; verbose : l'écho du fichier)
temps, position = lire_csv(FICHIER, verbose=True)
print("tpllg.fichiers.lire_csv :", temps, position)

# Conserver un tableau : numpy.save et numpy.load (ici dans un dossier temporaire)
with tempfile.TemporaryDirectory() as dossier:
    chemin = Path(dossier) / "temps.npy"
    np.save(chemin, temps)
    print("relu :", np.load(chemin))
