"""
Lire des mesures : un CSV de tableur, un export Latis Pro, un export Regressi,
et les fichiers que tpllg.acquisition.sauvegarder écrit ; écrire un CSV
qu'un tableur ouvre. Voir doc/fichiers.md.

Le script écrit d'abord de petits fichiers de démonstration, tous préfixés
demo_ pour n'écraser aucune mesure, dans le dossier courant, puis les relit.
Les trois lecteurs rendent la même chose, une liste de tableaux, un par
colonne, et se taisent : verbose=True fait imprimer ce qu'ils ont compris.
"""

from pathlib import Path

import numpy as np

from tpllg.acquisition import charger, sauvegarder
from tpllg.fichiers import ecrire_csv, lire_csv, lire_latispro, lire_regressi

# un tableur exporté en CSV (point-virgule, virgule décimale, une ligne de titres)
Path("demo_mesures.csv").write_text(
    "f (Hz);Ve (V);Vs (V);phi (deg)\n500;1,00;0,60;-98\n2000;1,00;4,80;178\n8000;1,00;0,70;97\n",
    encoding="utf8",
)
f, Ve, Vs, phi = lire_csv("demo_mesures.csv", verbose=True)
print("f =", f, " Vs =", Vs, " phi =", phi)

# l'inverse : un CSV que le tableur ouvre, à virgule décimale, relu tel quel
ecrire_csv("demo_resultats.csv", [f, Vs / Ve], noms=["f (Hz)", "G"], decimale=",")
f2, G = lire_csv("demo_resultats.csv")
print("ecrire_csv puis lire_csv : f =", f2, " G =", G)

# un export Latis Pro : Fichier > Exporter > CSV, une colonne de temps par courbe
Path("demo_latispro.csv").write_text(
    "Temps;EA0;Temps;EA1\n0;1,2;0;0,5\n0,001;1,3;0,001;0,4\n0,002;1,1;0,002;0,3\n", encoding="cp1252"
)
t0, ea0, t1, ea1 = lire_latispro("demo_latispro.csv")  # toutes les colonnes
print("Latis Pro : t =", t0, " EA0 =", ea0, " EA1 =", ea1)

# un export Regressi : Fichier > Enregistrer sous > CSV, trois lignes d'en-tête, tabulations
Path("demo_regressi.txt").write_text(
    "Regressi\nt\tVs\tVe\ns\tV\tV\n0\t0.5\t1\n0.001\t0.6\t1\n", encoding="utf8"
)
t, Vs, Ve = lire_regressi("demo_regressi.txt")  # le temps en tête
print("Regressi : t =", t, " Vs =", Vs, " Ve =", Ve)

# ce que sauvegarder() écrit : un fichier par voie, deux lignes, le temps puis la tension
sauvegarder("demo", [0], [np.arange(4) * 1e-3], [[0.1, 0.2, 0.15, 0.05]])
temps, tensions = charger("demo", [0])  # ce que acquerir avait rendu, (1, 4)
t, u = np.loadtxt("demo_EA0.txt")  # une voie, directement
print("sauvegarder : t =", t, " u =", u, " charger :", temps.shape)
