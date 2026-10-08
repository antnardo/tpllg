"""
Lire des mesures : un CSV de tableur, un export Latis Pro, un export Regressi,
et les fichiers que tpllg.acquisition.sauvegarder écrit. Voir doc/fichiers.md.

Le script écrit d'abord de petits fichiers de démonstration, tous préfixés
demo_ pour n'écraser aucune mesure, dans le dossier courant, puis les relit.
"""

from pathlib import Path

import numpy as np

from tpllg.acquisition import sauvegarder
from tpllg.fichiers import import_latispro, import_regressi, readcsv

# un tableur exporté en CSV (point-virgule, virgule décimale, une ligne de titres)
Path("demo_mesures.csv").write_text(
    "f (Hz);Ve (V);Vs (V);phi (deg)\n500;1,00;0,60;-98\n2000;1,00;4,80;178\n8000;1,00;0,70;97\n",
    encoding="utf8",
)
f, Ve, Vs, phi = readcsv("demo_mesures.csv")
print("f =", f, " Vs =", Vs, " phi =", phi)

# un export Latis Pro : Fichier > Exporter > CSV, une colonne de temps par courbe
Path("demo_latispro.csv").write_text(
    "Temps;EA0;Temps;EA1\n0;1,2;0;0,5\n0,001;1,3;0,001;0,4\n0,002;1,1;0,002;0,3\n", encoding="cp1252"
)
t0, ea0, t1, ea1 = import_latispro("demo_latispro.csv", colonnes=4)
print("Latis Pro : t =", t0, " EA0 =", ea0, " EA1 =", ea1)

# un export Regressi : Fichier > Enregistrer sous > CSV, trois lignes d'en-tête, tabulations
Path("demo_regressi.txt").write_text(
    "Regressi\nt\tVs\tVe\ns\tV\tV\n0\t0.5\t1\n0.001\t0.6\t1\n", encoding="utf8"
)
t, (Vs, Ve) = import_regressi("demo_regressi.txt", colonnes=2)
print("Regressi : t =", t, " Vs =", Vs, " Ve =", Ve)

# ce que sauvegarder() écrit : un fichier par voie, deux lignes, le temps puis la tension
sauvegarder("demo", [0], [np.arange(4) * 1e-3], [[0.1, 0.2, 0.15, 0.05]])
t, u = np.loadtxt("demo_EA0.txt")
print("sauvegarder : t =", t, " u =", u)
