# -*- coding: utf-8 -*-
"""
Lire des mesures : un CSV de tableur, un export Latis Pro, un export Regressi,
et les fichiers que tpllg.acquisition.sauvegarder écrit. Voir doc/fichiers.md.

Le script écrit d'abord de petits fichiers de démonstration dans le dossier
courant, puis les relit.
"""
import numpy as np

from tpllg.fichiers import import_latispro, import_regressi, readcsv

# un tableur exporté en CSV (point-virgule, virgule décimale, une ligne de titres)
with open("mesures.csv", "w", encoding="utf8") as fichier:
    fichier.write("f (Hz);Ve (V);Vs (V);phi (deg)\n"
                  "500;1,00;0,60;-98\n"
                  "2000;1,00;4,80;178\n"
                  "8000;1,00;0,70;97\n")
f, Ve, Vs, phi = readcsv("mesures.csv")
print("f =", f, " Vs =", Vs, " phi =", phi)

# un export Latis Pro : Fichier > Exporter > CSV, une colonne de temps par courbe
with open("latispro.csv", "w", encoding="iso8859") as fichier:
    fichier.write("Temps;EA0;Temps;EA1\n"
                  "0;1,2;0;0,5\n"
                  "0,001;1,3;0,001;0,4\n"
                  "0,002;1,1;0,002;0,3\n")
t0, ea0, t1, ea1 = import_latispro("latispro.csv", colonnes=4)
print("Latis Pro : t =", t0, " EA0 =", ea0, " EA1 =", ea1)

# un export Regressi : Fichier > Enregistrer sous > CSV, trois lignes d'en-tête, tabulations
with open("regressi.txt", "w", encoding="utf8") as fichier:
    fichier.write("Regressi\nt\tVs\tVe\ns\tV\tV\n"
                  "0\t0.5\t1\n"
                  "0.001\t0.6\t1\n")
t, (Vs, Ve) = import_regressi("regressi.txt", colonnes=2)
print("Regressi : t =", t, " Vs =", Vs, " Ve =", Ve)

# ce que sauvegarder() écrit : deux lignes, le temps puis la tension
np.savetxt("essai_EA0.txt", [np.arange(4)*1e-3, [0.1, 0.2, 0.15, 0.05]])
t, u = np.loadtxt("essai_EA0.txt")
print("sauvegarder : t =", t, " u =", u)
