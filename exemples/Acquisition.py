# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:35:03 2023

@author: a. marchand, f. legrand

Ce script dérive de deux exemples de Frédéric Legrand (f-legrand.fr,
CC BY-NC-SA 2.0 FR) : il est diffusé, comme le reste du dépôt, sous
CC BY-NC-SA 4.0, version ultérieure que la 2.0 FR autorise pour une adaptation.
- l'acquisition suit « Enregistrement d'un signal » :
  https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/pyacquis/pyacquis.html
- le spectre (tpllg.fft.spectre) reprend la fonction frequence() de
  « Mesure de déphasage » :
  https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/dephasage/dephasage.html
L'interface pycanum qu'ils emploient est documentée ici :
https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/interpy/interpy.html

Acquisition temporelles via Sysam SP5 des entrées analogiques EA
+ Analyse spectrale

CAN 12 bits
"""
from tpllg.sysam import Sysam
from tpllg.fft import spectre

import matplotlib.pyplot as plt
import numpy as np

# préfixe pour les noms des fichiers de sauvegarde
FILE_PREFIX = "signaltest"
# ENTREES ANALOGIQUES (EA)
ENTREES = [0]
# CALIBRE (0.2, 1, 5, 10)
# On peut aussi donner des valeurs différents si plusieurs voies, par ex. [10, 1, 1]
CALIBRE = 1

## PARAMETRES D'ÉCHANTILLONNAGE
# Fréquence d'échantillonnage en Hz (max 10 MHz)
fe = 20000.0
te = 1/fe
# durée de l'acquisition en s
T = 1.0
# nombre d'échantillons (max 130000 environ)
N = int(fe*T)

print(f"{fe=:.1e}Hz fréquence d'échantillonnage")
print(f"{te=:.1e}s pas de temps d'échantillonnage")
print(f"{N=:d} points d'acquisition")
print(f"Durée totale {T=:.1e}s")


# ACQUISITION SysamSP5
with Sysam(ENTREES, CALIBRE) as can:
    # configuration du CAN
    can.config_echantillon(te, N)
    # acquisition et récupération des données
    t, u = can.acquerir()

# BOUCLE sur les entrées pour exporter les résultats
for i in range(len(ENTREES)):
    # enregistrement dans un fichier texte des données
    t0 = t[i]
    u0 = u[i]
    np.savetxt(f"{FILE_PREFIX:s}_{i:02d}.txt", [t0, u0])

    # tracé et enregistrement de la figure
    plt.figure()
    plt.plot(t0, u0, "b")
    plt.xlabel("t (s)")
    plt.ylabel("u0 (V)")
    plt.axis([t0[0], t0[-1], -CALIBRE, CALIBRE])
    plt.grid()
    plt.savefig(f"{FILE_PREFIX:s}_{i:02d}.pdf")

    # il peut y avoir une différence avec les valeurs spécifiées au départ :
    fe = 1/(t0[1] - t0[0])

    # calcul des spectres
    f0, a0 = spectre(t0, u0)

    # tracé et enregistrement de la figure
    plt.figure()
    plt.plot(f0, a0)
    plt.xlabel("f (Hz)")
    plt.ylabel("Amplitude")
    plt.axis([0, fe, 0, CALIBRE])
    plt.grid()
    plt.savefig(f"{FILE_PREFIX:s}_{i:02d}_spectre.pdf")
    plt.show()
