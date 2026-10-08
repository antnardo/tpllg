"""
Acquérir les entrées analogiques à la Sysam SP5, enregistrer chaque voie, et
tracer son signal et son spectre.

Sans centrale (pycanum absent), le simulateur prend le relais et le script
tourne jusqu'au bout : les tracés ne montrent alors qu'un bruit de
quantification. Voir doc/centrale.md et doc/spectres.md.

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

@author: a. marchand, f. legrand
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.acquisition import sauvegarder
from tpllg.fft import spectre
from tpllg.sysam import Sysam

PREFIXE = "signaltest"  # signaltest_EA0.txt, signaltest_EA0.pdf, signaltest_EA0_spectre.pdf
ENTREES = [0]  # les entrées analogiques, par ex. [0, 1, 2]
CALIBRE = 1  # V : 0.2, 1, 5 ou 10 ; un par voie si plusieurs, par ex. [10, 1, 1]

# L'échantillonnage
fe = 20000.0  # Hz (10 MHz au plus sur EA0 à EA3 seules)
te = 1 / fe
T = 1.0  # s, la durée de l'acquisition
N = int(fe * T)  # au plus Sysam.n_max(len(ENTREES)) : 261 888 points pour une voie

print(f"{fe=:.1e} Hz, fréquence d'échantillonnage")
print(f"{te=:.1e} s, période d'échantillonnage")
print(f"{N=:d} points")
print(f"{T=:.1e} s, durée totale")

with Sysam(ENTREES, CALIBRE) as can:
    can.config_echantillon(te, N)
    temps, tensions = can.acquerir()
    calibres = can.calibres  # un par voie, ceux que la centrale a pris

sauvegarder(PREFIXE, ENTREES, temps, tensions)

for ea, calibre, t, u in zip(ENTREES, calibres, temps, tensions):
    plt.figure()
    plt.plot(t, u, "b")
    plt.xlabel("t (s)")
    plt.ylabel(f"EA{ea} (V)")
    plt.axis([t[0], t[-1], -calibre, calibre])
    plt.grid()
    plt.savefig(f"{PREFIXE}_EA{ea}.pdf")

    # le spectre, jusqu'à fe/2 : au-delà, ce n'est que son miroir
    fe_reelle = 1 / (t[1] - t[0])  # la centrale arrondit te au dixième de microseconde
    f, a = spectre(t, u)
    plt.figure()
    plt.plot(f, a)
    plt.xlabel("f (Hz)")
    plt.ylabel(f"amplitude sur EA{ea} (V)")
    plt.axis([0, fe_reelle / 2, 0, max(a.max() * 1.1, np.finfo(float).eps)])
    plt.grid()
    plt.savefig(f"{PREFIXE}_EA{ea}_spectre.pdf")

plt.show()
