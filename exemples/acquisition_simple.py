"""
Acquérir deux voies à la Sysam SP5, enregistrer un fichier par voie, tracer.

Sans centrale (pycanum absent), le simulateur prend le relais et le script
tourne jusqu'au bout : les tracés ne montrent alors qu'un bruit de
quantification. Voir doc/centrale.md.
"""

import matplotlib.pyplot as plt

from tpllg.acquisition import acquerir, sauvegarder

PREFIXE = "essai"  # essai_EA0.txt, essai_EA1.txt, essai.pdf
ENTREES = [0, 1]  # EA0 et EA1 ; [0] pour une seule voie
CALIBRE = 5  # V : 0.2, 1, 5 ou 10
fe = 100000.0  # Hz
T = 0.05  # s
te, N = 1 / fe, int(fe * T)

temps, tensions = acquerir(ENTREES, CALIBRE, te, N)
sauvegarder(PREFIXE, ENTREES, temps, tensions)
voies, points = temps.shape
print(f"acquis : {voies} voie(s) de {points} points, de {temps[0][0]} à {temps[0][-1]} s")

# squeeze=False : un tableau de repères même pour une seule voie
fig, axes = plt.subplots(len(ENTREES), 1, sharex=True, squeeze=False)
for ax, ea, t, u in zip(axes[:, 0], ENTREES, temps, tensions):
    ax.plot(t * 1e3, u)
    ax.set_ylabel(f"EA{ea} (V)")
    ax.grid()
axes[-1, 0].set_xlabel("t (ms)")
plt.savefig(PREFIXE + ".pdf")
plt.show()
