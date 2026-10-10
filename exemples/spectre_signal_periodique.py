"""
Le spectre d'un signal périodique échantillonné : un signal construit par ses
harmoniques (tpllg.harmoniques.Signal), échantillonné, et son spectre par
tpllg.fft.calcule_DFT. Voir doc/spectres.md et doc/harmoniques.md.

Un signal de fréquence f = 200 Hz est la somme d'harmoniques d'amplitudes et
de phases données. Échantillonné à fe = 4 kHz sur N = 4000 points, soit une
durée de 1 s, son spectre d'amplitude est calculé par calcule_DFT : une
sinusoïde d'amplitude A donne un pic de hauteur A, la composante continue
vaut la moyenne. Les pics tombent exactement sur la grille des fréquences
(pas fe/N = 1 Hz) parce que f est un multiple de fe/N ; sinon, ils
s'étalent sur plusieurs points (fuite spectrale), et tpllg.fft.spectre fait
mieux.
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from tpllg.fft import calcule_DFT
from tpllg.harmoniques import Signal

# le signal échantillonné
fe = 4_000
tau = 1 / fe
N = 4_000  # N = fe pour faciliter l'affichage (df = 1 Hz)
f = 200  # Hz, multiple de fe/N : sinon les pics s'étalent

amplitudes = [1.59, 2.5, 1.06, 0, 0.212, 0, 0.0909]  # (ua), la composante continue en premier
phases = [0, pi / 2, 0, 0, 0, 0, 0]
signal = Signal.depuis_coefficients(f, amplitudes, phases)

# endpoint=False pour exclure le dernier point et avoir bien tau entre chaque
t = np.linspace(0, N * tau, N, endpoint=False)
s = signal(t)
t_fin = np.linspace(0, N * tau, 10 * N, endpoint=False)
s_fin = signal(t_fin)

# le spectre
freq, spectre = calcule_DFT(t, s)
print("harmoniques retrouvées :", np.round(spectre[(freq % f == 0) & (freq <= 6 * f)], 3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_xlabel("temps $t$ (s)")
ax1.set_ylabel("$s(t)$ (ua)")
ax1.set_xlim(0, 3 / f)
ax1.plot(t, s, "k+", linestyle="None", label="échantillons")
ax1.plot(t_fin, s_fin, "--", linewidth=0.5, color="gray", label="signal")
ax1.legend()

ax2.set_xlabel("fréquence $f$ (Hz)")
ax2.set_ylabel("amplitude (ua)")
ax2.set_xlim(-50, 8 * f)
ax2.plot(freq, spectre, "ko")
fig.tight_layout()
plt.show()
