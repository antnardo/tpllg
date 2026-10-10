"""
La fréquence d'un défilement de franges : par comptage, par le pic de la
DFT (calcule_DFT), par le spectre fenêtré (spectre). Voir doc/spectres.md.

Quand la différence de marche d'un interféromètre varie à vitesse constante
(un miroir de Michelson translaté, par exemple), l'intensité reçue par un
détecteur est sinusoïdale, I(t) = I0 (1 + cos(omega t)). Deux façons de
mesurer omega sur le signal échantillonné :

- compter les passages montants par la valeur moyenne I0 sur la durée
  d'acquisition ;
- chercher le pic du spectre d'amplitude (tpllg.fft.calcule_DFT), dont la
  résolution vaut 1/durée : la fenêtre de Blackman avec ajout de zéros
  (tpllg.fft.spectre), appliquée au signal privé de sa moyenne, affine la
  position du pic.

La pulsation du signal est une valeur d'illustration.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.fft import calcule_DFT, spectre

I0 = 1
OMEGA = 61_250  # rad/s, valeur d'illustration
DUREE = 5e-3  # s
FE = 200e3  # Hz, fréquence d'échantillonnage


def intensite(t):
    return I0 * (1 + np.cos(OMEGA * t))


t = np.arange(0, DUREE, 1 / FE)
signal = intensite(t)

# Comptage des franges
dessus = signal > I0
passages = t[1:][dessus[1:] & ~dessus[:-1]]
f_comptage = (len(passages) - 1) / (passages[-1] - passages[0])

# Transformée de Fourier, sans puis avec fenêtre et ajout de zéros
freq, amplitudes = calcule_DFT(t, signal)
f_dft = freq[1:][np.argmax(amplitudes[1:])]  # hors composante continue
# moyenne retirée : sinon le lobe de la composante continue, élargi par la fenêtre,
# masque les basses fréquences
freq_fine, amplitudes_fines = spectre(t, signal - signal.mean())
moitie = freq_fine < FE / 2
f_fine = freq_fine[moitie][np.argmax(amplitudes_fines[moitie])]

f_vraie = OMEGA / (2 * np.pi)
print(f"fréquence vraie : {f_vraie:.1f} Hz")
print(f"comptage de {len(passages)} franges : {f_comptage:.1f} Hz")
print(f"pic de la DFT (résolution {1 / DUREE:.0f} Hz) : {f_dft:.1f} Hz")
print(f"pic avec fenêtre et zéros : {f_fine:.1f} Hz")

fig, (ax1, ax2) = plt.subplots(1, 2, num="Défilement de franges", figsize=(10, 4))
ax1.plot(t * 1e3, signal)
ax1.plot(passages * 1e3, [I0] * len(passages), "o")
ax1.set_xlim(0, 1)
ax1.set_xlabel("$t$ (ms)")
ax1.set_ylabel("$I$")
ax1.set_title("Intensité (première milliseconde)")
ax2.plot(freq, amplitudes, "o-", label="DFT")
ax2.plot(freq_fine, amplitudes_fines, label="Blackman et zéros")
ax2.axvline(f_vraie, color="k", linestyle="--", label="fréquence vraie")
ax2.set_xlim(f_vraie - 2e3, f_vraie + 2e3)
ax2.set_xlabel("fréquence (Hz)")
ax2.set_ylabel("amplitude")
ax2.legend()
fig.tight_layout()
plt.show()
