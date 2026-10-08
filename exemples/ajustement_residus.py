# -*- coding: utf-8 -*-
"""
Lire les résidus d'un ajustement : répartis au hasard autour de zéro, de
l'ordre des incertitudes de mesure, quand le modèle est le bon ; en pente
quand il lui manque quelque chose. Voir doc/ajustement.md.

Un passe-bande du second ordre ajusté sur vingt-trois mesures, puis sur des
mesures simulées où l'amplificateur coupe à 50 kHz, ce que le modèle ignore.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curve_fit_complex, residus_complexes, resume_parametres


def passe_bande(f, H0, f0, Q):
    return H0 / (1 + 1j * Q * (f / f0 - f0 / f))


f = np.array(
    [
        200,
        300,
        500,
        700,
        1000,
        1300,
        1500,
        1700,
        1800,
        1900,
        1950,
        2000,
        2050,
        2100,
        2200,
        2400,
        2700,
        3300,
        5000,
        7000,
        10000,
        15000,
        20000.0,
    ]
)
H = np.array(
    [
        0.09,
        0.12,
        0.21,
        0.30,
        0.53,
        0.88,
        1.33,
        2.18,
        3.06,
        4.21,
        4.62,
        5.13,
        4.85,
        4.42,
        3.16,
        1.95,
        1.24,
        0.72,
        0.39,
        0.24,
        0.16,
        0.11,
        0.08,
    ]
)
phi = np.radians(
    [
        -90,
        -94,
        -95,
        -92,
        -97,
        -103,
        -106,
        -114,
        -122,
        -152,
        -166,
        176,
        154,
        144,
        126,
        117,
        104,
        94,
        96,
        92,
        87,
        89,
        89,
    ]
)
NOMS, UNITES = ("H0", "f0", "Q"), ("", "Hz", "")

# les mêmes fréquences sur un montage dont l'amplificateur coupe à 50 kHz : un passe-bas
# du premier ordre en plus, que le modèle ne contient pas ; 3 % et 3° de bruit de mesure
rng = np.random.RandomState(5)
reel = passe_bande(f, -5, 1994.6, 6.27) / (1 + 1j * f / 50000)
H_simule = abs(reel) * (1 + rng.normal(0, 0.03, f.size))
phi_simule = np.angle(reel) + rng.normal(0, np.radians(3), f.size)

cas = (
    ("le modèle décrit les mesures", H, phi),
    ("il manque au modèle la coupure de l'amplificateur", H_simule, phi_simule),
)
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey="row")
for colonne, (titre, module, phase) in enumerate(cas):
    pfit, err, chi2 = curve_fit_complex(passe_bande, f, module, phase, p0=[-5, 2000, 6], verbose=False)
    res_H, res_phi = residus_complexes(passe_bande, f, module, phase, pfit)
    print("%s :" % titre, resume_parametres(NOMS, pfit, err, UNITES).replace("\n", "   "))
    print("  résidus du module, en %  :", " ".join("%+.0f" % r for r in 100 * res_H))
    print("  résidus de la phase, en ° :", " ".join("%+.0f" % r for r in res_phi))
    haut, bas = axes[0, colonne], axes[1, colonne]
    for ax, residus, demi in ((haut, 100 * res_H, 3), (bas, res_phi, 3)):
        ax.axhspan(-demi, demi, color="tab:orange", alpha=0.25, label="l'incertitude de mesure")
        ax.axhline(0, color="tab:orange")
        ax.semilogx(f, residus, "o", color="tab:blue", markersize=4)
    haut.set_title(titre, fontsize=10)
    haut.legend(fontsize=8, loc="lower left")
    bas.set_xlabel("f (Hz)")
axes[0, 0].set_ylabel("résidu du module (%)")
axes[1, 0].set_ylabel("résidu de la phase (°)")
fig.tight_layout()
plt.savefig("ajustement_residus.pdf")

plt.show()
