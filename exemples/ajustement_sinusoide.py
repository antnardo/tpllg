# -*- coding: utf-8 -*-
"""
Ajuster une sinusoïde, amplitude, fréquence et phase : le cas où les valeurs
de départ comptent le plus. Le chi2 en fonction de la fréquence est un puits
étroit — sa demi-largeur est l'inverse de la durée du signal — au milieu d'un
plateau à peine ondulé, dont chaque creux est un minimum local ; il faut
partir dans le puits, donc de la fréquence du pic de la FFT. Voir
doc/ajustement.md.

SIMULATION = True fabrique l'acquisition telle que la centrale la rendrait ;
False acquiert pour de bon.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.acquisition import acquerir
from tpllg.ajustement import curvefit, resume_parametres
from tpllg.signaux import frequence_pic

SIMULATION = True
VOIE, CALIBRE = 0, 5
te, N = 1e-4, 2000  # 0,2 s à 10 kHz
BRUIT = 0.005  # l'incertitude-type sur une tension, en volts
NOMS, UNITES = ("A", "f", "phi", "offset"), ("V", "Hz", "rad", "V")


def acquisition_simulee(te, N, A=1.5, f=52.3, phi=0.7, offset=0.2, bruit=BRUIT, graine=3):
    """Une sinusoïde bruitée, telle que la centrale la rendrait : deux tableaux (1, N)."""
    rng = np.random.RandomState(graine)
    t = np.arange(N) * te
    return np.array([t]), np.array([A * np.sin(2 * np.pi * f * t + phi) + offset + rng.normal(0, bruit, N)])


def sinus(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset


def forme_canonique(p):
    """La même sinusoïde écrite avec une amplitude positive et une phase entre -pi et pi :
    l'ajustement rend aussi bien A < 0 avec une phase décalée de pi."""
    A, f, phi, offset = p
    if A < 0:
        A, phi = -A, phi + np.pi
    return np.array([A, f, (phi + np.pi) % (2 * np.pi) - np.pi, offset])


def chi2_a_frequence_fixee(t, u, f, sigma):
    """Le chi2 réduit du meilleur ajustement à la fréquence f : à fréquence fixée, le
    modèle est linéaire en a sin + b cos + offset, et le minimum se calcule directement."""
    base = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)])
    coefs = np.linalg.lstsq(base, u, rcond=None)[0]
    return np.sum(((u - base.dot(coefs)) / sigma) ** 2) / (t.size - 4)


# 1. l'acquisition
if SIMULATION:
    temps, tensions = acquisition_simulee(te, N)
else:
    temps, tensions = acquerir([VOIE], CALIBRE, te, N)
t, u = temps[0], tensions[0]

# 2. les valeurs de départ : l'amplitude par l'écart-type, la fréquence par le pic de la FFT
p0 = [np.sqrt(2) * u.std(), frequence_pic(u, te), 0, u.mean()]
print("valeurs de départ :", resume_parametres(NOMS, p0, unites=UNITES).replace("\n", "   "))
pfit, err, chi2 = curvefit(sinus, t, u, p0, datayerrors=BRUIT, verbose=False)
print("tel que l'ajustement le rend :", resume_parametres(NOMS, pfit, unites=UNITES).replace("\n", "   "))
pfit = forme_canonique(pfit)
print(resume_parametres(NOMS, pfit, err, unites=UNITES))
print("chi2 réduit = %.2f" % chi2)

# 3. le même ajustement en partant d'une fréquence fausse de 12 Hz
p0_faux = [p0[0], p0[1] + 12, 0, p0[3]]
pfaux, efaux, chi2_faux = curvefit(sinus, t, u, p0_faux, datayerrors=BRUIT, verbose=False)
pfaux = forme_canonique(pfaux)
print(
    "en partant de f = %g Hz :" % p0_faux[1],
    resume_parametres(NOMS, pfaux, unites=UNITES).replace("\n", "   "),
    "  chi2 réduit = %.0f" % chi2_faux,
)

# 4. la figure : le signal et les deux ajustements, puis le chi2 en fonction de la fréquence
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(1e3 * t, u, ".", markersize=2, color="tab:blue", label="l'acquisition")
axes[0].plot(
    1e3 * t, sinus(t, *pfit), color="tab:orange", label="parti de %g Hz : f = %.3f Hz" % (p0[1], pfit[1])
)
axes[0].plot(
    1e3 * t, sinus(t, *pfaux), color="tab:red", label="parti de %g Hz : f = %.2f Hz" % (p0_faux[1], pfaux[1])
)
axes[0].set_xlabel("t (ms)")
axes[0].set_ylabel("u (V)")
axes[0].legend(fontsize=8, loc="upper right")
frequences = np.arange(20, 90, 0.02)
profil = np.array([chi2_a_frequence_fixee(t, u, f, BRUIT) for f in frequences])
axes[1].semilogy(frequences, profil, color="tab:blue", label="le meilleur chi2 réduit à fréquence fixée")
for depart, arrivee, valeur, couleur in (
    (p0[1], pfit[1], chi2, "tab:orange"),
    (p0_faux[1], pfaux[1], chi2_faux, "tab:red"),
):
    axes[1].plot(depart, chi2_a_frequence_fixee(t, u, depart, BRUIT), "o", color=couleur)
    axes[1].plot(
        arrivee,
        valeur,
        "*",
        color=couleur,
        markersize=12,
        label="de %g Hz à %.2f Hz : chi2 réduit = %.3g" % (depart, arrivee, valeur),
    )
axes[1].set_xlabel("f (Hz)")
axes[1].set_ylabel("chi2 réduit")
axes[1].legend(fontsize=8, loc="lower right")
# en médaillon, le plateau à droite du puits, en échelle linéaire : il ondule
medaillon = axes[1].inset_axes([0.1, 0.12, 0.36, 0.5])
plateau = (frequences > 56.5) & (frequences < 78)
medaillon.plot(frequences[plateau], profil[plateau], color="tab:blue")
medaillon.plot(p0_faux[1], chi2_a_frequence_fixee(t, u, p0_faux[1], BRUIT), "o", color="tab:red")
medaillon.plot(pfaux[1], chi2_faux, "*", color="tab:red", markersize=12)
medaillon.set_title("le plateau, de 57 à 78 Hz", fontsize=8)
medaillon.tick_params(labelsize=7)
fig.tight_layout()
plt.savefig("ajustement_sinusoide.pdf")

plt.show()
