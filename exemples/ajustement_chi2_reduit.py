# -*- coding: utf-8 -*-
"""
Lire le chi2 réduit : il vaut 1 à quelques dixièmes près quand le modèle est
bon et les incertitudes justes, bien plus quand elles sont sous-estimées ou
le modèle faux, bien moins quand elles sont surestimées. Voir
doc/ajustement.md.

Vingt points d'une exponentielle décroissante bruitée à 0,02 V : ajustée
avec trois incertitudes annoncées, puis par une droite ; et la loi du chi2
réduit sur deux mille jeux de mesures.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from tpllg.ajustement import curvefit, resume_parametres


def exponentielle(x, a, tau):
    return a * np.exp(-x / tau)


def droite(x, a, b):
    return a * x + b


BRUIT = 0.02  # le vrai bruit des mesures, en volts
x = np.linspace(0, 5, 20)
rng = np.random.RandomState(4)
y = 2 * np.exp(-x / 1.5) + rng.normal(0, BRUIT, x.size)
x_fin = np.linspace(0, 5, 200)

# 1. le bon modèle, trois incertitudes annoncées : la juste, une trop petite, une trop grande
fig, axes = plt.subplots(
    2, 3, figsize=(13, 6), sharex=True, sharey="row", gridspec_kw={"height_ratios": [3, 2]}
)
for colonne, sigma in enumerate((0.02, 0.005, 0.08)):
    pfit, err, chi2 = curvefit(exponentielle, x, y, p0=[1, 1], datayerrors=sigma, verbose=False)
    print(
        "sigma = %g" % sigma,
        resume_parametres(("a", "tau"), pfit, err).replace("\n", "   "),
        "  chi2 réduit = %.2f" % chi2,
    )
    haut, bas = axes[0, colonne], axes[1, colonne]
    haut.errorbar(x, y, yerr=sigma, fmt="o", markersize=3, capsize=2, label="les mesures")
    haut.plot(x_fin, exponentielle(x_fin, *pfit), color="tab:orange", label="l'ajustement")
    haut.set_title("sigma annoncé = %g V : chi2 réduit = %.2f" % (sigma, chi2), fontsize=10)
    haut.legend(fontsize=8)
    # les résidus, avec la même barre : c'est là qu'on voit si elle est à la bonne taille
    bas.axhline(0, color="tab:orange")
    bas.errorbar(x, y - exponentielle(x, *pfit), yerr=sigma, fmt="o", markersize=3, capsize=2)
    bas.set_xlabel("x")
axes[0, 0].set_ylabel("y (V)")
axes[1, 0].set_ylabel("y - modèle (V)")
fig.tight_layout()
plt.savefig("ajustement_chi2_reduit_sigma.pdf")

# 2. les mêmes points, la juste incertitude, un modèle faux : une droite
pfit, err, chi2 = curvefit(droite, x, y, p0=[-1, 2], datayerrors=BRUIT, verbose=False)
print(
    "une droite :",
    resume_parametres(("a", "b"), pfit, err).replace("\n", "   "),
    "  chi2 réduit = %.1f" % chi2,
)
fig, (haut, bas) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
haut.errorbar(x, y, yerr=BRUIT, fmt="o", markersize=3, capsize=2, label="les mesures")
haut.plot(x_fin, droite(x_fin, *pfit), color="tab:orange", label="la droite ajustée")
haut.set_title("un modèle faux, des incertitudes justes : chi2 réduit = %.1f" % chi2, fontsize=10)
haut.set_ylabel("y (V)")
haut.legend(fontsize=8)
bas.axhline(0, color="tab:orange")
bas.errorbar(x, y - droite(x, *pfit), yerr=BRUIT, fmt="o", markersize=3, capsize=2)
bas.set_xlabel("x")
bas.set_ylabel("y - modèle (V)")
fig.tight_layout()
plt.savefig("ajustement_chi2_reduit_modele.pdf")

# 3. la loi du chi2 réduit : deux mille jeux de vingt points, bon modèle, juste incertitude
rng = np.random.RandomState(0)
tirages = np.zeros(2000)
for k in range(tirages.size):
    y_k = 2 * np.exp(-x / 1.5) + rng.normal(0, BRUIT, x.size)
    tirages[k] = curvefit(exponentielle, x, y_k, p0=[1, 1], datayerrors=BRUIT, verbose=False)[2]
libertes = x.size - 2  # vingt points, deux paramètres
largeur = np.sqrt(2 / libertes)
print(
    "chi2 réduit sur %d jeux : moyenne %.3f, écart-type %.3f ; attendu 1 et sqrt(2/%d) = %.3f"
    % (tirages.size, tirages.mean(), tirages.std(ddof=1), libertes, largeur)
)
print(
    "part des jeux entre 1 - %.2f et 1 + %.2f : %.0f %%"
    % (largeur, largeur, 100 * np.mean(abs(tirages - 1) < largeur))
)
fig, ax = plt.subplots(figsize=(7, 4.5))
c = np.linspace(0, 2.6, 300)
ax.hist(
    tirages,
    bins=np.linspace(0, 2.6, 53),
    density=True,
    color="tab:blue",
    alpha=0.75,
    label="%d jeux de mesures simulés" % tirages.size,
)
ax.plot(
    c,
    libertes * stats.chi2.pdf(libertes * c, libertes),
    color="navy",
    label="la loi du chi2 à %d degrés de liberté" % libertes,
)
ax.axvspan(
    1 - largeur,
    1 + largeur,
    color="tab:orange",
    alpha=0.3,
    zorder=0,
    label="1 ± sqrt(2/(N - p)) = 1 ± %.2f" % largeur,
)
ax.axvline(1, color="darkred", label="1")
ax.set_xlabel("chi2 réduit")
ax.set_ylabel("densité")
ax.legend(fontsize=9)
fig.tight_layout()
plt.savefig("ajustement_chi2_reduit_loi.pdf")

plt.show()
