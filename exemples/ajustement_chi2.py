# -*- coding: utf-8 -*-
"""
Ce qu'un ajustement minimise, et comment : le chi2, somme des carrés des
écarts entre mesures et modèle rapportés aux incertitudes ; sa vallée dans
le plan des paramètres, dont la largeur donne les incertitudes ; le chemin
que curve_fit y suit depuis les valeurs de départ. Voir doc/ajustement.md.

Deux modèles : une droite, dont le chi2 est une cuvette parabolique que
curve_fit descend en un pas, et une exponentielle, dont la vallée est courbe
et se descend en plusieurs.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tpllg.ajustement import curvefit, formater


def droite(x, a, b):
    return a * x + b


def exponentielle(x, a, tau):
    return a * np.exp(-x / tau)


def chi2(modele, x, y, sigma, *p):
    """La somme des carrés des écarts au modèle, chacun rapporté à l'incertitude du point."""
    return np.sum(((y - modele(x, *p)) / sigma) ** 2)


def carte_chi2(modele, x, y, sigma, p1, p2):
    """Le chi2 sur une grille des deux paramètres : un tableau (len(p2), len(p1))."""
    return np.array([[chi2(modele, x, y, sigma, u, v) for u in p1] for v in p2])


def chemin(modele, jacobienne, x, y, sigma, p0):
    """Les valeurs des paramètres à chaque itération de curve_fit. On lui donne
    la jacobienne du modèle, ses dérivées par rapport aux paramètres : il
    l'appelle une fois par itération, au point où il en est, et on le note.
    Le dernier appel se fait au minimum, pour constater qu'il n'y a plus à descendre."""
    etapes = []

    def jac(x, *p):
        etapes.append(p)
        return jacobienne(x, *p)

    curve_fit(modele, x, y, p0=p0, sigma=sigma * np.ones_like(x), absolute_sigma=True, jac=jac)
    return np.array(etapes)


# la droite : dix points, 0,15 d'incertitude sur y
x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])
y = np.array([-0.85, -0.42, 0.11, 0.35, 0.84, 1.15, 1.70, 1.95, 2.36, 2.85])
sigma_y = 0.15
p0 = [1, 0]
pfit, err, chi2_reduit = curvefit(droite, x, y, p0, datayerrors=sigma_y, verbose=False)
chi2_min = chi2(droite, x, y, sigma_y, *pfit)
print("droite : a =", formater(pfit[0], err[0]), " b =", formater(pfit[1], err[1]))
print(
    "chi2 = %.1f aux valeurs de départ, %.2f au minimum, soit %.2f par degré de liberté"
    % (chi2(droite, x, y, sigma_y, *p0), chi2_min, chi2_reduit)
)

# 1. ce que le chi2 additionne : les écarts, aux valeurs de départ puis au minimum
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
x_fin = np.linspace(0, 2, 50)
for colonne, (titre, p) in enumerate(
    (("aux valeurs de départ, a = 1, b = 0", p0), ("au minimum, a = %.3f, b = %.3f" % tuple(pfit), pfit))
):
    haut, bas = axes[0, colonne], axes[1, colonne]
    ecarts = (y - droite(x, *p)) / sigma_y
    haut.plot(x_fin, droite(x_fin, *p), color="tab:orange", label="le modèle")
    haut.vlines(x, droite(x, *p), y, color="tab:red", linewidth=2, label="les écarts")
    haut.errorbar(x, y, yerr=sigma_y, fmt="o", color="tab:blue", markersize=4, capsize=3, label="les mesures")
    haut.set_title("%s : chi2 = %.1f" % (titre, np.sum(ecarts**2)), fontsize=10)
    haut.set_ylabel("y")
    haut.legend(fontsize=8, loc="upper left")
    bas.bar(x, ecarts**2, width=0.12, color="tab:red")
    bas.set_xlabel("x")
    bas.set_ylabel("(écart / sigma)²")
fig.tight_layout()
plt.savefig("ajustement_chi2_ecarts.pdf")

# 2. la vallée du chi2 dans le plan (a, b), autour du minimum
a = np.linspace(pfit[0] - 4.5 * err[0], pfit[0] + 4.5 * err[0], 201)
b = np.linspace(pfit[1] - 4.5 * err[1], pfit[1] + 4.5 * err[1], 201)
carte = carte_chi2(droite, x, y, sigma_y, a, b)
fig, ax = plt.subplots(figsize=(7, 5.5))
fond = ax.contourf(a, b, carte - chi2_min, levels=np.linspace(0, 20, 21), cmap="Blues_r", extend="max")
fig.colorbar(fond, ax=ax, label="chi2 - chi2 au minimum")
niveaux = ax.contour(a, b, carte - chi2_min, levels=[1, 4, 9], colors=["tab:red", "tab:orange", "gold"])
ax.clabel(niveaux, fmt={1: "+1", 4: "+4", 9: "+9"}, fontsize=9)
ax.plot(pfit[0], pfit[1], "k+", markersize=12, label="le minimum")
# les projections du contour chi2 min + 1 sur les axes sont les incertitudes-types
for valeur in (pfit[0] - err[0], pfit[0] + err[0]):
    ax.axvline(valeur, color="tab:red", linestyle="--", linewidth=1)
for valeur in (pfit[1] - err[1], pfit[1] + err[1]):
    ax.axhline(valeur, color="tab:red", linestyle="--", linewidth=1)
ax.plot([], [], color="tab:red", linestyle="--", linewidth=1, label="a ± %.3f et b ± %.3f" % tuple(err))
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
plt.savefig("ajustement_chi2_vallee.pdf")

# 3. le chemin des itérations : un pas pour la droite, plusieurs pour l'exponentielle
xe = np.linspace(0, 5, 20)
rng = np.random.RandomState(4)
ye = 2 * np.exp(-xe / 1.5) + rng.normal(0, 0.02, xe.size)
sigma_e = 0.02
cas = (
    (
        "une droite : a x + b",
        droite,
        lambda x, a, b: np.column_stack([x, np.ones_like(x)]),
        x,
        y,
        sigma_y,
        np.linspace(0.7, 2.6, 121),
        np.linspace(-1.7, 0.4, 121),
        ([1, 0], [2.4, 0.2]),
        ("a", "b"),
    ),
    (
        "une exponentielle : a exp(-x/tau)",
        exponentielle,
        lambda x, a, tau: np.column_stack([np.exp(-x / tau), a * x / tau**2 * np.exp(-x / tau)]),
        xe,
        ye,
        sigma_e,
        np.linspace(0.3, 3.3, 121),
        np.linspace(0.2, 4.7, 121),
        ([1, 1], [0.5, 4], [3, 0.3]),
        ("a", "tau"),
    ),
)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (titre, modele, jacobienne, xd, yd, sigma, p1, p2, departs, noms) in zip(axes, cas):
    carte = carte_chi2(modele, xd, yd, sigma, p1, p2)
    ax.contourf(p1, p2, np.log10(carte), levels=20, cmap="Blues_r")
    ax.contour(p1, p2, np.log10(carte), levels=12, colors="white", linewidths=0.4)
    for depart, couleur in zip(departs, ("tab:red", "tab:orange", "gold")):
        etapes = chemin(modele, jacobienne, xd, yd, sigma, depart)
        ax.plot(
            etapes[:, 0],
            etapes[:, 1],
            "o-",
            color=couleur,
            markersize=4,
            label="depuis (%g, %g) : %d pas" % (depart[0], depart[1], len(etapes) - 1),
        )
        print(
            "%s, depuis %s : %d pas, chi2 = %s"
            % (
                titre,
                depart,
                len(etapes) - 1,
                " → ".join("%.1f" % chi2(modele, xd, yd, sigma, *p) for p in etapes),
            )
        )
    ax.plot(etapes[-1, 0], etapes[-1, 1], "k+", markersize=12, label="le minimum")
    ax.set_title(titre, fontsize=10)
    ax.set_xlabel(noms[0])
    ax.set_ylabel(noms[1])
    ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig("ajustement_chi2_chemin.pdf")

plt.show()
