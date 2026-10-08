# -*- coding: utf-8 -*-
"""
Une droite sur dix points, ajustée de trois façons : sans incertitudes, avec
celles de y, avec celles de x et de y. Les trois donnent la même droite ; ce
qui change, c'est l'incertitude rendue sur les paramètres — donc la bande où
la droite peut passer — et le chi2 réduit. Voir doc/ajustement.md.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit, resume_parametres


def modele(x, a, b):
    return a * x + b


def modele_derivee(x, a, b):  # dérivée par rapport à x : constante, un nombre suffit
    return a


def bande(x, x_mesures, err):
    """L'incertitude-type sur la droite ajustée, en x, quand tous les points ont la
    même incertitude. La pente et la valeur de la droite au centre des mesures sont
    alors indépendantes : u(x)² = u_centre² + (x - centre)² u_a², et l'ordonnée à
    l'origine donne u_centre² = u_b² - centre² u_a²."""
    centre = np.mean(x_mesures)
    return np.sqrt(err[1] ** 2 - centre**2 * err[0] ** 2 + (x - centre) ** 2 * err[0] ** 2)


x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])
y = np.array([-0.85, -0.42, 0.11, 0.35, 0.84, 1.15, 1.70, 1.95, 2.36, 2.85])
sigma_x = 0.05  # la même pour tous les points ; un tableau, une par point, sinon
sigma_y = 0.15

cas = (
    ("sans incertitudes", {}, {}),
    ("incertitudes sur y", {"datayerrors": sigma_y}, {"yerr": sigma_y}),
    (
        "incertitudes sur x et y",
        {"datayerrors": sigma_y, "dataxerrors": sigma_x, "function_derivate": modele_derivee},
        {"xerr": sigma_x, "yerr": sigma_y},
    ),
)
x_fin = np.linspace(0, 2, 100)
fig, axes = plt.subplots(
    2, 3, figsize=(13, 6.5), sharex=True, sharey="row", gridspec_kw={"height_ratios": [3, 2]}
)
for colonne, (titre, incertitudes, barres) in enumerate(cas):
    pfit, err, chi2 = curvefit(modele, x, y, p0=[1, 0], verbose=False, **incertitudes)
    resume = resume_parametres(("a", "b"), pfit, err).replace("\n", "   ")
    # sans incertitudes fournies, le chi2 réduit rendu n'a pas de sens : on ne l'écrit pas
    suite = "chi2 réduit = %.2f" % chi2 if incertitudes else "chi2 réduit : sans objet"
    print("%-24s" % titre, resume, "  " + suite)
    haut, bas = axes[0, colonne], axes[1, colonne]
    haut.errorbar(x, y, fmt="o", markersize=4, capsize=2, label="les mesures", **barres)
    haut.plot(x_fin, modele(x_fin, *pfit), color="tab:orange", label="la droite ajustée")
    haut.set_title("%s\n%s\n%s" % (titre, resume, suite), fontsize=10)
    haut.legend(fontsize=8, loc="upper left")
    # la même chose une fois la droite retranchée : la bande devient visible
    u = bande(x_fin, x, err)
    bas.fill_between(
        x_fin,
        -u,
        u,
        color="tab:orange",
        alpha=0.35,
        label="où la droite peut passer, à ± une incertitude-type",
    )
    bas.axhline(0, color="tab:orange")
    bas.errorbar(x, y - modele(x, *pfit), fmt="o", markersize=4, capsize=2, **barres)
    bas.set_xlabel("x")
    bas.legend(fontsize=8, loc="upper left")
axes[0, 0].set_ylabel("y")
axes[1, 0].set_ylabel("y - (a x + b)")
axes[1, 0].set_ylim(-0.36, 0.36)
fig.tight_layout()
plt.savefig("ajustement_droite.pdf")

plt.show()
