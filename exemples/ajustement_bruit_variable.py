"""
Une droite sur des mesures dont l'incertitude change d'un point à l'autre :
sans incertitudes, avec celles de y, avec celles de x et de y (variance
effective), et par la régression de York. Voir doc/ajustement.md.

Avec des incertitudes constantes, ces ajustements donnent la même droite
(exemples/ajustement_droite.py) ; avec des incertitudes variables, les points
précis pèsent plus et la droite bouge. Elle ne bougerait pas si les
incertitudes sur y étaient proportionnelles à celles sur x : ici sigma_y
croît comme x² + 1 et sigma_x comme x + 1.

La régression de York trouve le minimum exact de la somme que la variance
effective cherche, sum (y - a x - b)²/(sigma_y² + a² sigma_x²) ; la variance
effective l'approche en itérant à poids figés et s'arrête un peu à côté (2 %
sur la pente ici), dans les incertitudes.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit, regression_york, resume_parametres


def modele(x, a, b):
    return a * x + b


def modele_derivee(x, a, b):
    return a


rng = np.random.default_rng(0)  # les mêmes mesures d'une exécution à l'autre
x_vrai = np.linspace(0.1, 2, 10)
sigma_x = 0.05 * (x_vrai + 1)
sigma_y = 0.15 * (x_vrai**2 + 1)
x = x_vrai + rng.normal(0, sigma_x)
y = modele(x_vrai, 2, -1) + rng.normal(0, sigma_y)

ajustements = (
    ("sans incertitudes", curvefit(modele, x, y, p0=[1, 0], verbose=False), "-"),
    ("incertitudes sur y", curvefit(modele, x, y, p0=[1, 0], datayerrors=sigma_y, verbose=False), "-"),
    (
        "incertitudes sur x et y",
        curvefit(modele, x, y, [1, 0], sigma_y, sigma_x, modele_derivee, verbose=False),
        "-",
    ),
    ("régression de York", regression_york(x, sigma_x, y, sigma_y), ":"),
)

x_fin = np.linspace(0, 2.1, 100)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, fmt="o", capsize=2, color="k", label="les mesures")
for nom, (pfit, err, chi2), trait in ajustements:
    resume = resume_parametres(("a", "b"), pfit, err).replace("\n", "   ")
    suite = f"chi2 réduit = {chi2:.2f}" if nom != "sans incertitudes" else "chi2 réduit : sans objet"
    print(f"{nom:24s} {resume}   {suite}")
    ax.plot(x_fin, modele(x_fin, *pfit), trait, linewidth=2, label=f"{nom} : {resume}")
ax.plot(x_fin, modele(x_fin, 2, -1), color="gray", linewidth=0.8, label="la droite vraie, a = 2 et b = -1")
ax.set_title("bruit variable : sigma_x et sigma_y croissent avec x", fontsize=10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
plt.savefig("ajustement_bruit_variable.pdf")

plt.show()
