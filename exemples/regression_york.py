"""
Une droite quand x et y sont tous deux incertains, d'incertitudes différentes
d'un point à l'autre : la régression de York (regression_york), qui remplace
scipy.odr, face au Monte-Carlo de la même droite (SerieLineaire) et à la
variance effective (curvefit). Voir doc/ajustement.md.

Les vingt mesures sont celles d'un script de D. Jurine qui comparait un
Monte-Carlo, des moindres carrés pondérés et scipy.odr. Son Monte-Carlo
ajustait chaque tirage sans pondération (np.polyfit) : il donnait une pente
de -1,64 ± 0,05 au lieu de -1,72 ± 0,04, parce qu'un point très incertain y
pèse autant qu'un point précis. Le script refait ce calcul pour le montrer.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit, formater, regression_york
from tpllg.montecarlo import SerieLineaire, fixer_graine

fixer_graine(0)

# fmt: off
x = np.array([0.6881, -0.9541, 0.2930, -0.1832, 0.7696, 0.0028, 0.4339, -0.1419, -0.4061, -0.5884,
              -0.6493, 0.5373, -0.4070, 0.0918, -0.6795, 0.3982, -0.2522, 0.6437, 0.7194, 0.0677])
y = np.array([-1.5548, 1.2249, -1.0927, -0.0302, -1.5204, -0.4743, -1.1590, -0.1612, 0.2074, 0.6150,
              0.5899, -0.9864, 0.2819, -0.5091, 0.7859, -1.1421, -0.0291, -1.6401, -1.6432, -0.7269])
u_x = np.array([0.1, 0.0231, 0.0607, 0.0486, 0.0891, 0.0762, 0.0456, 0.0019, 0.0821, 0.0445,
                0.0615, 0.0792, 0.0922, 0.0738, 0.0176, 0.0406, 0.0935, 0.0917, 0.0410, 0.0894])
u_y = np.array([0.0058, 0.0353, 0.0813, 0.0010, 0.0139, 0.0203, 0.0199, 0.0604, 0.0272, 0.0199,
                0.0015, 0.0747, 0.0445, 0.0932, 0.0466, 0.0419, 0.0846, 0.0525, 0.0203, 0.0672])
# fmt: on


def droite(x, a, b):
    return a * x + b


york = regression_york(x, u_x, y, u_y)
a_mc, b_mc = SerieLineaire(x, u_x, y, u_y).ajuster()
variance_effective = curvefit(droite, x, y, [-1, 0], u_y, u_x, lambda x, a, b: a, verbose=False)
# le Monte-Carlo sans pondération : chaque tirage ajusté par np.polyfit, tous les points à égalité
rng = np.random.default_rng(1)
tirages = np.array([np.polyfit(rng.normal(x, u_x), rng.normal(y, u_y), 1) for _ in range(20000)])

resultats = (
    ("régression de York", york.pfit, york.err),
    ("Monte-Carlo, York sur chaque tirage", (a_mc.val, b_mc.val), (a_mc.u, b_mc.u)),
    ("variance effective (curvefit)", variance_effective.pfit, variance_effective.err),
    ("Monte-Carlo sans pondération", tirages.mean(axis=0), tirages.std(axis=0, ddof=1)),
)
for nom, (a, b), (u_a, u_b) in resultats:
    print(f"{nom:36s} a = {formater(a, u_a):16s} b = {formater(b, u_b)}")
print(f"chi2 réduit de York : {york.chi2:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.errorbar(x, y, xerr=u_x, yerr=u_y, fmt="o", markersize=3, capsize=2, color="k", label="les mesures")
x_fin = np.linspace(-1, 0.8, 50)
for (nom, (a, b), _), trait in zip(resultats, ("-", ":", "--", "-")):
    ax1.plot(x_fin, droite(x_fin, a, b), trait, label=nom)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend(fontsize=8)
ax1.set_title("vingt points, incertitudes variables sur x et sur y", fontsize=10)
ax2.hist(a_mc.tirage, bins=200, density=True, alpha=0.6, label="pentes, York sur chaque tirage")
ax2.hist(tirages[:, 0], bins=200, density=True, alpha=0.6, label="pentes, polyfit sur chaque tirage")
ax2.axvline(york.pfit[0], color="k", label="regression_york : " + formater(york.pfit[0], york.err[0]))
ax2.set_xlabel("pente a")
ax2.legend(fontsize=8)
ax2.set_title("ce que la pondération change", fontsize=10)
fig.tight_layout()
plt.savefig("regression_york.pdf")

plt.show()
