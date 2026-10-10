"""
Une droite ajustée sans incertitudes, avec celles de y, avec celles de x et
de y (tpllg.ajustement.curvefit), quand le bruit est constant puis quand il
ne l'est pas. Voir doc/ajustement.md.

curvefit(fonction, x, y, p0, u_y=…, u_x=…, function_derivate=…) rend les
paramètres, leurs incertitudes-types et le chi2 réduit :

- sans incertitudes : moindres carrés ordinaires, les incertitudes rendues
  sont tirées de la dispersion des résidus ;
- incertitudes sur y : moindres carrés pondérés ;
- incertitudes sur x et y : méthode de la variance effective (J. Orear,
  Am. J. Phys. 50, 912, 1982), qui reporte sigma_x sur y par la dérivée du
  modèle.

Pour une droite, des incertitudes constantes ne changent pas les paramètres
ajustés, seulement leurs incertitudes et le chi2 ; des incertitudes variables
changent aussi les paramètres. Deux figures : bruit constant, bruit variable.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit


def modele(x, a, b):
    return a * x + b


def modele_derivee(x, a, b):
    return a


# Les mesures simulées : y = 2x - 1, bruitées en x et en y
rng = np.random.default_rng(3)
N = 10
x = np.linspace(0.1, 2, N)
y = modele(x, 2, -1)
sigma_x = 0.05 * np.ones(N)  # bruit en x constant…
sigma_x2 = sigma_x * (x + 1)  # …puis croissant
sigma_y = 0.15 * np.ones(N)  # bruit en y constant…
sigma_y2 = sigma_y * (x**2 + 1)  # …puis croissant (pas proportionnel à x : les méthodes se confondraient)
x_bruite, x_bruite2 = x + sigma_x * rng.standard_normal(N), x + sigma_x2 * rng.standard_normal(N)
y_bruite, y_bruite2 = y + sigma_y * rng.standard_normal(N), y + sigma_y2 * rng.standard_normal(N)

x_trace = np.linspace(0, 2.2, 100)
cas = [
    ("bruit constant", "a et b ne dépendent pas des incertitudes, leurs incertitudes si",
     x_bruite, y_bruite, sigma_x, sigma_y),
    ("bruit variable", "a et b dépendent des incertitudes",
     x_bruite2, y_bruite2, sigma_x2, sigma_y2),
]  # fmt: skip

for nom_cas, titre, xm, ym, sx, sy in cas:
    plt.figure(figsize=(8, 6))
    plt.title(f"{nom_cas} en x et y : {titre}", fontsize=9)
    plt.errorbar(xm, ym, xerr=sx, yerr=sy, fmt="o", capthick=2)
    ajustements = [
        ("sans incertitudes", curvefit(modele, xm, ym, p0=[1, 0])),
        ("incertitudes sur y", curvefit(modele, xm, ym, p0=[1, 0], u_y=sy)),
        (
            "incertitudes sur x et y",
            curvefit(modele, xm, ym, [1, 0], u_y=sy, u_x=sx, function_derivate=modele_derivee),
        ),
    ]
    for nom, (pfit, err, chi2) in ajustements:
        etiquette = (
            f"a = {pfit[0]:.3f} ± {err[0]:.3f}, b = {pfit[1]:.3f} ± {err[1]:.3f}, chi2 réduit = {chi2:.1f}"
        )
        plt.plot(x_trace, modele(x_trace, *pfit), label=f"{nom}\n{etiquette}")
        print(f"{nom_cas}, {nom} : {etiquette}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=8)
    plt.tight_layout()

plt.show()
