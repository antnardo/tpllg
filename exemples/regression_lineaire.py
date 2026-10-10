"""
Régression linéaire : np.polyfit, scipy.optimize.curve_fit et
tpllg.ajustement.curvefit sur les mêmes points. Voir doc/ajustement.md.

Sur des données simulées y = 2x - 1 + bruit (incertitude-type sigma_y = 0,15),
trois façons d'ajuster une droite :

- numpy.polyfit : rapide, pour tout polynôme, mais sans droite imposée par
  l'origine ni poids sur les points ;
- scipy.optimize.curve_fit : n'importe quel modèle ; ses paramètres sont dans
  pfit, et leurs incertitudes-types sont les racines de la diagonale de pcov ;
- tpllg.ajustement.curvefit : prend les incertitudes (sur y, ou sur x et y),
  rend directement les incertitudes-types et le chi2 réduit, que
  tpllg.ajustement.formater met en forme.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tpllg.ajustement import curvefit, formater


def modele(x, a, b):
    return a * x + b


# Données expérimentales simulées
N = 10
x = np.linspace(0.1, 2, N)
sigma_y = 0.15 * np.ones(N)
y_exp = modele(x, 2, -1) + sigma_y * np.random.default_rng(1).standard_normal(N)

# 1. polyfit
a1, b1 = np.polyfit(x, y_exp, 1)
print(f"polyfit : a = {a1:.3f}, b = {b1:.3f}")

# 2. curve_fit, avec les incertitudes
pfit, pcov = curve_fit(modele, x, y_exp, p0=[1, 0], sigma=sigma_y, absolute_sigma=True)
u_a2, u_b2 = np.sqrt(np.diag(pcov))
print(f"curve_fit : a = {formater(pfit[0], u_a2)}, b = {formater(pfit[1], u_b2)}")

# 3. tpllg
(a3, b3), (u_a3, u_b3), chi2 = curvefit(modele, x, y_exp, p0=[1, 0], u_y=sigma_y)
print(f"tpllg : a = {formater(a3, u_a3)}, b = {formater(b3, u_b3)}, chi2 réduit = {chi2:.2f}")

plt.figure("Régression linéaire")
plt.errorbar(x, y_exp, yerr=sigma_y, fmt="o", capthick=2, label="mesures")
plt.plot(x, modele(x, a3, b3), label=f"a = {formater(a3, u_a3)}, b = {formater(b3, u_b3)}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
