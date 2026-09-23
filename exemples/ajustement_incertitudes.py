# -*- coding: utf-8 -*-
"""
Une droite ajustée de trois façons — sans incertitudes, avec celles de y,
avec celles de x et de y (variance effective) — sur des mesures à bruit
constant, puis à bruit variable d'un point à l'autre. Voir doc/ajustement.md.

Avec des incertitudes constantes, les trois ajustements donnent les mêmes
paramètres et ne diffèrent que par les incertitudes rendues ; avec des
incertitudes variables, les paramètres bougent aussi.
"""
from tpllg.ajustement import curvefit
import numpy as np
import matplotlib.pyplot as plt


def modele(x, a, b):
    return a*x + b


def modele_derivee(x, a, b):
    return a


# création des données expérimentales
N = 10  # nb points exp
x = np.linspace(0.1, 2, N)
y = modele(x, 2, -1)
# bruit en x : constant
sigma_x = .05*np.ones(N)
x_noised = x + sigma_x*np.random.randn(N)
# bruit en x : non constant
sigma_x2 = sigma_x*(x+1)
x_noised2 = x + sigma_x2*np.random.randn(N)
# bruit en y
sigma_y = 0.15*np.ones(N)
y_noised = y + sigma_y*np.random.randn(N)
# bruit en y non constant - ne pas prendre proportionnel à x,
# sinon méthodes leastsquare et effective variance identiques
sigma_y2 = sigma_y*(x**2+1)
y_noised2 = y + sigma_y2*np.random.randn(N)


# fit et affichage
fmt = {'fmt': 'o', 'capthick': 2}
x_fit = np.linspace(min(x_noised.min(), x_noised2.min()), max(x_noised.max(), x_noised2.max()), 100)

plt.figure()
plt.title("Bruit en x et y constants")
plt.errorbar(x_noised, y_noised, xerr=sigma_x, yerr=sigma_y, **fmt)
# Fit sans prendre en compte les incertitudes
pfit, errs, chi2 = curvefit(modele, x_noised, y_noised, p0=[1, 0])
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit sans incertitudes\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en ne prenant en compte que l'incertitude sur y = c'est la même chose que précédemment
pfit, errs, chi2 = curvefit(
    modele, x_noised, y_noised, p0=[1, 0], datayerrors=sigma_y)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en prenant en compte les incertitudes sur x et y
pfit, errs, chi2 = curvefit(
    modele, x_noised, y_noised, p0=[1, 0], dataxerrors=sigma_x, datayerrors=sigma_y, function_derivate=modele_derivee)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur x et y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")

plt.legend()
plt.show(block=False)

plt.figure()
plt.title("Bruit en x et y non constants")
plt.errorbar(x_noised2, y_noised2, xerr=sigma_x2, yerr=sigma_y2, **fmt)
# Fit sans prendre en compte les incertitudes
pfit, errs, chi2 = curvefit(modele, x_noised2, y_noised2, p0=[1, 0])
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit sans incertitudes\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en ne prenant en compte que l'incertitude sur y
pfit, errs, chi2 = curvefit(
    modele, x_noised2, y_noised2, p0=[1, 0], datayerrors=sigma_y2)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en prenant en compte les incertitudes sur x et y
pfit, errs, chi2 = curvefit(
    modele, x_noised2, y_noised2, p0=[1, 0],
    dataxerrors=sigma_x2, datayerrors=sigma_y2, function_derivate=modele_derivee)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur x et y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")

plt.legend()
plt.show(block=False)

plt.show()
