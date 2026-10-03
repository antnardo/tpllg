# -*- coding: utf-8 -*-
"""
La variance effective : une incertitude sur x ramenée en y par la pente
locale du modèle, sigma² = sigma_y² + (f'(x) sigma_x)². Sur un modèle courbe,
elle pèse là où la courbe est raide et disparaît là où elle est plate.
Voir doc/ajustement.md.

Une exponentielle décroissante, dix points, 0,15 s d'incertitude sur le
temps et 0,03 V sur la tension.
"""
import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit, resume_parametres

np.random.seed(3)          # les mêmes mesures d'une exécution à l'autre


def modele(t, A, tau):
    return A*np.exp(-t/tau)


def modele_derivee(t, A, tau):   # la dérivée par rapport à t
    return -A/tau*np.exp(-t/tau)


sigma_t, sigma_u = 0.15, 0.03
t_vrai = np.linspace(0.3, 5, 10)
t = t_vrai + np.random.normal(0, sigma_t, t_vrai.size)
u = modele(t_vrai, 2, 1.5) + np.random.normal(0, sigma_u, t_vrai.size)

pfit_y, err_y, chi2_y = curvefit(modele, t, u, p0=[1, 1], datayerrors=sigma_u, verbose=False)
print("incertitudes sur u seules  ", resume_parametres(("A", "tau"), pfit_y, err_y, ("V", "s")).replace("\n", "   "),
      "  chi2 réduit = %.2f" % chi2_y)
pfit, err, chi2 = curvefit(modele, t, u, p0=[1, 1], datayerrors=sigma_u, dataxerrors=sigma_t,
                           function_derivate=modele_derivee, verbose=False)
print("incertitudes sur t et sur u", resume_parametres(("A", "tau"), pfit, err, ("V", "s")).replace("\n", "   "),
      "  chi2 réduit = %.2f" % chi2)
# ce que curvefit calcule pour peser chaque point
pente = modele_derivee(t, *pfit)
sigma_effectif = np.sqrt(sigma_u**2 + (pente*sigma_t)**2)
print("sigma effectif, du premier point au dernier :", " ".join("%.3f" % s for s in sigma_effectif))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# 1. sur un point : la barre en t, la tangente, et ce qu'elle vaut en u
ax = axes[0]
k = 1                                                    # le deuxième point, là où la courbe est raide
t0, u0, p0 = t[k], modele(t[k], *pfit), pente[k]
voisinage = np.linspace(t0 - 2.2*sigma_t, t0 + 2.2*sigma_t, 50)
ax.plot(voisinage, modele(voisinage, *pfit), color="tab:orange", label="le modèle")
ax.plot(voisinage, u0 + p0*(voisinage - t0), color="gray", linestyle="--", linewidth=0.8, label="sa tangente")
ax.plot([t0 - sigma_t, t0 + sigma_t], [u0, u0], color="tab:blue", linewidth=3, label="± sigma_t = %.2f s" % sigma_t)
for signe in (-1, 1):                                    # des bouts de la barre en t à la tangente
    ax.plot([t0 + signe*sigma_t]*2, [u0, u0 + signe*p0*sigma_t], color="gray", linestyle=":", linewidth=0.8)
    ax.plot([t0 + signe*sigma_t, t0 + 2.6*sigma_t], [u0 + signe*p0*sigma_t]*2, color="gray", linestyle=":",
            linewidth=0.8)
barres = (("± |f'| sigma_t = %.3f V" % abs(p0*sigma_t), abs(p0*sigma_t), "tab:green", 2.6),
          ("± sigma_u = %.3f V" % sigma_u, sigma_u, "tab:purple", 2.9),
          ("± sigma effectif = %.3f V" % sigma_effectif[k], sigma_effectif[k], "tab:red", 3.2))
for nom, demi, couleur, place in barres:                 # les trois barres en u, côte à côte
    ax.plot([t0 + place*sigma_t]*2, [u0 - demi, u0 + demi], color=couleur, linewidth=3, label=nom)
ax.plot(t0, u0, "o", color="tab:blue")
ax.set_title("une incertitude sur t ramenée sur u par la pente", fontsize=10)
ax.set_xlabel("t (s)")
ax.set_ylabel("u (V)")
ax.legend(fontsize=8, loc="upper right")

# 2. sur tous les points : la barre effective suit la pente
ax = axes[1]
t_fin = np.linspace(0, 5.3, 200)
ax.plot(t_fin, modele(t_fin, *pfit), color="tab:orange", label="le modèle ajusté")
ax.errorbar(t, u, yerr=sigma_effectif, fmt="none", ecolor="tab:red", elinewidth=4, alpha=0.5,
            label="± sigma effectif : ce que chaque point pèse")
ax.errorbar(t, u, xerr=sigma_t, yerr=sigma_u, fmt="o", color="tab:blue", markersize=4, capsize=2,
            label="les mesures, ± sigma_t et ± sigma_u")
ax.set_title("raide à gauche, plate à droite : la barre effective suit", fontsize=10)
ax.set_xlabel("t (s)")
ax.set_ylabel("u (V)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
plt.savefig("ajustement_variance_effective.pdf")

plt.show()
