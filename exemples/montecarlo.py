# -*- coding: utf-8 -*-
"""
Propager des incertitudes par la méthode de Monte-Carlo, sans formule de
dérivées partielles : g par un pendule, un quotient dont la loi est
dissymétrique, une droite ajustée sur cent mille tirages sans boucle, un
modèle quelconque ajusté par tirages, une résistance par la loi d'Ohm, la
valeur absolue d'une différence. Chaque cas trace sa figure, celles de
doc/incertitudes.md.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit, formater
from tpllg.montecarlo import Point, SerieLineaire, ajuster_modele

np.random.seed(0)  # pour retrouver les mêmes tirages d'une exécution à l'autre


def histogrammes(axes, points, noms, nbins=200):
    """L'histogramme de chaque Point dans son repère, par sa méthode show :
    les tirages, la moyenne, ± l'écart-type, la loi normale de mêmes paramètres."""
    for ax, point, nom in zip(axes, points, noms):
        point.show(ax, nbins=nbins)
        ax.set_xlabel(nom)
        ax.locator_params(axis="x", nbins=6)
        ax.legend(fontsize=8)


# 1. g par un pendule : L = 1,000 ± 0,002 m, T = 2,007 ± 0,010 s
L = Point(1.000, 0.002)
T = Point(2.007, 0.010)
g = 4 * np.pi**2 * L / T**2
print("g =", formater(g.val, g.u, "m/s²"), "(Monte-Carlo, %d tirages)" % g.N)
g_lin = 4 * np.pi**2 * 1.000 / 2.007**2
u_lin = g_lin * np.sqrt((0.002 / 1.000) ** 2 + (2 * 0.010 / 2.007) ** 2)
print("g =", formater(g_lin, u_lin, "m/s²"), "(formule de propagation linéaire)")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
histogrammes(axes, (L, T, g), ("L (m)", "T (s)", "g = 4π²L/T² (m/s²)"))
fig.tight_layout()
plt.savefig("montecarlo_g.pdf")

# 2. un quotient à grande incertitude relative : la loi n'est plus symétrique
a = Point(1.0, 0.1)
b = Point(2.0, 0.3)  # 15 %
q = a / b
bas, haut = q.quantiles()
print(
    "a/b =",
    formater(q.val, q.u),
    "; médiane %.3f ; 68 %% des tirages entre %.3f et %.3f" % (np.median(q.tirage), bas, haut),
)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
histogrammes(axes, (a, b, q), ("a", "b", "q = a/b"))
axes[2].axvline(np.median(q.tirage), color="k", linestyle=":", label="médiane=%.3f" % np.median(q.tirage))
axes[2].axvspan(bas, haut, color="gold", alpha=0.35, label="68 % des tirages")
axes[2].legend(fontsize=8)
fig.tight_layout()
plt.savefig("montecarlo_quotient.pdf")

# 3. une droite ajustée sur chaque tirage des mesures, face à curvefit
x = np.linspace(0, 10, 10)
x_mes = x + np.random.normal(0, 0.2, x.size)
y_mes = 2 * x + 1 + np.random.normal(0, 0.5, x.size)
debut = time.perf_counter()
serie = SerieLineaire(x_mes, 0.2, y_mes, 0.5)
pa, pb = serie.ajuster()
duree = time.perf_counter() - debut
print(
    "Monte-Carlo : a =",
    formater(pa.val, pa.u),
    " b =",
    formater(pb.val, pb.u),
    " (%d tirages en %.2f s)" % (serie.N, duree),
)
pfit, err, chi2 = curvefit(
    lambda x, a, b: a * x + b,
    x_mes,
    y_mes,
    p0=[1, 0],
    datayerrors=0.5,
    dataxerrors=0.2,
    function_derivate=lambda x, a, b: a,
    verbose=False,
)
print(
    "curvefit    : a =",
    formater(pfit[0], err[0]),
    " b =",
    formater(pfit[1], err[1]),
    " chi2 réduit = %.2f" % chi2,
)
plt.figure()
plt.errorbar(x_mes, y_mes, xerr=0.2, yerr=0.5, fmt="o", label="mesures")
plt.plot(x, pa.val * x + pb.val, label="Monte-Carlo")
plt.plot(x, pfit[0] * x + pfit[1], "--", label="curvefit")
plt.legend()
plt.savefig("montecarlo_droite.pdf")

# ce que les tirages contiennent : cent des cent mille droites, puis les lois de a et de b
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for k in range(100):
    axes[0].plot(x, pa.tirage[k] * x + pb.tirage[k], color="tab:orange", alpha=0.15)
axes[0].plot([], [], color="tab:orange", label="100 droites, une par tirage")
axes[0].errorbar(x_mes, y_mes, xerr=0.2, yerr=0.5, fmt="o", markersize=4, label="mesures")
axes[0].plot(serie.xi, serie.yi, "k+", label="un tirage des mesures")
axes[0].plot(x, pa.tirage[0] * x + pb.tirage[0], "k", linewidth=0.8, label="sa droite")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8)
histogrammes(axes[1:], (pa, pb), ("pente a", "ordonnée à l'origine b"))
fig.tight_layout()
plt.savefig("montecarlo_droite_tirages.pdf")


# 4. un modèle quelconque : une exponentielle, curve_fit sur chaque tirage
def exponentielle(t, A, tau):
    return A * np.exp(-t / tau)


t = np.linspace(0, 5, 12)
u = 2 * np.exp(-t / 1.5) + np.random.normal(0, 0.02, t.size)
debut = time.perf_counter()
pA, ptau = ajuster_modele(exponentielle, t, 0.01, u, 0.02, p0=[1, 1], N=2000)
duree = time.perf_counter() - debut
print(
    "exponentielle : A =",
    formater(pA.val, pA.u),
    " tau =",
    formater(ptau.val, ptau.u, "s"),
    " (2000 tirages en %.1f s)" % duree,
)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
t_fin = np.linspace(0, 5, 200)
for k in range(100):
    axes[0].plot(t_fin, exponentielle(t_fin, pA.tirage[k], ptau.tirage[k]), color="tab:orange", alpha=0.15)
axes[0].plot([], [], color="tab:orange", label="100 ajustements, un par tirage")
axes[0].errorbar(t, u, xerr=0.01, yerr=0.02, fmt="o", markersize=4, label="mesures")
axes[0].set_xlabel("t (s)")
axes[0].set_ylabel("u (V)")
axes[0].legend(fontsize=8)
histogrammes(axes[1:], (pA, ptau), ("A (V)", "tau (s)"), nbins=60)
fig.tight_layout()
plt.savefig("montecarlo_modele.pdf")

# 5. une résistance par la loi d'Ohm, avec les incertitudes des multimètres
U = Point(4.87, 0.5 * 0.01 * 4.87 + 0.005)  # 0,5 % + 5 mV : ce que la notice dit
I = Point(0.0213, 0.008 * 0.0213 + 0.0001)
R = U / I
print("R =", formater(R.val, R.u, "Ω"))

# 6. une fonction non dérivable : la valeur absolue d'une différence
d = abs(Point(1.02, 0.05) - Point(1.00, 0.05))
print("|x1 - x2| =", formater(d.val, d.u), " médiane %.3f" % np.median(d.tirage))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
histogrammes(axes, (R, d), ("R = U/I (Ω)", "|x1 - x2|"))
axes[1].axvline(np.median(d.tirage), color="k", linestyle=":", label="médiane=%.3f" % np.median(d.tirage))
axes[1].axvline(0.02, color="tab:purple", label="|1,02 - 1,00| = 0,02")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.savefig("montecarlo_cas.pdf")

plt.show()
