# -*- coding: utf-8 -*-
"""
Propager des incertitudes par la méthode de Monte-Carlo, sans formule de
dérivées partielles : g par un pendule, un quotient dont la loi est
dissymétrique, une droite ajustée sur cent mille tirages sans boucle, un
modèle quelconque ajusté par tirages. Voir doc/incertitudes.md.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit, formater
from tpllg.montecarlo import Point, SerieLineaire, ajuster_modele

np.random.seed(0)          # pour retrouver les mêmes tirages d'une exécution à l'autre

# 1. g par un pendule : L = 1,000 ± 0,002 m, T = 2,007 ± 0,010 s
L = Point(1.000, 0.002)
T = Point(2.007, 0.010)
g = 4*np.pi**2*L/T**2
print("g =", formater(g.val, g.u, "m/s²"), "(Monte-Carlo, %d tirages)" % g.N)
g_lin = 4*np.pi**2*1.000/2.007**2
u_lin = g_lin*np.sqrt((0.002/1.000)**2 + (2*0.010/2.007)**2)
print("g =", formater(g_lin, u_lin, "m/s²"), "(formule de propagation linéaire)")
g.show()
plt.savefig("montecarlo_g.pdf")

# 2. un quotient à grande incertitude relative : la loi n'est plus symétrique
a = Point(1.0, 0.1)
b = Point(2.0, 0.3)          # 15 %
q = a/b
bas, haut = q.quantiles()
print("a/b =", formater(q.val, q.u), "; médiane %.3f ; 68 %% des tirages entre %.3f et %.3f"
      % (np.median(q.tirage), bas, haut))

# 3. une droite ajustée sur chaque tirage des mesures, face à curvefit
x = np.linspace(0, 10, 10)
x_mes = x + np.random.normal(0, 0.2, x.size)
y_mes = 2*x + 1 + np.random.normal(0, 0.5, x.size)
debut = time.perf_counter()
serie = SerieLineaire(x_mes, 0.2, y_mes, 0.5)
pa, pb = serie.ajuster()
duree = time.perf_counter() - debut
print("Monte-Carlo : a =", formater(pa.val, pa.u), " b =", formater(pb.val, pb.u),
      " (%d tirages en %.2f s)" % (serie.N, duree))
pfit, err, chi2 = curvefit(lambda x, a, b: a*x + b, x_mes, y_mes, p0=[1, 0],
                           datayerrors=0.5, dataxerrors=0.2,
                           function_derivate=lambda x, a, b: a, verbose=False)
print("curvefit    : a =", formater(pfit[0], err[0]), " b =", formater(pfit[1], err[1]),
      " chi2 réduit = %.2f" % chi2)
plt.figure()
plt.errorbar(x_mes, y_mes, xerr=0.2, yerr=0.5, fmt="o", label="mesures")
plt.plot(x, pa.val*x + pb.val, label="Monte-Carlo")
plt.plot(x, pfit[0]*x + pfit[1], "--", label="curvefit")
plt.legend()
plt.savefig("montecarlo_droite.pdf")

# 4. un modèle quelconque : une exponentielle, curve_fit sur chaque tirage
t = np.linspace(0, 5, 12)
u = 2*np.exp(-t/1.5) + np.random.normal(0, 0.02, t.size)
debut = time.perf_counter()
pA, ptau = ajuster_modele(lambda t, A, tau: A*np.exp(-t/tau), t, 0.01, u, 0.02, p0=[1, 1], N=2000)
duree = time.perf_counter() - debut
print("exponentielle : A =", formater(pA.val, pA.u), " tau =", formater(ptau.val, ptau.u, "s"),
      " (2000 tirages en %.1f s)" % duree)
plt.show()
