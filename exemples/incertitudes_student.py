# -*- coding: utf-8 -*-
"""
Le coefficient de Student en fonction du nombre N de mesures, à 68 % et à
95 % : ce par quoi on multiplie s/sqrt(N) pour une incertitude de type A.
Voir doc/incertitudes.md.

Il tend vers celui de la loi normale, 1 et 2, quand N grandit ; il s'envole
quand les mesures sont peu nombreuses, parce que l'écart-type s est alors
lui-même mal connu.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from tpllg.incertitudes import loi_normale_cumulee, student_coef

N = np.arange(2, 101)
N_table = np.array([2, 3, 5, 10, 30, 100])  # les valeurs du tableau de la fiche

fig, ax = plt.subplots(figsize=(7, 4.5))
for sigma, couleur in ((1, "tab:blue"), (2, "tab:red")):
    niveau = 100 * loi_normale_cumulee(sigma)
    ax.plot(N, student_coef(sigma, N), color=couleur, label="sigma = %d : à %.0f %%" % (sigma, niveau))
    ax.axhline(sigma, color=couleur, linestyle="--", linewidth=0.8)
    t_table = student_coef(sigma, N_table)
    ax.plot(N_table, t_table, "o", color=couleur)
    for n, t in zip(N_table, t_table):
        # l'étiquette au-dessus à droite ; à gauche pour N = 2 à 68 %, où elle toucherait l'asymptote t = 2
        decalage = (-27, -3) if (sigma, n) == (1, 2) else (4, 5)
        ax.annotate(
            "%.2f" % t, (n, t), textcoords="offset points", xytext=decalage, fontsize=8, color=couleur
        )
    print("sigma = %d :" % sigma, "  ".join("N = %d : %.2f" % (n, t) for n, t in zip(N_table, t_table)))
ax.plot([], [], "k--", linewidth=0.8, label="loi normale : t = sigma")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(N_table)
ax.set_yticks([1, 2, 3, 5, 10, 15])
ax.set_xlim(1.55, 130)
ax.set_ylim(0.8, 18)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.minorticks_off()
ax.set_xlabel("nombre de mesures N")
ax.set_ylabel("coefficient de Student t")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.savefig("incertitudes_student.pdf")

plt.show()
