# -*- coding: utf-8 -*-
"""
La loi normale et son cumul sur le même graphe : la densité, et la
probabilité qu'un tirage tombe à moins de t écarts-types de la moyenne.
Voir doc/incertitudes.md.

L'aire sous la densité entre -t et +t est la valeur du cumul en t : 68 %,
95 % et 99,7 % à un, deux et trois écarts-types.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.incertitudes import loi_normale, loi_normale_cumulee

x = np.linspace(-4, 4, 801)  # en écarts-types, autour de la moyenne
t = np.linspace(0, 4, 401)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, loi_normale(x), color="tab:blue", label="loi_normale(x) : la densité")
ax.plot(
    t,
    loi_normale_cumulee(t),
    color="tab:red",
    label="loi_normale_cumulee(t) : la probabilité d'un tirage entre -t et +t",
)
# l'aire sous la densité entre -k et +k, et le point du cumul qui la mesure
for k, opacite in ((3, 0.15), (2, 0.3), (1, 0.5)):
    dedans = abs(x) <= k
    p = loi_normale_cumulee(k)
    ax.fill_between(x[dedans], loi_normale(x[dedans]), color="tab:blue", alpha=opacite, linewidth=0)
    ax.plot([k, k], [loi_normale(k), p], color="tab:red", linestyle=":", linewidth=0.8)
    ax.plot(k, p, "o", color="tab:red")
    ax.annotate(
        ("%.2f %%" % (100 * p)).replace(".", ","),
        (k, p),
        textcoords="offset points",
        xytext=(6, -12),
        fontsize=9,
        color="tab:red",
    )
    print("à %d écart(s)-type(s) : %.4f" % (k, p))
ax.set_xlabel("écart à la moyenne, en écarts-types")
ax.set_xticks(np.arange(-4, 5))
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
plt.savefig("incertitudes_loi_normale.pdf")

plt.show()
