"""
Une série de mesures répétées : la moyenne m, l'écart-type s de la série et
l'incertitude-type delta = s/sqrt(N) sur la moyenne, lus sur l'histogramme
des mesures, puis sur les mesures dans l'ordre où elles ont été faites.
Voir doc/incertitudes.md.

Deux séries côte à côte, cinq mesures puis cinquante de même dispersion :
s ne change pas, c'est la dispersion d'une mesure ; delta rétrécit comme
1/sqrt(N), c'est ce qu'on gagne à répéter.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import formater
from tpllg.incertitudes import incertitudes, loi_normale

rng = np.random.default_rng(0)  # les mêmes mesures d'une exécution à l'autre

PAS = 0.01  # la résolution de la mesure : la largeur d'une classe de l'histogramme
mesures = np.array([9.78, 9.81, 9.85, 9.79, 9.83])  # g, en m/s², cinq fois
suite = np.round(rng.normal(9.81, 0.03, 45), 2)  # quarante-cinq mesures de plus, simulées
series = (("5 mesures", mesures), ("50 mesures", np.concatenate([mesures, suite])))


def reperes(ax, m, s, delta, vertical):
    """La moyenne en trait, m ± s et m ± delta en deux bandes colorées :
    verticales sur un histogramme, horizontales sur les mesures dans l'ordre."""
    trait, bande = (ax.axvline, ax.axvspan) if vertical else (ax.axhline, ax.axhspan)
    bande(m - s, m + s, color="tab:orange", alpha=0.3, zorder=0, label="m ± s : la dispersion d'une mesure")
    bande(m - delta, m + delta, color="tab:red", alpha=0.5, zorder=0, label="m ± delta : le résultat")
    trait(m, color="darkred", label="m : la moyenne")


def titre_de(nom, m, s, delta):
    return f"{nom}\nm = {m:.3f}   s = {s:.3f}   delta = {delta:.3f}"


def legende_commune(fig, ax):
    """Une seule légende sous les deux repères, qui portent les mêmes tracés."""
    traces, noms = ax.get_legend_handles_labels()
    fig.legend(traces, noms, loc="lower center", ncol=len(noms), fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))


for nom, serie in series:
    m, delta, s = incertitudes(serie)
    print(f"{nom:10s} : m = {m:.4f}  s = {s:.4f}  delta = {delta:.4f}  soit g = {formater(m, delta, 'm/s²')}")

# 1. l'histogramme des mesures : une classe par valeur lisible, centrée sur elle
classes = np.arange(9.705, 9.925, PAS)
x = np.linspace(classes[0], classes[-1], 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
for ax, (nom, serie) in zip(axes, series):
    m, delta, s = incertitudes(serie)
    ax.hist(serie, bins=classes, color="tab:blue", alpha=0.75, label="les mesures")
    # la loi normale de mêmes m et s, à l'échelle de l'histogramme : N × largeur de classe × densité
    ax.plot(x, len(serie) * PAS * loi_normale(x, m, s), color="navy", label="la loi normale de mêmes m et s")
    reperes(ax, m, s, delta, vertical=True)
    ax.set_title(titre_de(nom, m, s, delta), fontsize=10)
    ax.set_xlabel("g (m/s²)")
    ax.set_ylabel("nombre de mesures")
legende_commune(fig, axes[1])
plt.savefig("incertitudes_serie_histogramme.pdf")

# 2. les mesures dans l'ordre où elles ont été faites
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, gridspec_kw={"width_ratios": [2, 5]})
for ax, (nom, serie) in zip(axes, series):
    m, delta, s = incertitudes(serie)
    ax.plot(np.arange(1, len(serie) + 1), serie, "o", color="tab:blue", label="les mesures")
    reperes(ax, m, s, delta, vertical=False)
    ax.set_title(titre_de(nom, m, s, delta), fontsize=10)
    ax.set_xlabel("numéro de la mesure")
axes[0].set_ylabel("g (m/s²)")
axes[0].set_xticks(np.arange(1, len(mesures) + 1))
legende_commune(fig, axes[1])
plt.savefig("incertitudes_serie_indices.pdf")

plt.show()
