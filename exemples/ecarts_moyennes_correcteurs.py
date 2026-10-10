"""
Combien de copies faut-il pour comparer les moyennes de plusieurs
correcteurs ? Une simulation de Monte-Carlo, avec la loi normale de
tpllg.incertitudes. Voir doc/incertitudes.md.

n_corr correcteurs se partagent des copies notées sur 20. Même s'ils notent
tous de la même façon, les moyennes de leurs paquets diffèrent, par le seul
hasard de la répartition des copies. Le script estime par Monte-Carlo, en
fonction du nombre de copies par correcteur, la dispersion de ces moyennes
(écart-type des moyennes des correcteurs, et écart absolu moyen à la moyenne
générale), avec son incertitude à 68 %.

Deux distributions des notes, simulées, sont comparées :

- celle d'un devoir difficile, dissymétrique (loi normale asymétrique :
  densité normale × (1 + erf(alpha (x - m)/(s sqrt 2))), alpha < 0) ;
- une distribution centrée, normale.

L'écart-type des moyennes décroît comme sigma/sqrt(n_copies) : c'est l'écart
en dessous duquel une différence de moyennes entre correcteurs ne signifie
rien. Aucune note réelle : les distributions, le nombre de correcteurs et
les nombres de copies sont des valeurs d'illustration.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from tpllg.incertitudes import loi_normale

TOTAL = 20
NOTES = np.arange(0, 2 * TOTAL + 1) / 2  # notes au demi-point
N_MC = 2_000  # tirages Monte-Carlo par point
CORRECTEURS = [5]
COPIES = np.arange(5, 81, 5)  # nombres de copies par correcteur
rng = np.random.default_rng()


def loi_normale_asymetrique(x, m=0, s=1, alpha=1):
    return loi_normale(x, m, s) * (1 + special.erf((x - m) * alpha / s / np.sqrt(2)))


def histo(n):
    return loi_normale_asymetrique(n, m=12, s=6, alpha=-3)  # devoir difficile


def histo_2(n):
    return loi_normale(n, m=10, s=4)  # distribution centrée


def calc_pdf(func):
    proba = func(NOTES)
    return proba / proba.sum()


def show_pdf(p, titre):
    notes = rng.choice(NOTES, N_MC, p=p)
    m, s, haut = notes.mean(), notes.std(), p.max()
    fig, ax = plt.subplots(num=titre)
    ax.plot(NOTES, p)
    ax.plot([m, m], [0, haut], "--", label=f"moyenne = {m:.2f}")
    (ligne,) = ax.plot([m - s, m - s], [0, haut], label=f"écart-type = {s:.2f}")
    ax.plot([m + s, m + s], [0, haut], "--", color=ligne.get_color())
    ax.set_title(titre)
    ax.set_xlabel(f"note sur {TOTAL}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def calc_ecart(n_corr, n_copies, p):
    """(écarts absolus moyens, écarts-types) des moyennes des correcteurs, N_MC tirages."""
    notes = rng.choice(NOTES, (N_MC, n_corr, n_copies), p=p)
    moyennes = notes.mean(axis=2)  # (N_MC, n_corr)
    moyenne_generale = notes.mean(axis=(1, 2))[:, np.newaxis]
    ecarts_abs = np.abs(moyennes - moyenne_generale).mean(axis=1)
    ecarts_std = moyennes.std(axis=1)
    return ecarts_abs, ecarts_std


def show_histo(values, titre, largeur=5, nbins=100):
    fig, ax = plt.subplots(num=titre)
    centre, u = values.mean(), values.std()
    x1, x2 = centre - largeur * u, centre + largeur * u
    n, _, _ = ax.hist(values, bins=np.linspace(x1, x2, nbins), density=True)
    haut = max(n) * 1.1
    ax.plot([centre, centre], [0, haut], "--", label=f"moyenne = {centre:.2f}")
    (ligne,) = ax.plot([centre - u, centre - u], [0, haut], "--", label=f"écart-type = {u:.2f}")
    ax.plot([centre + u, centre + u], [0, haut], "--", color=ligne.get_color())
    x = np.linspace(x1, x2, 500)
    ax.plot(x, loi_normale(x, centre, u), label="loi normale")
    ax.set_title(titre)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def generate(correcteurs, copies, p):
    """Moyennes et écarts-types, sur les tirages, des deux indicateurs de dispersion."""
    taille = (len(correcteurs), len(copies))
    resultats = {nom: np.zeros(taille) for nom in ("abs", "u_abs", "std", "u_std")}
    for j, n_corr in enumerate(correcteurs):
        for i, n_copies in enumerate(copies):
            e_abs, e_std = calc_ecart(n_corr, n_copies, p)
            resultats["abs"][j, i], resultats["u_abs"][j, i] = e_abs.mean(), e_abs.std()
            resultats["std"][j, i], resultats["u_std"][j, i] = e_std.mean(), e_std.std()
    return resultats


distributions = {
    "devoir difficile": calc_pdf(histo),
    "distribution centrée": calc_pdf(histo_2),
}
for nom, p in distributions.items():
    show_pdf(p, f"Notes supposées : {nom}")

# Une simulation détaillée : 5 correcteurs, 30 copies chacun
_, e_std = calc_ecart(n_corr=5, n_copies=30, p=distributions["distribution centrée"])
show_histo(e_std, "Écart-type des moyennes, 5 correcteurs × 30 copies")

fig, ax = plt.subplots(num="Écarts entre correcteurs")
for nom, p in distributions.items():
    moyenne = (NOTES * p).sum()
    resultats = generate(CORRECTEURS, COPIES, p)
    for j, n_corr in enumerate(CORRECTEURS):
        e, u = resultats["std"][j], resultats["u_std"][j]
        (ligne,) = ax.plot(COPIES, e, label=f"{n_corr} correcteurs, {nom}, moyenne {moyenne:.1f}")
        ax.fill_between(COPIES, e - u, e + u, color=ligne.get_color(), alpha=0.2)
        ax.plot(COPIES, resultats["abs"][j], "--", color=ligne.get_color())
        print(
            f"{nom} : écart-type des moyennes de {e[0]:.2f} ({COPIES[0]} copies) à {e[-1]:.2f} ({COPIES[-1]})"
        )
ax.legend()
ax.grid()
ax.set_ylim(bottom=0)
ax.set_xlim(0, COPIES.max())
ax.set_xlabel("nombre de copies corrigées par correcteur")
ax.set_ylabel("écart-type des moyennes (± écart-type) ;\nen tirets : écart absolu moyen")
fig.tight_layout()
plt.show()
