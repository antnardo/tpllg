"""
L'incertitude sur une moyenne : loi normale et coefficient de Student, avec
tpllg.incertitudes. Voir doc/incertitudes.md.

Pour n mesures d'une même grandeur, de moyenne m et d'écart-type expérimental
s, l'incertitude sur la moyenne au niveau de confiance de la loi normale à
± k sigma vaut t s/sqrt(n), où le coefficient de Student t dépend de n - 1
degrés de liberté (il tend vers k quand n est grand). incertitudes rend
l'incertitude-type s/sqrt(n) ; t est le facteur d'élargissement qu'on donne à
formater, k=student_coef(1, n) pour 68 %, k=student_coef(2, n) pour 95 %.
Trois figures :

- l'histogramme de 30 tirages normaux, la moyenne et l'écart-type estimés,
  et l'incertitude sur la moyenne avec Student, à 68 % et à 95 % côte à côte
  (incertitudes, student_coef, formater) ;
- la probabilité de sortir de ± t écarts-types pour la loi normale
  (loi_normale_cumulee) ;
- les lois de Student pour différents n, et les coefficients t à 68 %
  (student_coef).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

from tpllg.ajustement import formater
from tpllg.incertitudes import incertitudes, loi_normale, loi_normale_cumulee, student_coef

rng = np.random.default_rng(30)


def histogramme(N=30, B=20):
    """N tirages de la loi normale centrée réduite : ce qu'on en estime."""
    mesures = rng.standard_normal(N)
    m, delta, s = incertitudes(mesures)  # delta : l'incertitude-type sur la moyenne, s/sqrt(N)
    t68, t95 = student_coef(1, N), student_coef(2, N)  # les facteurs d'élargissement de Student
    print(f"{N} mesures : écart-type {s:.3f}, incertitude-type sur la moyenne {delta:.3f}")
    print(f"moyenne {formater(m, delta, k=t68)} (68 %, Student, t = {t68:.3f})")
    print(f"moyenne {formater(m, delta, k=t95)} (95 %, Student, t = {t95:.3f})")
    x = np.linspace(-4, 4, 500)

    effectifs, bords = np.histogram(mesures, bins=min(B, N))
    largeur = bords[1] - bords[0]
    centres = (bords[:-1] + bords[1:]) / 2
    densite = effectifs / (N * largeur)
    h = max(loi_normale(0) * 1.1, densite.max())

    plt.figure("Histogramme et moyenne")
    plt.plot(x, loi_normale(x), "k")
    plt.plot([0, 0], [0, h], "k", label="moyenne et écart-type théoriques")
    plt.plot([1, 1], [0, h], "k-.")
    plt.plot([-1, -1], [0, h], "k-.")
    plt.bar(centres, densite, align="center", width=0.95 * largeur)
    plt.plot(x, loi_normale(x, m, s), "r")
    plt.plot([m, m], [0, h], "r--", label="moyenne et écart-type estimés")
    plt.plot([m + s, m + s], [0, h], "r-.")
    plt.plot([m - s, m - s], [0, h], "r-.")
    for t, niveau, style in ((t68, "68 %", "g--"), (t95, "95 %", "m:")):
        plt.plot([m + t * delta] * 2, [0, h], style, label=f"incertitude sur la moyenne, {niveau}, Student")
        plt.plot([m - t * delta] * 2, [0, h], style)
    plt.legend(fontsize=8)
    plt.tight_layout()


def tableau_proba():
    """La probabilité de sortir de ± t écarts-types, loi normale."""
    plt.figure("Probabilité normale réduite")
    xmax = 5
    t = np.linspace(0, xmax, 100)
    ymin = 1 - loi_normale_cumulee(xmax)
    plt.plot(t, 1 - loi_normale_cumulee(t))
    plt.yscale("log")
    plt.title("Probabilité de sortir de ± t écarts-types")
    plt.ylabel("$1 - p$")
    plt.xlabel("écart symétrique à la moyenne, en nombre d'écarts-types")
    for t in (1, 2, 3, 4):
        y = loi_normale_cumulee(t)
        print(f"à ± {t} écart(s)-type(s) : {100 * y:.4g} % des tirages")
        plt.plot([t, t, 0], [ymin, 1 - y, 1 - y], "r--")
        plt.text(t, 1 - y, f"{100 * y:.4g} %")
    plt.tight_layout()


def coef_student(sigma=1):
    """Les lois de Student à n - 1 degrés de liberté et le coefficient t à sigma écarts-types."""
    n_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 100, 1e7]
    plt.figure("Lois de Student")
    x = np.linspace(-10, 10, 1000)
    alpha = special.erf(sigma / np.sqrt(2))  # 1 -> 0.6827
    gamma = (1 - alpha) / 2
    print(f"coefficient de Student t au niveau de confiance {100 * alpha:.2f} %")
    print(f"(probabilité {100 * gamma:.2f} % d'un tirage plus grand que t)")
    print("n (k = n - 1 degrés de liberté)\tt")
    for n in n_values:
        t = student_coef(sigma, n)
        print(f"{n:g}\t{t:.4f}")
        plt.plot(x, stats.t(n - 1).sf(x), label=f"n = {n:g}" if n <= 10 else None)
        if t < 10:
            plt.plot([t, t], [0, gamma], "r--")
    plt.xlabel("t")
    plt.ylabel("P(T > t)")
    plt.legend(fontsize=8)
    plt.tight_layout()


histogramme()
tableau_proba()
coef_student()
plt.show()
