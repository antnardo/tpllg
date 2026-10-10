"""
Incertitudes : loi normale, coefficient de Student, estimateurs d'une série de
mesures — le module incertitudes de dataanalysis (2018), fusionné ici.

@author: a. marchand
"""

import numpy as np
from scipy import special, stats

__all__ = ["incertitudes", "loi_normale", "loi_normale_cumulee", "student_coef"]


def loi_normale(x, m=0, s=1):
    """La densité de la loi normale de moyenne m et d'écart-type s :
    normalisée, son intégrale vaut 1."""
    return 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-((x - m) ** 2) / (2 * s**2))


def loi_normale_cumulee(t):
    """L'intégrale de -t à t de la loi normale centrée réduite : la
    probabilité qu'un tirage tombe à moins de t écarts-types de la moyenne
    (0,6827 pour t = 1, 0,9545 pour t = 2)."""
    return special.erf(t / np.sqrt(2))


def student_coef(sigma, n):
    """Le coefficient de Student t pour n mesures, au niveau de confiance de
    la loi normale à `sigma` écarts-types : sigma = 1 pour 68,27 %, sigma = 2
    pour 95,45 % (et 1,96 pour 95 %).

    Pour T de Student à k = n - 1 degrés de liberté, t est tel que
    P(-t < T < t) = loi_normale_cumulee(sigma). Il tend vers sigma quand n
    grandit. `n` peut être un tableau. Un `n` qui n'est pas un entier au
    moins égal à 2 est refusé (ValueError) : c'est le plus souvent les deux
    arguments inversés, student_coef(n, 0.95)."""
    n = np.asarray(n)
    if n.dtype.kind not in "iuf" or np.any(n < 2) or np.any(n != np.floor(n)):
        raise ValueError(
            f"student_coef(sigma, n) : n = {n} doit être le nombre de mesures, un entier au moins égal "
            f"à 2, et sigma = {sigma} le niveau en écarts-types de la loi normale "
            "(1 pour 68,27 %, 2 pour 95,45 %, 1,96 pour 95 %)"
        )
    if not sigma > 0:
        raise ValueError(f"student_coef(sigma, n) : sigma = {sigma} doit être strictement positif")
    niveau = special.erf(sigma / np.sqrt(2))
    gamma = (1 - niveau) / 2  # la probabilité de tirer au-dessus de t
    return stats.t(n - 1).isf(gamma)


def incertitudes(liste, sigma=1, advanced=False, debug=False):
    """Une série de mesures répétées : rend (m, delta, s), la moyenne,
    l'incertitude sur la moyenne et l'écart-type de la série.

    s est l'estimateur à N - 1 de l'écart-type (sa variance s² est sans
    biais ; s lui-même sous-estime un peu l'écart-type quand N est petit).
    delta = k s/√N, avec k = sigma (la loi normale : sigma = 1 pour une
    incertitude-type, à 68 %, sigma = 2 pour 95 %) ; avec advanced=True,
    k est le coefficient de Student au même niveau, plus grand quand les
    mesures sont peu nombreuses parce que s est alors mal connu.

    source : https://fr.wikipedia.org/wiki/Loi_de_Student#Application_:_intervalle_de_confiance_associ%C3%A9_%C3%A0_l%E2%80%99esp%C3%A9rance_d%E2%80%99une_variable_de_loi_normale_de_variance_inconnue
    """
    mesures = np.asarray(liste, dtype=float)
    n = mesures.size
    if n < 2:
        raise ValueError("incertitudes : il faut au moins deux mesures")
    m = mesures.mean()
    s = mesures.std(ddof=1)
    k = student_coef(sigma, n) if advanced else sigma
    delta = k * s / np.sqrt(n)
    if debug:
        print(n, mesures)
        print(m, delta, s, k)
    return m, delta, s
