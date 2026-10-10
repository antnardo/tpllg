"""
Ce que plusieurs modules partagent sans que l'utilisateur ait à le voir : la
mise en forme des tableaux reçus, et le générateur de nombres aléatoires.
Rien ici n'est public ; les modules l'importent, pas les scripts.
"""

import numpy as np

__all__ = ["generateur", "incertitude", "serie", "voie"]


def voie(t, v, minimum=1):
    """Un signal (t, v) en deux tableaux 1D de flottants de même longueur,
    d'au moins `minimum` points : une seule voie, pas le tableau 2D d'une
    acquisition."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.ndim != 1 or t.shape != v.shape or t.size < minimum:
        raise ValueError(
            "t et v : deux tableaux 1D de même longueur, une seule voie (temps[0] et tensions[0])"
        )
    return t, v


def incertitude(valeur, forme):
    """Une incertitude ramenée à la forme des données : un nombre vaut pour
    tous les points, un tableau est rendu tel quel."""
    return np.broadcast_to(np.asarray(valeur, dtype=float), forme)


def serie(x, u_x, y, u_y):
    """Une série de mesures (x ± u_x, y ± u_y) : x et y en tableaux 1D de
    même longueur, les incertitudes ramenées à leur forme."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x et y doivent être deux tableaux 1D de même longueur")
    return x, incertitude(u_x, x.shape), y, incertitude(u_y, y.shape)


def generateur(graine=None):
    """Un générateur de nombres aléatoires, reproductible pour une `graine`
    entière : np.random.default_rng, ou RandomState sur les postes dont le
    numpy (< 1.17) ne le connaît pas. Les deux ont normal, uniform et
    triangular, avec les mêmes arguments."""
    try:
        return np.random.default_rng(graine)
    except AttributeError:  # numpy < 1.17 : pas de Generator
        return np.random.RandomState(graine)
