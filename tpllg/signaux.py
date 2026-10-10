"""
Signaux acquis : repérer des fronts, découper une fenêtre, estimer une
fréquence, trouver les extremums d'une oscillation amortie et son taux
d'amortissement.

Toutes les fonctions prennent le signal comme l'acquisition le rend, les
instants `t` (en secondes) puis les tensions `v` : la période
d'échantillonnage s'en déduit, il n'y a ni te ni fe à passer. Les formes
de 2026.9, `frequence_pic(v, te)` et `extremums(v, fe, f, offset)`, sont
toujours reconnues (le second argument est alors un nombre) et donnent le
même résultat, sans avertissement ; `decrement_logarithmique` est l'ancien
nom de `taux_amortissement`.

@author: a. marchand
"""

import numpy as np
from scipy.signal import find_peaks

from tpllg._interne import voie

__all__ = [
    "extremums",
    "fenetre",
    "frequence_pic",
    "front_utile",
    "fronts_descendants",
    "fronts_montants",
    "taux_amortissement",
]


def _niveaux(v):
    """Les niveaux bas et haut d'un créneau : la médiane des points de chaque
    côté du milieu entre minimum et maximum. Contrairement à des percentiles,
    elle trouve le niveau haut d'impulsions brèves (rapport cyclique de 1 %)."""
    milieu = (v.min() + v.max()) / 2
    dessus = v > milieu
    if not dessus.any() or dessus.all():
        return v.min(), v.max()
    return np.median(v[~dessus]), np.median(v[dessus])


def fronts_montants(t, v):
    """Les instants des fronts montants d'un créneau.

    Seuil à mi-hauteur entre les niveaux bas et haut, avec hystérésis d'un
    quart de l'amplitude : on passe à l'état haut au-dessus du seuil haut, à
    l'état bas au-dessous du seuil bas, et un front est une transition bas ->
    haut. L'instant est interpolé linéairement au passage à mi-hauteur. Rend
    (instants, (niveau bas, niveau haut)).
    """
    t, v = voie(t, v)
    v_bas, v_haut = _niveaux(v)
    milieu = (v_bas + v_haut) / 2
    marge = (v_haut - v_bas) / 4
    # l'état, +1 haut ou -1 bas, ne change qu'en franchissant le seuil opposé
    etat = np.where(v > milieu + marge, 1, np.where(v < milieu - marge, -1, 0))
    if etat[0] == 0:
        etat[0] = 1 if v[0] > milieu else -1
    indices = np.arange(v.size)
    etat = etat[np.maximum.accumulate(np.where(etat != 0, indices, 0))]
    montees = np.flatnonzero((etat[1:] == 1) & (etat[:-1] == -1)) + 1
    # le dernier point sous la mi-hauteur avant chaque montée, et le suivant
    j = np.maximum.accumulate(np.where(v <= milieu, indices, 0))[montees]
    j = np.minimum(j, v.size - 2)
    pente = v[j + 1] - v[j]
    fraction = np.divide(milieu - v[j], pente, out=np.zeros_like(pente), where=pente != 0)
    return t[j] + fraction * (t[j + 1] - t[j]), (v_bas, v_haut)


def fronts_descendants(t, v):
    """Comme fronts_montants, pour les fronts descendants."""
    fronts, (v_bas, v_haut) = fronts_montants(t, -np.asarray(v, dtype=float))
    return fronts, (-v_haut, -v_bas)


def front_utile(t, t_fronts, fraction=0.45):
    """Le premier front qui laisse derrière lui `fraction` de période avant la
    fin de l'acquisition, et la période estimée entre fronts. Rend (t0,
    periode)."""
    t_fronts = np.asarray(t_fronts, dtype=float)
    if len(t_fronts) == 0:
        raise ValueError("aucun front")
    t_fin = np.asarray(t, dtype=float)[-1]
    periode = np.median(np.diff(t_fronts)) if len(t_fronts) > 1 else t_fin - t_fronts[0]
    for tf in t_fronts:
        if tf + fraction * periode < t_fin:
            return tf, periode
    return t_fronts[0], periode


def _ancienne_forme(t, v, cadence):
    """L'ancienne forme (v, te) ou (v, fe) d'une fonction de 2026.9, reconnue
    au second argument, un nombre : rend (t, v) comme la nouvelle forme les
    attend."""
    if cadence == "te":
        return np.arange(np.size(t)) * float(v), t
    return np.arange(np.size(t)) / float(v), t


def frequence_pic(t, v):
    """La fréquence du pic de la FFT du signal (t, v), moyenne retirée : une
    première estimation de la fréquence d'une oscillation, à 1/durée près."""
    if np.ndim(v) == 0:
        t, v = _ancienne_forme(t, v, "te")
    t, v = voie(t, v, minimum=2)
    spectre = np.abs(np.fft.rfft(v - v.mean()))
    frequences = np.fft.rfftfreq(len(v), t[1] - t[0])
    return frequences[np.argmax(spectre[1:]) + 1]


def fenetre(t, v, t_debut, duree):
    """Les points de (t, v) dans [t_debut, t_debut + duree[."""
    t = np.asarray(t, dtype=float)
    masque = (t >= t_debut) & (t < t_debut + duree)
    return t[masque], np.asarray(v, dtype=float)[masque]


def extremums(t, v, f, offset=0.0, seuil=0.0):
    """Les indices des extremums d'une oscillation de fréquence `f`, en
    valeur absolue autour de `offset`, séparés d'au moins 0,4 période ; ceux
    dont l'écart à `offset` ne dépasse pas `seuil` sont écartés : cinq à dix
    fois l'écart-type du bruit, pour ne pas prendre ses bosses en fin
    d'amortissement, qui aplatiraient l'enveloppe."""
    if np.ndim(v) == 0:
        t, v = _ancienne_forme(t, v, "fe")
    t, v = voie(t, v, minimum=2)
    distance = max(1, int(0.4 / (f * (t[1] - t[0]))))
    pics, _ = find_peaks(np.abs(v - offset), distance=distance, height=seuil if seuil > 0 else None)
    return pics


def taux_amortissement(t_pics, v_pics, offset=0.0):
    """Le taux d'amortissement alpha (s⁻¹) d'une enveloppe exp(-alpha t), par
    régression du logarithme des amplitudes des extremums. Le décrément
    logarithmique, sur une pseudo-période T, est delta = alpha T ; le facteur
    de qualité Q ≈ pi/delta quand l'amortissement est faible."""
    t_pics = np.asarray(t_pics, dtype=float)
    if t_pics.size < 2:
        raise ValueError("il faut au moins deux extremums")
    pente, _ = np.polyfit(t_pics, np.log(np.abs(np.asarray(v_pics, dtype=float) - offset)), 1)
    return -pente


def decrement_logarithmique(t_pics, v_pics, offset=0.0):
    """L'ancien nom (2026.9) de taux_amortissement, qui rend la même chose :
    alpha en s⁻¹, pas le décrément delta = alpha T."""
    return taux_amortissement(t_pics, v_pics, offset)
