# -*- coding: utf-8 -*-
"""
Signaux acquis : repérer des fronts, découper une fenêtre, estimer une
fréquence, trouver les extremums d'une oscillation amortie et son décrément.

@author: a. marchand
"""
import numpy as np
from scipy.signal import find_peaks

__all__ = ["fronts_montants", "fronts_descendants", "front_utile",
           "frequence_pic", "fenetre", "extremums", "decrement_logarithmique"]


def fronts_montants(t, v):
    """Les instants des fronts montants d'un créneau.

    Seuil à mi-hauteur entre les niveaux bas et haut (percentiles 5 et 95)
    avec hystérésis d'un quart de l'amplitude : on passe à l'état haut
    au-dessus du seuil haut, à l'état bas au-dessous du seuil bas, et un front
    est une transition bas -> haut. L'instant est interpolé linéairement au
    passage à mi-hauteur. Rend (instants, (niveau bas, niveau haut)).
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    v_bas, v_haut = np.percentile(v, [5, 95])
    milieu = (v_bas + v_haut)/2
    marge = (v_haut - v_bas)/4
    etat_haut = v[0] > milieu
    fronts = []
    for i in range(1, len(v)):
        if not etat_haut and v[i] > milieu + marge:
            etat_haut = True
            j = i
            while j > 0 and v[j] > milieu:
                j -= 1
            if v[j + 1] != v[j]:
                tj = t[j] + (milieu - v[j])/(v[j + 1] - v[j])*(t[j + 1] - t[j])
            else:
                tj = t[j]
            fronts.append(tj)
        elif etat_haut and v[i] < milieu - marge:
            etat_haut = False
    return np.array(fronts), (v_bas, v_haut)


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
        if tf + fraction*periode < t_fin:
            return tf, periode
    return t_fronts[0], periode


def frequence_pic(v, te):
    """La fréquence du pic de la FFT d'un signal échantillonné à `te`, moyenne
    retirée : une première estimation de la fréquence d'une oscillation."""
    v = np.asarray(v, dtype=float)
    spectre = np.abs(np.fft.rfft(v - v.mean()))
    frequences = np.fft.rfftfreq(len(v), te)
    return frequences[np.argmax(spectre[1:]) + 1]


def fenetre(t, v, t_debut, duree):
    """Les points de (t, v) dans [t_debut, t_debut + duree[."""
    t = np.asarray(t, dtype=float)
    masque = (t >= t_debut) & (t < t_debut + duree)
    return t[masque], np.asarray(v, dtype=float)[masque]


def extremums(v, fe, f, offset=0.0):
    """Les indices des extremums d'une oscillation de fréquence `f`, en
    valeur absolue autour de `offset`, séparés d'au moins 0,4 période."""
    v = np.asarray(v, dtype=float)
    pics, _ = find_peaks(np.abs(v - offset), distance=max(1, int(0.4*fe/f)))
    return pics


def decrement_logarithmique(t_pics, v_pics, offset=0.0):
    """Le taux d'amortissement alpha d'une enveloppe exp(-alpha t), par
    régression du logarithme des amplitudes des extremums."""
    pente, _ = np.polyfit(np.asarray(t_pics, dtype=float),
                          np.log(np.abs(np.asarray(v_pics, dtype=float) - offset)), 1)
    return -pente
