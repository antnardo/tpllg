"""
Mesurer une fonction de transfert : à une fréquence, par détection synchrone
de l'entrée et de la sortie (fonction_transfert) ; le choix de
l'échantillonnage du Bode automatique ; et, sur un spectre, les harmoniques
d'un signal périodique (indices_plages, detecte_maxima_secondaires,
valeurs_correspondantes).

Les anciens noms de 2026.9, `gain` et `gain_std`, fonctionnent toujours,
sans avertissement, et rendent (G, phi) comme avant, calculés par
fonction_transfert.

@author: a. marchand
"""

import math

import numpy as np

from tpllg.signaux import frequence_pic

__all__ = [
    "choix_echantillonnage",
    "detecte_maxima_secondaires",
    "fonction_transfert",
    "indices_plages",
    "valeurs_correspondantes",
]


def indices_plages(freq, fondamental, delta_freq):
    """Les plages d'indices du tableau `freq` (régulièrement espacé, depuis
    0) où chercher chaque harmonique de `fondamental` : une plage de largeur
    `delta_freq` centrée sur la case de n × fondamental, pour n = 1, 2… tant
    qu'elle est dans le spectre.

    Le centre de chaque plage est arrondi séparément, round(n f1/df) : la
    multiplier par n arrondie une fois accumulait l'erreur, et dès n = 25 les
    plages manquaient les harmoniques.

    Rend [(début, fin), (début, fin), ...] pour detecte_maxima_secondaires.
    """
    freq = np.asarray(freq, dtype=float)
    df = freq[1] - freq[0]
    n = len(freq)
    if not freq[0] < fondamental <= freq[-1]:
        raise ValueError(f"fondamental {fondamental:g} Hz hors du spectre ({freq[0]:g} à {freq[-1]:g} Hz)")
    demi = int(delta_freq / df / 2)
    plages = []
    rang = 1
    while True:
        centre = round((rang * fondamental - freq[0]) / df)
        if centre >= n:
            return plages
        plages.append((max(0, centre - demi), min(centre + demi, n)))
        rang += 1


def detecte_maxima_secondaires(valeurs, indices_bords, seuil=0.1):
    """Le maximum de `valeurs` dans chacune des plages d'indices_bords
    (indices_plages les donne), s'il dépasse `seuil`. Rend leurs indices.

    scipy.signal.find_peaks fait la même chose, mais se règle moins
    facilement sur un spectre d'harmoniques."""
    indices = []
    for debut, fin in indices_bords:
        valeurs_secondaires = valeurs[debut:fin]
        i = np.argmax(valeurs_secondaires)
        if valeurs_secondaires[i] > seuil:
            indices.append(debut + i)
    return indices


def valeurs_correspondantes(indexes1, indexes2, delta_indices):
    """Deux listes d'indices croissants (les harmoniques détectées sur
    l'entrée et sur la sortie) : ceux qui se correspondent à delta_indices
    près, appariés un à un, sans trou. Rend deux tableaux de même longueur."""
    i, j = 0, 0
    indices_final1 = []
    indices_final2 = []
    while i < len(indexes1) and j < len(indexes2):
        index1 = indexes1[i]
        index2 = indexes2[j]
        if abs(index1 - index2) <= delta_indices:
            # c'est le même, on enregistre et on avance
            indices_final1.append(index1)
            indices_final2.append(index2)
            i += 1
            j += 1
        elif index1 < index2:
            # il manque un 2, on avance sur 1
            i += 1
        else:
            # il manque un 1, on avance sur 2
            j += 1
    return np.array(indices_final1), np.array(indices_final2)


def fonction_transfert(t, e, s, freq=None):
    """La fonction de transfert complexe H = S/E à la fréquence `freq`,
    mesurée sur l'entrée e(t) et la sortie s(t) : |H| est le gain,
    np.angle(H) la phase de s par rapport à e, en radians.

    Détection synchrone : chaque signal est projeté sur exp(-2j pi freq t),
    avec une fenêtre de Hann, et H est le rapport des deux projections. Ce
    qui est commun aux deux — la fenêtre, un nombre non entier de périodes,
    une petite erreur sur freq — s'élimine du rapport ; la fenêtre rend
    négligeables les composantes continues et la fréquence négative. Il faut
    une dizaine de périodes au moins. Sans `freq`, elle est prise au pic de
    la FFT de e (frequence_pic), à 1/durée près, ce qui suffit.

    Remplace gain_std, d'après la fonction mesure() de Frédéric Legrand,
    « Diagramme de Bode » (f-legrand.fr, CC BY-NC-SA 2.0 FR), dont la phase,
    mesurée par un décalage d'un quart de période arrondi au point, était
    biaisée de 0,15 à 3° :
    https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/pybode/pybode.html
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    s = np.asarray(s, dtype=float)
    if not (t.ndim == 1 and t.shape == e.shape == s.shape):
        raise ValueError("t, e et s : trois tableaux 1D de même longueur")
    if freq is None:
        freq = frequence_pic(t, e)
    reference = np.hanning(t.size) * np.exp(-2j * np.pi * freq * (t - t[0]))
    return np.sum((s - s.mean()) * reference) / np.sum((e - e.mean()) * reference)


def choix_echantillonnage(freq, temin, Npmin, permin, Nmax, Tmax):
    """Le pas et le nombre de points pour acquérir une fréquence donnée.

    Reprend le calcul de la fonction mesure() de Frédéric Legrand, « Diagramme
    de Bode » (f-legrand.fr, CC BY-NC-SA 2.0 FR) :
    https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/pybode/pybode.html

    On veut au moins Npmin points par période, une période d'échantillonnage
    multiple de temin (et pas plus petite), et un temps total d'acquisition
    au plus Tmax. On vérifie qu'on acquiert au moins permin périodes et pas
    plus de Nmax points (Sysam.n_max le donne, sorties comprises), en le
    disant sinon.

    Rend techant (en s) et n, le nombre de points.
    """
    Np = min(Npmin, 1 / (temin * freq))
    # un multiple de temin ; la marge relative de 1e-9 évite que 49,9999999 soit pris pour 49
    techant = max(1, math.floor(1 / (Np * freq * temin) * (1 + 1e-9))) * temin
    n = min(Nmax, int(Tmax / techant))
    periodes = n * techant * freq  # nb de périodes acquises
    if periodes < permin:
        print(f"[WARNING] : nb de périodes faible {periodes:.1f}<{permin}")
    if 1 / (freq * techant) < Npmin:
        print(f"[WARNING] : nb de points par période faible {1 / (freq * techant):.1f}<{Npmin}")
    return techant, n


def gain_std(t, e, s, Np=0, ninter=0):
    """L'ancien nom (2026.9) : le gain G et la phase phi (radians) de s par
    rapport à e, comme avant, mais par fonction_transfert, qui n'a besoin ni
    de `Np` ni de `ninter` (ignorés)."""
    H = fonction_transfert(t, e, s)
    return abs(H), np.angle(H)


def gain(t, e, s, freq, Np=0, method="std", **kwargs):
    """L'ancien nom (2026.9) : (G, phi) à la fréquence `freq`, par
    fonction_transfert ; `Np` et les mots-clés sont ignorés, method='fit'
    n'a jamais existé."""
    if method != "std":
        raise NotImplementedError(f"gain(method={method!r}) n'a jamais été implémentée : seul 'std' l'était")
    H = fonction_transfert(t, e, s, freq)
    return abs(H), np.angle(H)
