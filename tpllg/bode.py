"""
Diagramme de Bode : le tracé des mesures et, s'il y en a un, du modèle ajusté,
gain au-dessus et phase au-dessous.

La phase est tracée continue : celle du modèle déroulée le long des
fréquences, en partant de la plus basse prise entre -45° et 315° ; chaque
mesure au tour le plus proche du modèle. Un passe-bas va ainsi de 0 à -90°,
un passe-bande de 90° à -90°, un passe-bande inverseur de 270° à 90° en
passant par 180° à la résonance — comme le repli sur [0, 360[ d'avant, mais
sans ses défauts : il faisait sauter de 360° la phase d'un passe-bande non
inverseur en pleine résonance, et dispersait entre 0 et 359° des mesures
voisines de 0°. (Les phases de départ usuelles sont 0, ±90° et 180° : la
frontière, à -45°, est à mi-chemin, loin du bruit d'une mesure ; à -180°
ou à -90°, elle passerait sur le départ d'un passe-haut du second ordre ou
d'un passe-bande inverseur, qui basculeraient d'une mesure à l'autre.)

@author: a. marchand
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import _radians, resume_parametres

__all__ = ["phase_continue", "phase_repliee", "tracer_bode"]


def phase_continue(f, phase, reference=None):
    """La phase (radians) en degrés, sans saut de 360° d'une fréquence à la
    suivante.

    Sans `reference` : déroulée dans l'ordre des fréquences, la plus basse
    prise entre -45° et 315° (exclu). Avec `reference` (en degrés, une
    valeur par fréquence, celle d'un modèle par exemple) : chaque phase prise
    au tour le plus proche de la référence.
    """
    f = np.asarray(f, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        return reference + (np.degrees(phase) - reference + 180) % 360 - 180
    ordre = np.argsort(f)
    deroulee = np.empty_like(phase)
    deroulee[ordre] = np.degrees(np.unwrap(phase[ordre]))
    premiere = deroulee[ordre[0]]
    return deroulee - 360 * np.floor((premiere + 45) / 360)


def phase_repliee(degres):
    """L'inverse de phase_continue : une phase en degrés, continue ou non,
    ramenée en radians dans ]-pi, pi], là où np.angle la rend et où
    curve_fit_complex, residus_complexes et tracer_bode l'attendent."""
    return np.angle(np.exp(1j * np.radians(np.asarray(degres, dtype=float))))


def phase_0_360(phase):
    """L'ancien nom (2026.9) : la phase (radians) en degrés dans [0, 360[,
    comme avant. phase_continue(f, phase) la déroule sans saut."""
    return np.degrees(phase) % 360


def tracer_bode(
    f, norm, phase, modele=None, pfit=None, err=None, noms=None, unites=None, fichier=None, gain_log=True
):
    """Les points (f, |H|, phi) et, si `modele` et `pfit` sont donnés, la
    courbe ajustée avec les valeurs des paramètres dans la légende.

    phase en radians (refusée au-delà de 2 pi en valeur absolue : elle
    serait en degrés), tracée en degrés ; modele(f, *pfit) rend la fonction
    de transfert complexe ; `err` les incertitudes-types des paramètres, ce
    que curve_fit_complex rend (une matrice de covariance de curve_fit
    convient aussi) ; `noms` et `unites` servent à la légende (par ex.
    ("$H_0$", "$f_0$", "$Q$") et ("", "Hz", "")). Écrit la figure dans
    `fichier` s'il est donné. Rend la figure.
    """
    f = np.asarray(f, dtype=float)
    phase = _radians(phase)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(f, norm, "o", label="mesures")
    ax1.set_ylabel("$|H|$")
    ax1.set_xscale("log")
    if gain_log:
        ax1.set_yscale("log")
    ax1.grid(which="both")
    ax2.set_ylabel(r"$\varphi$ ($^\circ$)")
    ax2.set_xlabel("$f$ (Hz)")
    ax2.grid(which="both")
    if modele is not None and pfit is not None:
        noms = noms if noms is not None else [f"p{i}" for i in range(len(pfit))]
        texte = resume_parametres(noms, pfit, err, unites)
        freq = np.geomspace(f.min(), f.max(), num=2000)
        H_ajuste = modele(freq, *pfit)
        phase_modele = phase_continue(freq, np.angle(H_ajuste))
        reference = np.interp(np.log(f), np.log(freq), phase_modele)
        ax2.plot(f, phase_continue(f, phase, reference), "o", label="mesures")
        ax1.plot(freq, np.abs(H_ajuste), "r", label="ajustement\n" + texte)
        ax2.plot(freq, phase_modele, "r", label="ajustement")
    else:
        ax2.plot(f, phase_continue(f, phase), "o", label="mesures")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    if fichier is not None:
        plt.savefig(fichier)
    return fig
