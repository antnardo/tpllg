# -*- coding: utf-8 -*-
"""
Diagramme de Bode : le tracé des mesures et, s'il y en a un, du modèle ajusté,
gain au-dessus et phase au-dessous.

@author: a. marchand
"""
import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import resume_parametres

__all__ = ["phase_0_360", "tracer_bode"]


def phase_0_360(phase):
    """La phase (radians) en degrés dans [0, 360[ : autour d'une résonance
    inverseuse elle passe par 180°, et +175° comme -175° y trouvent leur
    place sans saut."""
    return np.degrees(phase) % 360


def tracer_bode(f, norm, phase, modele=None, pfit=None, pcov=None, noms=None,
                unites=None, fichier=None, gain_log=True):
    """Les points (f, |H|, phi) et, si `modele` et `pfit` sont donnés, la
    courbe ajustée avec les valeurs des paramètres dans la légende.

    modele(f, *pfit) rend la fonction de transfert complexe ; `noms` et
    `unites` servent à la légende (par ex. ("$H_0$", "$f_0$", "$Q$") et
    ("", "Hz", "")). Écrit la figure dans `fichier` s'il est donné.
    """
    f = np.asarray(f, dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(f, norm, "o", label="mesures")
    ax1.set_ylabel("$|H|$")
    ax1.set_xscale("log")
    if gain_log:
        ax1.set_yscale("log")
    ax1.grid(which="both")
    ax2.plot(f, phase_0_360(phase), "o", label="mesures")
    ax2.set_ylabel(r"$\varphi$ ($^\circ$)")
    ax2.set_xlabel("$f$ (Hz)")
    ax2.grid(which="both")
    if modele is not None and pfit is not None:
        noms = noms if noms is not None else [f"p{i}" for i in range(len(pfit))]
        texte = resume_parametres(noms, pfit, pcov, unites)
        freq = np.geomspace(f.min(), f.max(), num=2000)
        H_ajuste = modele(freq, *pfit)
        ax1.plot(freq, np.abs(H_ajuste), "r", label="ajustement\n" + texte)
        ax2.plot(freq, phase_0_360(np.angle(H_ajuste)), "r", label="ajustement")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    if fichier is not None:
        plt.savefig(fichier)
    return fig
