# -*- coding: utf-8 -*-
"""
Acquisition à la Sysam SP5 en une fonction, déclenchement compris, et
sauvegarde des voies acquises.

@author: a. marchand
"""
import numpy as np

__all__ = ["acquerir", "sauvegarder"]


def acquerir(voies, calibre, te, nbpoints, trigger=None):
    """Une acquisition : `voies` (par ex. [0, 1]), `calibre` (V, un pour
    toutes ou une liste), période d'échantillonnage `te` (s), `nbpoints`.

    trigger : None pour démarrer tout de suite, sinon (voie, seuil, pretrigger)
    ou (voie, seuil, pretrigger, montant) pour déclencher sur le passage de
    la voie par le seuil (montant=1 front montant, 0 descendant), en gardant
    `pretrigger` points avant le front.

    Rend (temps, tensions) comme tpllg.sysam : une ligne par voie.
    """
    from tpllg.sysam import Sysam
    with Sysam(voies, calibre) as can:
        can.config_echantillon(te, nbpoints)
        if trigger is not None:
            voie, seuil, pretrigger = trigger[:3]
            montant = trigger[3] if len(trigger) > 3 else 1
            can.config_trigger(voie, seuil, montant=montant, pretrigger=pretrigger)
        temps, tensions = can.acquerir()
    return temps, tensions


def sauvegarder(prefixe, voies, temps, tensions):
    """Un fichier texte par voie, `<prefixe>_EA<n>.txt`, deux lignes : temps
    et tensions — le format du TP2."""
    for ea, t, u in zip(voies, temps, tensions):
        np.savetxt(f"{prefixe}_EA{ea}.txt", [t, u])
