# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:35:03 2023

@author: a. marchand, f. legrand

Ce script dérive de l'exemple « Diagramme de Bode » de Frédéric Legrand
(f-legrand.fr, CC BY-NC-SA 2.0 FR) : il est diffusé, comme le reste du dépôt,
sous CC BY-NC-SA 4.0, version ultérieure que la 2.0 FR autorise pour une
adaptation.
https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/pybode/pybode.html
La mesure du gain (gain_std), l'interpolation par FFT et le choix de
l'échantillonnage, dans tpllg.traitement, en viennent aussi.

Bode automatique
Utilisation de la sortie pour générer un signal sinusoïdal
"""
import numpy as np
import matplotlib.pyplot as plt

from tpllg.sysam import Sysam
from tpllg.traitement import gain, choix_echantillonnage

# I/O
# Entrée sur EA0, sortie sur EA1
# Mettre un cable entre S1 et EA0
VOIES = [0, 1]  # entrée, sortie
SORTIE = 1  # ou 2
NOM_FICHIER = "filtreLC.txt"

# Boucle sur les fréquences entre 10^LOGFMIN et 10^LOGFMAX :
LOGFMIN = 2
LOGFMAX = 4
NB_POINTS_BODE = 20
AMPLITUDE = 1.7  # V pour le signal d'entrée généré par Sysam.
                 # Attention à ce que Hmax*AMPLITUDE ne dépasse pas 10V
CALIBRES_INIT = [2, 2]  # V pour EA0 et EA1
METHODE = 'std'  # ou 'fit'. Méthode de mesure du gain (cf docstring)

# Paramètres d'execution, ne pas toucher a priori
TE_MIN = Sysam.TE_MIN_SORTIE  # pas de temps d'échantillonnage minimal (Sysam)
                              # = 2e-7 lors de l'utilisation de la sortie
N_MAX = Sysam.N_MAX  # nb de points acquis max (Sysam) = 2**18
T_MAX = 1  # temps total max d'acquisition par courbe
PER_MIN = 20  # nb minimal de période dont faire l'acquisition (précision du spectre)
NP_MIN = 100  # nb minimal de points par période (shannon : >2)
DELAI_TRANSITOIRE = 5  # en nb de périodes, supprimées du signal à analyser
N_INTERPOLATION = 0  # pour la méthode 'std', lourd en calculs si >0, mais plus précis
DEFAUT_PLOT = False  # affiche le plot à chaque itération

# Variables d'execution
parametres = {
    'delai': DELAI_TRANSITOIRE,
    'method': METHODE,
    'ninter': N_INTERPOLATION
}

frequences_mesurees = np.zeros((NB_POINTS_BODE))
gains_mesures = np.zeros((NB_POINTS_BODE))
phases_mesurees = np.zeros((NB_POINTS_BODE))
frequences = np.logspace(LOGFMIN, LOGFMAX, NB_POINTS_BODE)

amp = AMPLITUDE


def acquisition(can: Sysam, techant, N, amp, Np, delai, calibres):
    """Génère un signal sinusoidal sur la sortie SORTIE,
    Fait l'acquisition en même temps sur les 2 voies VOIES
    L'entrée doit être sur VOIES[0], et la sortie sur VOIES[1]
    
    Renvoit (t, e, s) (trois ndarray) en supprimant les premiers points
    
        can: objet Sysam, ouvert
        techant: temps échantillonnage
        N: nb de points d'acquisition
        amp: amplitude du signal à générer sur la sortie SORTIE (1 ou 2)
        Np: nombre de points par période à générer
        delai: nb de période du début du signal à ne pas exporter (transitoire)
        calibres: liste des calibres sur les voies pour la lecture. En pratique,
        les valeurs entières immédiatement supérieures sont sélectionnées
    """
    # on crée le signal de sortie
    e1 = amp * np.cos(2 * np.pi * np.arange(N) / Np)
    # on lance la sortie et l'acquisition
    can.config_entrees(VOIES, calibres)
    can.config_echantillon(techant, N)
    sorties = (e1, 0) if SORTIE == 1 else (0, e1)
    t, signaux = can.acquerir_avec_sorties(*sorties)
    t, e, s = t[0], signaux[0], signaux[1]
    # on exporte après un nombre entier de périodes pour éviter le transitoire
    n1 = int(delai * Np)
    t, e, s = t[n1:] - t[n1], e[n1:], s[n1:]
    return t, e, s

def mesure_gain(can, freq, amp, calibres, delai, plot=DEFAUT_PLOT, **kwargs):
    """
    Génère un signal sinusoidal de sortie à freq
    Mesure les signaux d'entree
    Retourne les paramètres ajustés utiles pour le tracé du diagramme de Bode
    
    Méthodes de mesure des amplitudes, phases et fréquences
        method="std", par valeur efficace, avec une éventuelle interpolation
            ninter=4 par défaut, 0 auaucne interpolation, par une TF
        method="fit", par ajustement des valeurs
        method="fit_fft", par ajustement des valeurs avec un premier guess via fft

        can : objet Sysam ouvert
        freq : fréquence à  laquelle généer la sortie
        amp : amplitude du signal de sortie
        calibres : calibres à appliquer aux voies de mesure
        delai : nb de périodes de trnsitoire à ne pas prendre en compte
        plot : affiche les reevés temporels pendant l'acquisition
        **kwargs, donnés à la méthode de mesure gain()    
    """
    techant, N = choix_echantillonnage(freq, TE_MIN, NP_MIN, PER_MIN, N_MAX, T_MAX)
    P = int(freq * N * techant)  # période en nb de points
    freq = P / (N * techant)  # recalcul pour tomber juste
    Np = N / P  # nombre de points par période (float)
    print(f"  ACQUISITION {techant=:.1e}s, fe={1/techant:.1e}Hz,",
          f"points par période={Np:.1f}, {N=:.1e}, Ttotal={N*techant:.2f}s")
    t, e, s = acquisition(can, techant, N, amp, Np, delai, calibres)
    print(f"  ACQUISITION finie, mesure...")
    G, phi = gain(t, e, s, freq, Np, **kwargs)
    print(f"  MESURE {G=:.1e}, {phi=:.1e} rad")
    if plot: affiche_temporel(freq, t, e, s)
    return freq, G, phi


def affiche_temporel(freq, t, e, s):
    # faire une animation plutôt
    plt.figure()
    plt.plot(t, e, "b")
    plt.plot(t, s, "r")
    plt.grid()
    plt.ylim(-10, 10)
    plt.xlim(0, 10 / freq)
    plt.show()


# BOUCLE PRINCIPALE
with Sysam() as can:
    for i, f in enumerate(frequences):
        print(f"[{i:03d}] {f=:.1e}Hz")
        # on fait une première mesure
        f, G, phi = mesure_gain(
            can, f, amp, calibres=CALIBRES_INIT,
            **parametres
        )
        # puis une seconde pour plus de précision sur le calibre une fois l'amplitude déterminée
        amp = min(AMPLITUDE, AMPLITUDE / G)  # pour ne pas dépasser AMPLITUDE en sortie
        f, G, phi = mesure_gain(
            can, f, amp, calibres=[amp*1.1, amp*G*1.1],
            **parametres, plot=False
        )
        frequences_mesurees[i] = f
        gains_mesures[i] = G
        phases_mesurees[i] = phi
        print('', flush=True)


# Ajuste les phases à 2pi près pour garantir une continuité
phases_mesurees = np.unwrap(phases_mesurees)

# Export data
np.savetxt(
    NOM_FICHIER, np.array([frequences_mesurees, gains_mesures, phases_mesurees]).T, header="f\t G\t phi"
)

GdB = 20*np.log10(gains_mesures)

plt.figure()
plt.plot(frequences_mesurees, GdB, "b-")
plt.xscale("log")
plt.grid()
plt.xlabel("f (Hz)")
plt.ylabel("G (dB)")
plt.show()
plt.savefig("gain.pdf")

plt.figure()
plt.plot(frequences_mesurees, phases_mesurees, "b-")
plt.xscale("log")
plt.xlabel("f (Hz)")
plt.ylabel("phi (rad)")
plt.grid()
plt.show()
plt.savefig("phase.pdf")
