"""
Le diagramme de Bode automatique : la centrale génère elle-même une sinusoïde
sur sa sortie SA1, la relit sur EA0 et lit la sortie du filtre sur EA1,
fréquence par fréquence, puis trace gain et phase et les enregistre.

Le montage : un câble de SA1 à EA0 et à l'entrée du filtre, la sortie du
filtre sur EA1. SIMULATION = True remplace la centrale par un passe-bande
inverseur (f0 = 2 kHz, Q = 6, H0 = -5) dont on calcule la réponse, bruit et
quantification compris ; False mesure pour de bon. Voir doc/bode.md.

Ce script dérive de l'exemple « Diagramme de Bode » de Frédéric Legrand
(f-legrand.fr, CC BY-NC-SA 2.0 FR) : il est diffusé, comme le reste du dépôt,
sous CC BY-NC-SA 4.0, version ultérieure que la 2.0 FR autorise pour une
adaptation.
https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/pybode/pybode.html
La mesure de la fonction de transfert et le choix de l'échantillonnage, dans
tpllg.traitement, en viennent aussi.

@author: a. marchand, f. legrand
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.bode import tracer_bode
from tpllg.sysam import Sysam
from tpllg.traitement import choix_echantillonnage, fonction_transfert

SIMULATION = True

# Le montage
VOIES = [0, 1]  # EA0 : l'entrée du filtre (la sortie SA1 relue), EA1 : sa sortie
NOM_FICHIER = "bode_automatique.txt"

# Les fréquences : NB_POINTS_BODE entre 10^LOGFMIN et 10^LOGFMAX
LOGFMIN = 2
LOGFMAX = 4
NB_POINTS_BODE = 20
AMPLITUDE = 1.7  # V, la sinusoïde générée ; |H| × AMPLITUDE doit rester sous 10 V
CALIBRES_INIT = [5, 10]  # V, pour la première mesure, avant de connaître le gain : 0.2, 1, 5 ou 10

# L'échantillonnage, à ne pas toucher a priori
TE_MIN = Sysam.TE_MIN_SORTIE  # 0,2 µs : les sorties ne tournent qu'à un multiple de 0,2 µs
N_MAX = Sysam.n_max(len(VOIES), 1)  # la mémoire, partagée entre les 2 voies et la sortie
T_MAX = 1  # s, la durée maximale d'une acquisition
PER_MIN = 20  # le nombre minimal de périodes acquises
NP_MIN = 100  # le nombre minimal de points par période
DELAI_TRANSITOIRE = 5  # le nombre de périodes écartées au début : le régime transitoire


def reponse_simulee(t, e1, freq, calibres, H0=-5.0, f0=2000.0, Q=6.0, bruit=0.003):
    """Ce que la centrale rendrait : la sinusoïde e1 relue sur EA0, la réponse
    du passe-bande en régime établi sur EA1, un bruit et la quantification
    de chaque calibre."""
    H = H0 / (1 + 1j * Q * (freq / f0 - f0 / freq))
    sortie = abs(H) * abs(e1).max() * np.cos(2 * np.pi * freq * t + np.angle(H))
    rng = np.random.default_rng(round(freq))
    lignes = []
    for signal, calibre in zip((e1, sortie), calibres):
        calibre = Sysam.get_calibre(calibre)
        pas = 2 * calibre / 4096
        mesure = np.round((signal + rng.normal(0, bruit, t.size)) / pas) * pas
        lignes.append(np.clip(mesure, -calibre, calibre))
    return np.array([t, t]), np.array(lignes)


def acquisition(can, techant, N, amp, Np, freq, calibres):
    """Génère `amp` cos(2 pi n/Np) sur SA1 (Np points par période, N points en
    tout : un nombre entier de périodes, qui se répète sans saut) et acquiert
    les deux voies ; rend (t, e, s) sans les DELAI_TRANSITOIRE premières
    périodes."""
    e1 = amp * np.cos(2 * np.pi * np.arange(N) / Np)
    if SIMULATION:
        temps, tensions = reponse_simulee(np.arange(N) * techant, e1, freq, calibres)
    else:
        can.config_entrees(VOIES, calibres)
        can.config_echantillon(techant, N)
        temps, tensions = can.acquerir_avec_sorties(e1, None)
    n1 = int(DELAI_TRANSITOIRE * Np)
    return temps[0][n1:], tensions[0][n1:], tensions[1][n1:]


def mesure(can, freq, amp, calibres):
    """La fonction de transfert H à la fréquence la plus proche de `freq` qui
    fait un nombre entier de périodes dans l'acquisition. Rend (freq, H)."""
    techant, N = choix_echantillonnage(freq, TE_MIN, NP_MIN, PER_MIN, N_MAX, T_MAX)
    periodes = int(freq * N * techant)  # le nombre entier de périodes acquises
    freq = periodes / (N * techant)
    Np = N / periodes  # le nombre de points par période, pas forcément entier
    print(
        f"  acquisition : te = {techant:.1e} s, fe = {1 / techant:.1e} Hz, {Np:.1f} points par période, "
        f"N = {N}, durée {N * techant:.2f} s"
    )
    t, e, s = acquisition(can, techant, N, amp, Np, freq, calibres)
    H = fonction_transfert(t, e, s, freq)
    print(f"  mesure : G = {abs(H):.3g}, phi = {np.degrees(np.angle(H)):.1f}°")
    return freq, H


frequences = np.logspace(LOGFMIN, LOGFMAX, NB_POINTS_BODE)
frequences_mesurees = np.zeros(NB_POINTS_BODE)
H_mesures = np.zeros(NB_POINTS_BODE, dtype=complex)

with Sysam() as can:
    for i, f in enumerate(frequences):
        print(f"[{i:02d}] f = {f:.3g} Hz")
        # une première mesure, à l'amplitude et aux calibres de départ, donne le gain...
        f, H = mesure(can, f, AMPLITUDE, CALIBRES_INIT)
        # ... puis une seconde, à une amplitude qui ne sature pas la sortie et aux
        # calibres ajustés (10 % de marge), pour la précision
        amp = min(AMPLITUDE, AMPLITUDE / abs(H))
        f, H = mesure(can, f, amp, [amp * 1.1, amp * abs(H) * 1.1])
        frequences_mesurees[i], H_mesures[i] = f, H

gains, phases = abs(H_mesures), np.angle(H_mesures)
np.savetxt(
    NOM_FICHIER,
    np.column_stack([frequences_mesurees, gains, phases]),
    header="f (Hz)\tG\tphi (rad)",
    delimiter="\t",
)
print(f"mesures enregistrées dans {NOM_FICHIER}")

tracer_bode(frequences_mesurees, gains, phases, fichier="bode_automatique.pdf")
plt.show()
