"""
Spectres d'amplitude d'un signal régulièrement échantillonné : une sinusoïde
d'amplitude A donne un pic de hauteur A, la composante continue vaut la
moyenne.

- calcule_DFT : la transformée de Fourier discrète brute, de 0 à fe/2 par pas
  de 1/durée ; exacte quand le signal contient un nombre entier de périodes,
  et alors les phases aussi ;
- spectre : une fenêtre de Blackman et des zéros ajoutés, pour l'amplitude
  d'une composante qui ne tombe pas sur la grille des fréquences.

numpy.fft seul : scipy.fft n'apporte rien à ces tailles.
"""

import numpy as np

from tpllg._interne import voie

__all__ = ["calcule_DFT", "spectre"]


def calcule_DFT(temps, valeurs, phases=False):
    """Le spectre d'amplitude : les fréquences positives, de 0 à fe/2 exclu
    par pas de 1/T (T la durée), et pour chacune l'amplitude en volts.

    La FFT rend N complexes c_k = sum(valeurs * exp(-2j pi k n/N)) ; la
    moitié positive du spectre est ramenée en amplitude par 2|c_k|/N, la
    composante continue par |c_0|/N. Avec phases=True, rend aussi les phases
    (radians) : le signal vaut sum(A_k cos(2 pi f_k (t - temps[0]) + phi_k)),
    c'est l'argument de c_k qui donne la phase, pas sa partie imaginaire.

    Exact quand le signal contient un nombre entier de périodes de chaque
    composante (f multiple de 1/T) ; sinon le pic s'étale et baisse, et
    spectre() fait mieux.
    """
    temps, valeurs = voie(temps, valeurs, minimum=2)
    n = len(valeurs)
    tfd = np.fft.rfft(valeurs)[: (n + 1) // 2]  # sans fe/2, dont l'amplitude se compte autrement
    amplitudes = np.abs(tfd) * 2 / n
    amplitudes[0] /= 2
    freq = np.fft.rfftfreq(n, d=temps[1] - temps[0])[: (n + 1) // 2]
    if phases:
        return freq, amplitudes, np.angle(tfd)
    return freq, amplitudes


def spectre(temps, valeurs, p=6):
    """Le spectre d'amplitude avec une fenêtre de Blackman et p*N zéros
    ajoutés : les fréquences de 0 à fe exclu, par pas de fe/((p+1)N), et
    l'amplitude en volts, normalisée par la fenêtre. Seule la moitié
    inférieure à fe/2 a un sens, l'autre en est le miroir : tracez jusqu'à
    fe/2. La fenêtre élargit chaque pic (sur 6 fe/N environ) mais en rend la
    hauteur juste à 0,5 % près, où qu'il tombe ; la composante continue vaut
    la moyenne.

    Fonction frequence() de Frédéric Legrand, « Mesure de déphasage »
    (f-legrand.fr, CC BY-NC-SA 2.0 FR), reprise ici :
    https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/dephasage/dephasage.html
    """
    temps, valeurs = voie(temps, valeurs, minimum=2)
    n = len(valeurs)
    fenetre = np.blackman(n)
    tfd = np.fft.fft(np.concatenate((valeurs * fenetre, np.zeros(p * n))))
    amplitudes = np.abs(tfd) * 2 / fenetre.sum()
    amplitudes[0] /= 2
    freq = np.arange(len(tfd)) / (len(tfd) * (temps[1] - temps[0]))
    return freq, amplitudes
