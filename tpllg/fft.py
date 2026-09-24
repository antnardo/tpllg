import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman


def calcule_DFT(temps, valeurs):
    """Le spectre d'amplitude d'un signal régulièrement échantillonné : les
    fréquences positives, de 0 à fe/2 exclu par pas de 1/T, et pour chacune
    l'amplitude en volts — une sinusoïde d'amplitude A donne un pic de
    hauteur A, la composante continue vaut la moyenne.

    La FFT rend N complexes ; la moitié positive du spectre est ramenée en
    amplitude par 2/N (sauf la composante continue, par 1/N). Le calcul
    revient à fourier[k] = sum(valeurs * exp(-2j pi k n/N)).
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
    """
    temps = np.asarray(temps, dtype=float)
    valeurs = np.asarray(valeurs, dtype=float)
    tau = temps[1] - temps[0]
    N = len(temps)
    fourier = np.abs(fft(valeurs)) * 2 / N
    fourier[0] *= 0.5
    if N % 2 == 0:
        nmax = N//2
    else:
        nmax = (N+1)//2
    freq = fftfreq(N, d=tau)[:nmax]
    return freq, fourier[:nmax]


def spectre(temps, valeurs, p=6):
    """Le spectre d'amplitude avec une fenêtre de Blackman et p*N zéros
    ajoutés : les fréquences de 0 à fe exclu, par pas de fe/((p+1)N), et
    l'amplitude en volts normalisée par la fenêtre. Seule la moitié
    inférieure à fe/2 a un sens, l'autre en est le miroir.

    Fonction frequence() de Frédéric Legrand, « Mesure de déphasage »
    (f-legrand.fr, CC BY-NC-SA 2.0 FR), reprise ici :
    https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/dephasage/dephasage.html
    """
    temps = np.asarray(temps, dtype=float)
    valeurs = np.asarray(valeurs, dtype=float)
    N = len(valeurs)
    te = temps[1] - temps[0]
    zeros = np.zeros(p * N)
    valeurs_fenetrees = np.concatenate((valeurs * blackman(N), zeros))
    spectre = np.absolute(np.fft.fft(valeurs_fenetrees)) * 2.0 / N / 0.42
    N = len(valeurs_fenetrees)
    freq = np.arange(N) * 1.0 / (N * te)
    return (freq, spectre)
