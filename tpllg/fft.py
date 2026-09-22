import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman


def calcule_DFT(temps, valeurs):
    """
    le temps doit évidemment être régulièrement espacé

    la fonction fft renvoie des complexes permettant de calculer la partie cosinus et sinus ou amplitude/phase
    L'amplitude est à redimensionnée par 2*tau, sauf pour la composante continue !
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#scipy.fft.fft

    pour les temps k allant de 0 à N-1, les fréquences renvoyées sont de 0 à N//2+1 pour rfft

    le calcul (optimisé) revient à :
        fourier[k] = np.sum(valeurs * np.exp(-2j * np.pi * k * np.arange(N)/N))
    """
    tau = temps[1] - temps[0]
    N = len(temps)
    fourier = np.abs(fft(valeurs)) * 2 * tau
    fourier[0] *= 0.5
    if N % 2 == 0:
        nmax = N//2
    else:
        nmax = (N+1)//2
    freq = fftfreq(N, d=tau)[:nmax]
    return freq, fourier[:nmax]


def spectre(temps, valeurs, p=6):
    """ calcul du spectre par TFD"""
    N = len(valeurs)
    te = temps[1] - temps[0]
    zeros = np.zeros(p * N)
    valeurs_fenetrees = np.concatenate((valeurs * blackman(N), zeros))
    spectre = np.absolute(np.fft.fft(valeurs_fenetrees)) * 2.0 / N / 0.42
    N = len(valeurs_fenetrees)
    freq = np.arange(N) * 1.0 / (N * te)
    return (freq, spectre)
