# -*- coding: utf-8 -*-
"""
Incertitudes : loi normale, coefficient de Student, estimateurs d'une série de
mesures — le module incertitudes de dataanalysis (2018), fusionné ici.

@author: a. marchand
"""
import numpy as np
from scipy import special, stats

__all__ = ["loi_normale", "loi_normale_cumulee", "student_coef", "incertitudes"]


def loi_normale(x, m=0, s=1):
    '''normalisée
    - m=0 moyenne (centrage)
    - s=1 écart-type (largeur) - variance = s**2
    '''
    return 1/np.sqrt(2*np.pi*s)*np.exp(-(x-m)**2/(2*s**2))


def loi_normale_cumulee(t):
    ''' intégrale de -t à t de la loi normale
    ie proba d'avoir un tirage dans cet intervalle'''
    return special.erf(t/np.sqrt(2))


def student_coef(sigma, n):
    """
    Pour une variable T suivant la loi de Student à k=n-1 degrés de liberté, on définit tγk comme
    la quantité telle que la probabilité d’obtenir T > tγk soit égale à γ, ie cdf(t)=1-gamma ou sf(t)=gamma

    Le niveau de confiance alpha est tq on tire entre -t et t donc alpha = 1-2*gamma

    sigma = 1 : à alpha = 68% de niveau de confiance
    sigma = 2 : à 95%

    k = n-1 : degrés de liberté, n nb de points initiaux

    """
    alpha = special.erf(sigma/np.sqrt(2))  # 1 -> 0.68
    gamma = (1-alpha)/2  # proba de tirer au dessus du niveau de confiance
    k = n-1  # degrés de liberté
    student_law = stats.t(k)
    # inverse survival function (sf = 1-cdf, cdf = cumulative)
    return student_law.isf(gamma)


def incertitudes(liste, sigma=1, advanced=False, debug=False):
    '''
    pour un ensemble de données dans la liste
    renvoie estimateur de la moyenne et estimateur de l'écart à la moyenne
    sigma = 1 : à 68% de niveau de confiance
    sigma = 2 : à 95%
    (sigma : toute valeur flottante positive non nulle est acceptée)

    advanced : prend en compte le coef de student

    source : https://fr.wikipedia.org/wiki/Loi_de_Student#Application_:_intervalle_de_confiance_associ%C3%A9_%C3%A0_l%E2%80%99esp%C3%A9rance_d%E2%80%99une_variable_de_loi_normale_de_variance_inconnue
    '''
    if not isinstance(liste, np.ndarray):
        L = np.array(liste)
    else:
        L = liste
    N = len(L)
    assert N > 1, "La liste doit contenir plus d'un élément"
    m = L.mean()
    sigma_estim = L.std(ddof=1)  # estimateur sans biais de l'écart-type, sqrt(sum (x-m)^2/(N-1))
    if not advanced:
        delta = sigma_estim/np.sqrt(N)
        return m, delta, sigma_estim
    st = student_coef(sigma, N)
    delta = st*sigma_estim/np.sqrt(N)
    if debug:
        print(N, L)
        print(m, delta, sigma_estim, st)
    return m, delta, sigma_estim
