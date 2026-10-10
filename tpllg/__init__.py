"""
tpllg — acquisition à la Sysam SP5 et exploitation des mesures en TP de physique.

Modules : sysam (la centrale, ou son simulateur sysam_factice), acquisition,
ajustement, incertitudes, montecarlo, signaux, bode, fft, traitement,
harmoniques, fichiers. On les importe un à un : `from tpllg.ajustement import
curvefit`. Un TP qui a besoin de fonctions à lui les pose dans un module
tpNN.py déposé dans la copie du paquet distribuée avec ses scripts — pas ici.

Python 3.8 ou plus, numpy, scipy et matplotlib, et rien d'autre : le paquet
se copie tel quel dans un dossier de TP. Les noms de 2026.9 fonctionnent
toujours, sans avertissement.
"""

__all__ = ["__version__"]

__version__ = "2026.10.1"
