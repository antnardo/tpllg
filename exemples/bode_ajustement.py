# -*- coding: utf-8 -*-
"""
Un diagramme de Bode relevé point par point — fréquence, amplitudes d'entrée
et de sortie, déphasage — ajusté par une fonction de transfert, gain et phase
ensemble, puis tracé. Voir doc/bode.md et doc/ajustement.md.

Les mesures ci-dessous sont celles d'un passe-bande actif du second ordre
(f0 ≈ 2 kHz) ; remplacez-les par les vôtres.
"""
import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curve_fit_complex, residus_complexes, resume_parametres
from tpllg.bode import tracer_bode

# 1. les mesures
f = [200, 300, 500, 700, 1000, 1300, 1500, 1700, 1800, 1900, 1950, 2000,
     2050, 2100, 2200, 2400, 2700, 3300, 5000, 7000, 10000, 15000, 20000]      # Hz
Ve = [1.0]*len(f)                                                              # V crête
Vs = [0.09, 0.12, 0.21, 0.30, 0.53, 0.88, 1.33, 2.18, 3.06, 4.21, 4.62, 5.13,
      4.85, 4.42, 3.16, 1.95, 1.24, 0.72, 0.39, 0.24, 0.16, 0.11, 0.08]        # V crête
phi = [-90, -94, -95, -92, -97, -103, -106, -114, -122, -152, -166, 176,
       154, 144, 126, 117, 104, 94, 96, 92, 87, 89, 89]                        # degrés, sortie - entrée

f = np.array(f, dtype=float)
H = np.array(Vs)/np.array(Ve)                # le module de la fonction de transfert
phi = np.radians(phi)                        # la phase, en radians pour l'ajustement


# 2. le modèle, et les valeurs de départ lues sur les mesures
def passe_bande(f, H0, f0, Q):
    return H0/(1 + 1j*Q*(f/f0 - f0/f))


PARAM_INIT = [-5, 2000, 6]     # H0 : |H| au maximum, signe donné par la phase (180° : négatif)
                               # f0 : la fréquence du maximum ; Q : f0 sur la largeur à -3 dB

# 3. l'ajustement simultané du gain et de la phase
pfit, err, chi2 = curve_fit_complex(passe_bande, f, norm=H, phase=phi, p0=PARAM_INIT,
                                    datayerrors=(0.03*H, np.radians(3)))   # 3 % sur |H|, 3° sur la phase
print(resume_parametres(("H0", "f0", "Q"), pfit, err, unites=("", "Hz", "")))
print("chi2 réduit = %.2f" % chi2)

# 4. les résidus : l'écart de chaque point à la courbe, sur le module et sur la phase
res_H, res_phi = residus_complexes(passe_bande, f, H, phi, pfit)
print("résidus relatifs sur |H| : écart-type %.1f %%, maximum %.1f %%"
      % (100*res_H.std(), 100*abs(res_H).max()))
print("résidus sur phi          : écart-type %.1f°, maximum %.1f°"
      % (res_phi.std(), abs(res_phi).max()))

# 5. la figure, gain au-dessus et phase au-dessous, l'ajustement en légende
tracer_bode(f, H, phi, passe_bande, pfit, err, noms=("$H_0$", "$f_0$", "$Q$"),
            unites=("", "Hz", ""), fichier="bode_ajustement.pdf")
plt.show()
