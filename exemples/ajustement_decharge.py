# -*- coding: utf-8 -*-
"""
Une constante de temps : la décharge d'un condensateur acquise sur EA0,
déclenchée sur le passage par 2 V en descendant, puis ajustée par une
exponentielle avec les incertitudes de la mesure. Voir doc/ajustement.md.

SIMULATION = True fabrique l'acquisition telle que la centrale la rendrait ;
False acquiert pour de bon.
"""
import matplotlib.pyplot as plt
import numpy as np

from tpllg.acquisition import acquerir
from tpllg.ajustement import curvefit, resume_parametres

SIMULATION = True
VOIE, CALIBRE = 0, 5
te, N = 1e-5, 5000         # 50 ms à 100 kHz
SEUIL, PRETRIGGER = 2.0, 20
BRUIT = 0.005              # l'incertitude-type sur une tension : bruit et quantification, en volts


def acquisition_simulee(te, N, tau=8.3e-3, u_inf=0.04, bruit=BRUIT, graine=2):
    """La décharge telle que la centrale la rendrait : elle passe par SEUIL au
    point PRETRIGGER. Deux tableaux (1, N), temps en secondes."""
    rng = np.random.RandomState(graine)
    t = np.arange(N)*te
    u = u_inf + (SEUIL - u_inf)*np.exp(-(t - PRETRIGGER*te)/tau) + rng.normal(0, bruit, N)
    return np.array([t]), np.array([u])


def decharge(t, U0, tau, u_inf):
    return u_inf + (U0 - u_inf)*np.exp(-t/tau)


# 1. l'acquisition
if SIMULATION:
    temps, tensions = acquisition_simulee(te, N)
else:
    temps, tensions = acquerir([VOIE], CALIBRE, te, N, trigger=(VOIE, SEUIL, PRETRIGGER, 0))   # front descendant
t, u = temps[0], tensions[0]

# 2. les valeurs de départ, lues sur les données : la première tension, l'instant où elle
#    est divisée par e, la dernière tension
p0 = [u[0], t[np.argmin(abs(u - u[0]/np.e))], u[-1]]
print("valeurs de départ :",
      resume_parametres(("U0", "tau", "u_inf"), p0, unites=("V", "s", "V")).replace("\n", "   "))

# 3. l'ajustement, avec l'incertitude sur la tension
pfit, err, chi2 = curvefit(decharge, t, u, p0, datayerrors=BRUIT)
print(resume_parametres(("U0", "tau", "u_inf"), pfit, err, unites=("V", "s", "V")))
print("chi2 réduit = %.2f" % chi2)

# 4. la figure : l'acquisition et l'ajustement, puis les résidus
fig, (haut, bas) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
haut.plot(1e3*t, u, ".", markersize=1, color="tab:blue", label="l'acquisition, %d points" % N)
haut.plot(1e3*t, decharge(t, *p0), "--", color="gray", label="le modèle aux valeurs de départ")
haut.plot(1e3*t, decharge(t, *pfit), color="tab:orange",
          label="l'ajustement : tau = %.3f ± %.3f ms" % (1e3*pfit[1], 1e3*err[1]))
haut.set_ylabel("u (V)")
haut.legend(fontsize=9)
bas.axhspan(-1e3*BRUIT, 1e3*BRUIT, color="tab:orange", alpha=0.25, label="± l'incertitude sur la tension")
bas.plot(1e3*t, 1e3*(u - decharge(t, *pfit)), ".", markersize=1, color="tab:blue")
bas.axhline(0, color="tab:orange")
bas.set_xlabel("t (ms)")
bas.set_ylabel("u - modèle (mV)")
bas.legend(fontsize=9, loc="upper right")
fig.tight_layout()
plt.savefig("ajustement_decharge.pdf")

plt.show()
