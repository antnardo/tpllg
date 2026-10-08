"""
Ajuster un modèle qui n'a pas d'expression : la solution numérique d'une
équation différentielle, calculée par solve_ivp à chaque appel. curvefit ne
voit qu'une fonction modele(t, *paramètres) comme une autre. Voir
doc/ajustement.md.

Un pendule lâché de 2 rad (115°) : theta'' = -w0² sin theta. Aux grands
angles la période s'allonge, et une sinusoïde ajustée trouve la pulsation
des oscillations, pas w0 ; le modèle non linéaire retrouve w0.

SIMULATION = True fabrique les mesures (un angle relevé toutes les 20 ms,
à 0,02 rad près) ; remplacez-les par les vôtres.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from tpllg.ajustement import curvefit, resume_parametres

SIMULATION = True
U_THETA = 0.02  # rad, l'incertitude-type sur un angle


def pendule(t, w0, theta0):
    """L'angle aux instants t, lâché sans vitesse de theta0 : l'équation
    intégrée numériquement, avec une tolérance bien sous l'incertitude."""

    def derivee(_, etat):
        theta, omega = etat
        return [omega, -(w0**2) * np.sin(theta)]

    solution = solve_ivp(derivee, (0, t[-1]), [theta0, 0.0], t_eval=t, rtol=1e-9, atol=1e-9)
    return solution.y[0]


def sinusoide(t, w, theta0):
    return theta0 * np.cos(w * t)


if SIMULATION:
    rng = np.random.default_rng(4)
    t = np.arange(0, 6, 0.02)
    theta = pendule(t, 2 * np.pi, 2.0) + rng.normal(0, U_THETA, t.size)

# d'abord une sinusoïde, partie de la période lue sur le tracé (1,3 s) : elle donne la
# pulsation w des oscillations et l'amplitude ; puis le modèle non linéaire, parti de la
# formule de Borda, w0 ≈ w (1 + theta0²/16) — parti de w, 25 % trop bas, il s'égare
lineaire = curvefit(sinusoide, t, theta, [2 * np.pi / 1.3, theta[0]], datayerrors=U_THETA, verbose=False)
w, theta0 = lineaire.pfit
depart = [w * (1 + theta0**2 / 16), theta0]
non_lineaire = curvefit(pendule, t, theta, depart, datayerrors=U_THETA, verbose=False)
for nom, resultat, noms in (
    ("une sinusoïde            ", lineaire, ("w", "theta0")),
    ("theta'' = -w0² sin theta ", non_lineaire, ("w0", "theta0")),
):
    resume = resume_parametres(noms, resultat.pfit, resultat.err, ("rad/s", "rad")).replace("\n", "   ")
    print(f"{nom}: {resume}   chi2 réduit = {resultat.chi2:.2f}")

fig, (haut, bas) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
haut.plot(t, theta, ".", markersize=3, color="k", label="les mesures")
haut.plot(t, pendule(t, *non_lineaire.pfit), label="theta'' = -w0² sin theta, w0 ajustée")
haut.plot(t, sinusoide(t, *lineaire.pfit), "--", label="une sinusoïde ajustée")
haut.set_ylabel("theta (rad)")
haut.legend(fontsize=8, loc="lower left")
for modele, resultat, nom in ((pendule, non_lineaire, "non linéaire"), (sinusoide, lineaire, "sinusoïde")):
    bas.plot(t, theta - modele(t, *resultat.pfit), ".", markersize=3, label=f"résidus, {nom}")
bas.axhspan(-U_THETA, U_THETA, color="gray", alpha=0.3, label="± l'incertitude")
bas.set_xlabel("t (s)")
bas.set_ylabel("résidus (rad)")
bas.legend(fontsize=8, loc="lower left")
fig.tight_layout()
plt.savefig("ajustement_equadiff.pdf")

plt.show()
