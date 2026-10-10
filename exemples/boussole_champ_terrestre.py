"""
La composante horizontale du champ magnétique terrestre par la période d'une
boussole placée dans le champ de bobines : deux droites ajustées par
curvefit, un résultat mis en forme par formater. Voir doc/ajustement.md.

Une aiguille aimantée (moment magnétique mu, moment d'inertie J) écartée de
sa position d'équilibre dans un champ horizontal B oscille avec la période
T = 2 pi sqrt(J/(mu B)). On place la boussole au centre de bobines qui créent
un champ K I parallèle à la composante horizontale B_H du champ terrestre :

- même sens : B = B_H + K I, et 1/T1² = mu/(4 pi² J) (B_H + K I) est affine
  en I, de pente a = mu K/(4 pi² J), d'où K = 4 pi² a/(mu/J) ;
- sens inverse (I > B_H/K) : B = K I - B_H, période T2.

En combinant les deux mesures, mu/J disparaît :
B_H = K I (T2² - T1²)/(T1² + T2²).

Les mesures sont simulées par le script (bruit gaussien sur les périodes), à
partir de valeurs d'illustration de mu/J, de K et de B_H.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curvefit, formater

# Valeurs d'illustration
MU_SUR_J = 8e4  # A m²/(kg m²), rapport moment magnétique / moment d'inertie
K_BOBINES = 7.8e-4  # T/A, champ créé par les bobines par ampère
B_H_SIMULATION = 2.0e-5  # T, composante horizontale du champ terrestre
U_PERIODE = 5e-3  # s, incertitude-type sur chaque période mesurée


def periode(B):
    """Période d'oscillation de l'aiguille dans un champ B (T)."""
    return 2 * np.pi / np.sqrt(MU_SUR_J * B)


# --- Mesures simulées ---
rng = np.random.default_rng(2)
intensites = np.linspace(0.05, 0.5, 20)  # A
periodes = periode(B_H_SIMULATION + K_BOBINES * intensites)  # s, champs de même sens
periodes += U_PERIODE * rng.standard_normal(intensites.size)
periodes_inversees = periode(K_BOBINES * intensites - B_H_SIMULATION)  # s, sens opposés
periodes_inversees += U_PERIODE * rng.standard_normal(intensites.size)


def affine(x, a, b):
    return a * x + b


# --- Exploitation ---
# 1/T1² en fonction de I : la pente donne K
(a, b), (u_a, u_b), _ = curvefit(affine, intensites, 1 / periodes**2, p0=[1, 0])
K = 4 * np.pi**2 / MU_SUR_J * a  # T/A
print(f"pente a = {formater(a, u_a, 's⁻²/A')}, K = {formater(K, K * u_a / a, 'T/A')}")

# B_H à partir des deux sens du courant
B_H = K * intensites * (periodes_inversees**2 - periodes**2) / (periodes**2 + periodes_inversees**2)
u_B_H = B_H.std(ddof=1) / np.sqrt(len(B_H))
print(f"B_H = {formater(B_H.mean(), u_B_H, 'T')}")

fig, axes = plt.subplots(1, 3, num="Boussole", figsize=(13, 4))
axes[0].plot(intensites, periodes, "o", label="champs de même sens")
axes[0].plot(intensites, periodes_inversees, "o", label="champs de sens opposés")
axes[0].set_xlabel("intensité (A)")
axes[0].set_ylabel("période (s)")
axes[0].set_title("Périodes d'oscillation")
axes[0].legend()

axes[1].plot(intensites, 1 / periodes**2, "o", label="mesures")
axes[1].plot(intensites, affine(intensites, a, b), label=f"pente {a:.2f} s⁻²/A")
axes[1].set_xlabel("intensité (A)")
axes[1].set_ylabel("$1/T_1^2$ (s$^{-2}$)")
axes[1].legend()

axes[2].plot(intensites, B_H, "o")
axes[2].axhline(B_H.mean(), color="k", linestyle="--")
axes[2].set_xlabel("intensité (A)")
axes[2].set_ylabel("$B_H$ (T)")
axes[2].set_title(f"$B_H$ = {formater(B_H.mean(), u_B_H, 'T')}")
for ax in axes:
    ax.grid()
fig.tight_layout()
plt.show()
