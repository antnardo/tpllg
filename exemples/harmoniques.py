"""
Un signal périodique par ses harmoniques (tpllg.harmoniques) : la synthèse
d'un créneau, son passage dans un passe-bande qui isole une harmonique, et un
analyseur de spectre analogique — la valeur efficace en sortie d'un
passe-bande qu'on accorde fréquence par fréquence. Voir doc/spectres.md.

Rien n'est mesuré ici : tout se calcule harmonique par harmonique, ce qui en
fait l'exemple pour préparer un TP de filtrage ou un exercice.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.harmoniques import Signal, passe_bande, spectre_carre

UA = 2.5  # V, le créneau va de Um - UA à Um + UA
Um = 2.5  # V, sa composante continue
F_GBF = 170.0  # Hz, son fondamental

# 1. la synthèse : le créneau avec 1, 3, 5, 10 puis 100 harmoniques
t = np.linspace(0, 2 / F_GBF, 2000)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
for nmax in (1, 3, 5, 10, 100):
    creneau = Signal(f0=F_GBF, spectre=spectre_carre, moyenne=Um, amplitude=UA, nmax=nmax)
    ax1.plot(t * 1e3, creneau(t), label=f"{nmax} harmonique(s)")
ax1.set_xlabel("t (ms)")
ax1.set_ylabel("tension (V)")
ax1.legend(fontsize=8)
ax1.set_title("la somme des premières harmoniques", fontsize=10)
ax2.stem(creneau.frequences_0[:16], creneau.amplitudes_0[:16])
ax2.set_xlabel("f (Hz)")
ax2.set_ylabel("amplitude (V)")
ax2.set_title("son spectre : 4 UA/(n pi) sur les rangs impairs, et Um", fontsize=10)
fig.tight_layout()
plt.savefig("harmoniques_synthese.pdf")
# Parseval : Veff² = Um² + UA² pour le créneau entier
t_periode = np.linspace(0, 1 / F_GBF, 100000, endpoint=False)
veff_mesuree = np.sqrt(np.mean(creneau(t_periode) ** 2))
print(f"Veff du créneau à 100 harmoniques : {creneau.Veff:.4f} V par Parseval,", end=" ")
print(f"{veff_mesuree:.4f} V sur une période")
print(f"Veff du créneau entier : racine de Um² + UA² = {np.hypot(Um, UA):.4f} V")

# 2. le passe-bande accordé sur la troisième harmonique l'isole
creneau = Signal(f0=F_GBF, spectre=spectre_carre, moyenne=Um, amplitude=UA, nmax=100)
filtre = passe_bande(3 * F_GBF, Q=10, H0=5)
sortie = creneau.filtre(filtre)
print(f"en sortie : moyenne {sortie.moyenne:.3g} V, harmonique 3 de {sortie.amplitudes[2]:.3f} V,", end=" ")
print(f"Veff {sortie.Veff:.3f} V")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
t = np.linspace(0, 5 / F_GBF, 3000)
ax1.plot(t * 1e3, creneau(t), label="entrée")
ax1.plot(t * 1e3, sortie(t), label="sortie du passe-bande")
ax1.set_xlabel("t (ms)")
ax1.set_ylabel("tension (V)")
ax1.legend(fontsize=8)
ax1.set_title("le filtre accordé sur 3 f, Q = 10 : une sinusoïde à 3 f", fontsize=10)
f = np.linspace(1, 16 * F_GBF, 2000)
ax2.plot(creneau.frequences_0[:16], creneau.amplitudes_0[:16], "o", label="spectre de l'entrée")
ax2.plot(sortie.frequences_0[:16], sortie.amplitudes_0[:16], "s", label="spectre de la sortie")
ax2.plot(f, abs(filtre(f)), color="gray", linewidth=0.8, label="|H(f)|")
ax2.set_xlabel("f (Hz)")
ax2.set_ylabel("amplitude (V), et |H|")
ax2.legend(fontsize=8)
fig.tight_layout()
plt.savefig("harmoniques_filtrage.pdf")

# 3. un analyseur de spectre analogique : la valeur efficace en sortie d'un passe-bande
#    qu'on accorde de 10 Hz à 2 kHz ; plus Q est grand, mieux il sépare les harmoniques
accords = np.linspace(10, 2000, 1000)
fig, ax = plt.subplots(figsize=(8, 4.5))
for Q in (3, 10, 50):
    veff = [creneau.filtre(passe_bande(f0, Q)).Veff for f0 in accords]
    ax.plot(accords, veff, label=f"Q = {Q}")
rangs = creneau.n[creneau.amplitudes > 0][:6]
ax.plot(
    rangs * F_GBF, creneau.amplitudes[rangs - 1] / np.sqrt(2), "ko", label="valeurs efficaces des harmoniques"
)
ax.set_xlabel("fréquence d'accord du passe-bande (Hz)")
ax.set_ylabel("Veff en sortie (V)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig("harmoniques_analyseur.pdf")

plt.show()
