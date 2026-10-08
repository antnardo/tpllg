# -*- coding: utf-8 -*-
"""
Ajuster une fonction de transfert par son module et sa phase à la fois :
curve_fit_complex face à l'ajustement du module seul, les mesures dans le
plan complexe, et ce que les incertitudes de mesure changent. Voir
doc/ajustement.md.

Un passe-bande du second ordre, vingt-trois mesures.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tpllg.ajustement import curve_fit_complex, resume_parametres


def passe_bande(f, H0, f0, Q):
    return H0 / (1 + 1j * Q * (f / f0 - f0 / f))


def module_seul(f, H0, f0, Q):
    return abs(H0) / np.sqrt(1 + Q**2 * (f / f0 - f0 / f) ** 2)


def phase_deg(z):
    """La phase d'un complexe en degrés, entre 0 et 360 : celle d'un passe-bande
    inverseur va de 270° à 90° sans saut."""
    return np.degrees(np.angle(z)) % 360


f = np.array(
    [
        200,
        300,
        500,
        700,
        1000,
        1300,
        1500,
        1700,
        1800,
        1900,
        1950,
        2000,
        2050,
        2100,
        2200,
        2400,
        2700,
        3300,
        5000,
        7000,
        10000,
        15000,
        20000.0,
    ]
)
H = np.array(
    [
        0.09,
        0.12,
        0.21,
        0.30,
        0.53,
        0.88,
        1.33,
        2.18,
        3.06,
        4.21,
        4.62,
        5.13,
        4.85,
        4.42,
        3.16,
        1.95,
        1.24,
        0.72,
        0.39,
        0.24,
        0.16,
        0.11,
        0.08,
    ]
)
phi = np.radians(
    [
        -90,
        -94,
        -95,
        -92,
        -97,
        -103,
        -106,
        -114,
        -122,
        -152,
        -166,
        176,
        154,
        144,
        126,
        117,
        104,
        94,
        96,
        92,
        87,
        89,
        89,
    ]
)
NOMS, UNITES = ("H0", "f0", "Q"), ("", "Hz", "")
f_fin = np.logspace(np.log10(150), np.log10(26000), 3000)
f_zoom = np.linspace(1750, 2250, 400)

# module et phase ensemble, puis le module seul
pfit, err, chi2 = curve_fit_complex(passe_bande, f, H, phi, p0=[-5, 2000, 6], verbose=False)
print("module et phase ensemble :", resume_parametres(NOMS, pfit, err, UNITES).replace("\n", "   "))
pmod, pcov = curve_fit(module_seul, f, H, p0=[5, 2000, 6])
print(
    "le module seul           :",
    resume_parametres(("|H0|", "f0", "Q"), pmod, pcov, UNITES).replace("\n", "   "),
)

# 1. le module seul ne connaît ni le signe de H0 ni la phase, et place f0 moins bien
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
axes[0].loglog(f, H, "o", markersize=4, label="les mesures")
axes[0].loglog(f_fin, abs(passe_bande(f_fin, *pfit)), label="module et phase ensemble")
axes[0].loglog(f_fin, module_seul(f_fin, *pmod), "--", label="le module seul")
axes[0].set_ylabel("|H|")
axes[0].set_title("le module : les deux ajustements se valent", fontsize=10)
for ax, frequences in zip(axes[1:], (f_fin, f_zoom)):
    ax.plot(f, phase_deg(np.exp(1j * phi)), "o", markersize=4, label="les mesures")
    ax.plot(frequences, phase_deg(passe_bande(frequences, *pfit)), label="module et phase ensemble")
    # le module seul rend |H0| : la phase qu'il prédit dépend du signe qu'on lui donne
    ax.plot(
        frequences,
        np.degrees(np.angle(passe_bande(frequences, abs(pmod[0]), *pmod[1:]))),
        "--",
        color="tab:green",
        label="le module seul, si H0 > 0",
    )
    ax.plot(
        frequences,
        phase_deg(passe_bande(frequences, -abs(pmod[0]), *pmod[1:])),
        "--",
        color="tab:red",
        label="le module seul, si H0 < 0",
    )
    ax.set_ylabel("phase (°)")
axes[1].set_xscale("log")
axes[1].set_yticks(np.arange(-90, 271, 90))
axes[1].set_title("la phase : le signe de H0 se lit ici", fontsize=10)
axes[2].set_xlim(f_zoom[0], f_zoom[-1])
axes[2].set_ylim(100, 260)
axes[2].set_title("autour de la résonance : f0 = %.0f Hz ou %.0f Hz" % (pfit[1], pmod[1]), fontsize=10)
for ax in axes:
    ax.set_xlabel("f (Hz)")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig("ajustement_complexe_module.pdf")

# 2. dans le plan complexe : ce que l'ajustement rapproche, ce sont des points du plan
mesures = H * np.exp(1j * phi)
modele = passe_bande(f, *pfit)
cercle = passe_bande(f_fin, *pfit)
fig, ax = plt.subplots(figsize=(7, 6.2))
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.plot(cercle.real, cercle.imag, color="tab:orange", label="le modèle ajusté, de 150 Hz à 26 kHz")
for k in range(f.size):  # l'écart de chaque mesure au modèle, à la même fréquence
    ax.plot(
        [mesures[k].real, modele[k].real], [mesures[k].imag, modele[k].imag], color="tab:red", linewidth=1.5
    )
ax.plot([], [], color="tab:red", linewidth=1.5, label="les écarts que l'ajustement réduit")
ax.plot(mesures.real, mesures.imag, "o", color="tab:blue", markersize=4, label="les mesures, |H| exp(j phi)")
for k, decalage in (
    (0, (8, -10)),
    (7, (8, -4)),
    (9, (8, 4)),
    (10, (10, -2)),
    (11, (10, -3)),
    (13, (4, -16)),
    (15, (8, 2)),
    (22, (8, 6)),
):
    ax.annotate(
        "%g Hz : %+.0f°" % (f[k], np.degrees(phi[k])),
        (mesures[k].real, mesures[k].imag),
        textcoords="offset points",
        xytext=decalage,
        fontsize=8,
    )
ax.set_aspect("equal")
ax.set_xlim(-5.6, 1.9)
ax.set_xlabel("partie réelle de H")
ax.set_ylabel("partie imaginaire de H")
ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.46, 0.5))
fig.tight_layout()
plt.savefig("ajustement_complexe_plan.pdf")

# 3. avec les incertitudes de mesure : 3 % sur le module, 3° sur la phase
u_H, u_phi = 0.03 * H, np.radians(3)
pinc, einc, chi2 = curve_fit_complex(
    passe_bande, f, H, phi, p0=[-5, 2000, 6], datayerrors=(u_H, u_phi), verbose=False
)
print(
    "avec les incertitudes    :",
    resume_parametres(NOMS, pinc, einc, UNITES).replace("\n", "   "),
    "  chi2 réduit = %.2f" % chi2,
)
ajustements = (
    ("sans incertitudes", pfit, err, "tab:orange", "-"),
    ("avec 3 % et 3°", pinc, einc, "tab:green", "--"),
)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
barres = {"fmt": "o", "markersize": 3, "capsize": 2, "color": "tab:blue", "label": "les mesures"}
axes[0].errorbar(f, H, yerr=u_H, **barres)
axes[1].errorbar(f, H, yerr=u_H, **barres)
axes[2].errorbar(f, phase_deg(np.exp(1j * phi)), yerr=np.degrees(u_phi), **barres)
for nom, p, e, couleur, trait in ajustements:
    legende = "%s : %s" % (nom, resume_parametres(NOMS, p, e, UNITES).replace("\n", ", "))
    axes[0].plot(f_fin, abs(passe_bande(f_fin, *p)), trait, color=couleur, label=legende)
    axes[1].plot(f_zoom, abs(passe_bande(f_zoom, *p)), trait, color=couleur)
    axes[2].plot(f_fin, phase_deg(passe_bande(f_fin, *p)), trait, color=couleur)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_ylabel("|H|")
axes[0].set_title("le module, à 3 % près", fontsize=10)
axes[1].set_xlim(f_zoom[0], f_zoom[-1])
axes[1].set_ylim(2, 5.6)
axes[1].set_ylabel("|H|")
axes[1].set_title("autour de la résonance", fontsize=10)
axes[2].set_xscale("log")
axes[2].set_yticks(np.arange(90, 271, 45))
axes[2].set_ylabel("phase (°)")
axes[2].set_title("la phase, à 3° près", fontsize=10)
for ax in axes:
    ax.set_xlabel("f (Hz)")
traces, noms = axes[0].get_legend_handles_labels()
fig.legend(traces, noms, loc="lower center", ncol=1, fontsize=9)  # une légende pour les trois repères
fig.tight_layout(rect=(0, 0.16, 1, 1))
plt.savefig("ajustement_complexe_incertitudes.pdf")

plt.show()
