"""
Les valeurs de départ d'un ajustement : tracer le modèle avec elles avant
d'ajuster, voir ce qu'un mauvais départ donne, et pourquoi — le chi2 en
fonction de f0 est une vallée étroite, et loin d'elle rien ne dit de quel
côté descendre. Voir doc/ajustement.md.

Un passe-bande du second ordre, vingt-trois mesures.
"""

import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import curve_fit_complex, resume_parametres
from tpllg.bode import phase_continue


def passe_bande(f, H0, f0, Q):
    return H0 / (1 + 1j * Q * (f / f0 - f0 / f))


def phase_deg(frequences, z):
    """La phase en degrés, continue le long des fréquences, comme tracer_bode la trace."""
    return phase_continue(frequences, np.angle(z))


# fmt: off
f = np.array([200, 300, 500, 700, 1000, 1300, 1500, 1700, 1800, 1900, 1950, 2000, 2050, 2100, 2200,
              2400, 2700, 3300, 5000, 7000, 10000, 15000, 20000.0])
H = np.array([0.09, 0.12, 0.21, 0.30, 0.53, 0.88, 1.33, 2.18, 3.06, 4.21, 4.62, 5.13, 4.85, 4.42,
              3.16, 1.95, 1.24, 0.72, 0.39, 0.24, 0.16, 0.11, 0.08])
phi = np.radians([-90, -94, -95, -92, -97, -103, -106, -114, -122, -152, -166, 176, 154, 144, 126,
                  117, 104, 94, 96, 92, 87, 89, 89])
# fmt: on
NOMS, UNITES = ("H0", "f0", "Q"), ("", "Hz", "")
f_fin = np.logspace(np.log10(150), np.log10(26000), 3000)


def ajuster(p0):
    """L'ajustement depuis p0, ou None s'il n'aboutit pas. Le modèle ne change
    pas quand f0 et Q changent de signe ensemble : on rend f0 > 0."""
    try:
        pfit = curve_fit_complex(passe_bande, f, H, phi, p0=p0, verbose=False).pfit
    except RuntimeError:
        return None
    return pfit * [1, -1, -1] if pfit[1] < 0 else pfit


# 1. le modèle tracé avec les valeurs de départ, puis ce que l'ajustement en fait
departs = (("un bon départ", [-5, 2000, 6]), ("un mauvais départ", [-1, 1500, 2]))
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey="row")
for colonne, (titre, p0) in enumerate(departs):
    pfit = ajuster(p0)
    if pfit is None:
        raise SystemExit(f"{titre} : l'ajustement depuis {p0} n'aboutit pas")
    resume = resume_parametres(NOMS, pfit, unites=UNITES).replace("\n", ", ")
    print(f"{titre}, p0 = {p0} : {resume}")
    haut, bas = axes[0, colonne], axes[1, colonne]
    haut.loglog(f, H, "o", markersize=4, label="les mesures")
    haut.loglog(
        f_fin, abs(passe_bande(f_fin, *p0)), "--", color="gray", label="le modèle aux valeurs de départ"
    )
    haut.loglog(f_fin, abs(passe_bande(f_fin, *pfit)), color="tab:orange", label="le modèle ajusté")
    haut.set_title(f"{titre} : p0 = {p0}\nrésultat : {resume}", fontsize=10)
    haut.legend(fontsize=8)
    modele = phase_deg(f_fin, passe_bande(f_fin, *pfit))
    reference = np.interp(np.log(f), np.log(f_fin), modele)  # chaque mesure au tour du modèle
    bas.semilogx(f, phase_continue(f, phi, reference), "o", markersize=4)
    bas.semilogx(f_fin, phase_deg(f_fin, passe_bande(f_fin, *p0)), "--", color="gray")
    bas.semilogx(f_fin, modele, color="tab:orange")
    bas.set_yticks(np.arange(-270, 91, 45))
    bas.set_xlabel("f (Hz)")
axes[0, 0].set_ylabel("|H|")
axes[1, 0].set_ylabel("phase (°)")
fig.tight_layout()
plt.savefig("ajustement_depart_trace.pdf")

# 2. la somme des carrés des écarts en fonction de f0, H0 et Q étant ceux du bon ajustement
pfit = ajuster([-5, 2000, 6])
if pfit is None:
    raise SystemExit("l'ajustement depuis [-5, 2000, 6] n'aboutit pas")
mesures = H * np.exp(1j * phi)
f0s = np.logspace(2, np.log10(30000), 2000)
ecarts = np.array([np.sum(abs(mesures - passe_bande(f, pfit[0], f0, pfit[2])) ** 2) for f0 in f0s])


def issue(f0):
    """Ce que donne l'ajustement parti de [-5, f0, 6] : 0 le bon minimum, 1 un autre
    résultat, sans erreur, 2 pas de convergence (RuntimeError)."""
    p = ajuster([-5, f0, 6])
    if p is None:
        return 2
    return 0 if abs(p[1] - pfit[1]) < 1 else 1


# un ajustement par fréquence de départ, trois cents départs de 100 Hz à 30 kHz
f0_depart = np.logspace(2, np.log10(30000), 300)
issues = np.array([issue(f0) for f0 in f0_depart])
autour = np.argmin(abs(f0_depart - pfit[1]))  # la plage sans trou autour de la résonance
gauche = droite = autour
while gauche > 0 and issues[gauche - 1] == 0:
    gauche -= 1
while droite < issues.size - 1 and issues[droite + 1] == 0:
    droite += 1
print(
    f"sur {issues.size} départs [-5, f0, 6] : {np.sum(issues == 0)} mènent au bon minimum, "
    f"{np.sum(issues == 1)} à un autre résultat sans erreur, {np.sum(issues == 2)} ne convergent pas"
)
print(f"tous les départs de {f0_depart[gauche]:.0f} à {f0_depart[droite]:.0f} Hz mènent au bon minimum")

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.loglog(f0s, ecarts, color="tab:blue", label="somme des carrés des écarts, H0 et Q fixés")
ax.plot(f, np.full(f.size, ecarts.max() * 2.5), "|", color="k", markersize=8, label="les fréquences mesurées")
bas = ecarts.min() / 2.5
for code, couleur, nom in (
    (0, "tab:green", "départ qui mène au bon minimum"),
    (1, "tab:red", "départ qui mène ailleurs, sans erreur"),
    (2, "gray", "départ qui ne converge pas"),
):
    ax.plot(
        f0_depart[issues == code],
        np.full(np.sum(issues == code), bas),
        "|",
        color=couleur,
        markersize=10,
        markeredgewidth=1.5,
        label=nom,
    )
for f0 in (200, 2000, 20000):
    resultat = ajuster([-5, f0, 6])
    hauteur = np.sum(abs(mesures - passe_bande(f, pfit[0], f0, pfit[2])) ** 2)
    arrivee = "ne converge pas" if resultat is None else f"arrive à f0 = {resultat[1]:.4g} Hz"
    ax.plot(f0, hauteur, "o", color="k")
    # l'étiquette sous le point, sauf au fond de la vallée où elle couvrirait la courbe
    cote = (
        {"xytext": (10, -4), "ha": "left"} if resultat is not None else {"xytext": (0, -26), "ha": "center"}
    )
    ax.annotate(
        f"départ à {f0:g} Hz :\n{arrivee}", (f0, hauteur), textcoords="offset points", fontsize=8, **cote
    )
    print(f"départ à f0 = {f0:g} Hz : {arrivee}")
ax.set_ylim(bas / 2, ecarts.max() * 5)
ax.set_xlabel("f0 (Hz)")
ax.set_ylabel("somme des carrés des écarts")
ax.legend(fontsize=8, loc="center left")
fig.tight_layout()
plt.savefig("ajustement_depart_vallee.pdf")

plt.show()
