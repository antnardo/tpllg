# -*- coding: utf-8 -*-
"""
Le spectre d'un créneau, ses harmoniques, et le gain d'un filtre mesuré sur
chaque harmonique — la méthode « FFT » de mesure d'une fonction de transfert.
Voir doc/spectres.md.

SIMULATION = True fabrique l'entrée et la sortie d'un passe-bas du premier
ordre (fc = 510 Hz) attaqué par un créneau à 200 Hz ; False les acquiert.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tpllg.acquisition import acquerir
from tpllg.ajustement import resume_parametres
from tpllg.fft import calcule_DFT, spectre
from tpllg.traitement import detecte_maxima_secondaires, indices_plages, valeurs_correspondantes

SIMULATION = True
ENTREES = [0, 1]
CALIBRE = 5
fe = 100000.0
T = 0.1                    # 20 périodes du créneau
te, N = 1/fe, int(fe*T)


def acquisition_simulee(te, N, E=2.0, f1=200.0, fc=510.0, n_harmoniques=200, bruit=0.01, graine=2):
    """Un créneau ±E à f1 sur EA0, sa réponse par un passe-bas du premier ordre
    sur EA1 — harmonique par harmonique, 4E/(n pi) sur les impairs."""
    rng = np.random.RandomState(graine)
    t = np.arange(N)*te
    ve = np.zeros(N)
    vs = np.zeros(N)
    for n in range(1, n_harmoniques + 1, 2):
        H = 1/(1 + 1j*n*f1/fc)
        ve += 4*E/(n*np.pi)*np.sin(2*np.pi*n*f1*t)
        vs += 4*E/(n*np.pi)*abs(H)*np.sin(2*np.pi*n*f1*t + np.angle(H))
    ve += rng.normal(0, bruit, N)
    vs += rng.normal(0, bruit, N)
    return np.array([t, t]), np.array([ve, vs])


if SIMULATION:
    temps, tensions = acquisition_simulee(te, N)
else:
    temps, tensions = acquerir(ENTREES, CALIBRE, te, N)
t, ve, vs = temps[0], tensions[0], tensions[1]

# 1. les spectres bruts, en volts : une sinusoïde d'amplitude A donne un pic de hauteur A
freq, S_ve = calcule_DFT(t, ve)
freq, S_vs = calcule_DFT(t, vs)
df = freq[1] - freq[0]
fondamental = freq[np.argmax(S_ve)]
print("résolution %.1f Hz, fondamental à %.0f Hz, %.1f périodes acquises"
      % (df, fondamental, t[-1]*fondamental))

# 2. les harmoniques : un maximum par plage autour de chaque multiple du fondamental
plages = indices_plages(freq, fondamental, delta_freq=0.3*fondamental)
i_ve = detecte_maxima_secondaires(S_ve, plages, seuil=0.1)     # V : les harmoniques au-dessus du bruit
i_vs = detecte_maxima_secondaires(S_vs, plages, seuil=0.02)
print("%d harmoniques détectées sur l'entrée, %d sur la sortie" % (len(i_ve), len(i_vs)))
amplitude = np.mean(ve[ve > 0])                      # E, lu sur le créneau
for i in i_ve[:5]:
    n = int(round(freq[i]/fondamental))
    print("   n = %2d : %.3f V mesuré, 4E/(n pi) = %.3f V" % (n, S_ve[i], 4*amplitude/(n*np.pi)))

# 3. le gain, harmonique par harmonique, sur celles présentes des deux côtés
i_ve, i_vs = valeurs_correspondantes(i_ve, i_vs, delta_indices=int(0.1*fondamental/df) + 1)
f_gain = freq[i_ve]
gain = S_vs[i_vs]/S_ve[i_ve]
print("%d points de gain, de %.0f à %.0f Hz" % (len(f_gain), f_gain[0], f_gain[-1]))


# 4. un passe-bas du premier ordre ajusté sur ces points
def passe_bas(f, H0, fc):
    return H0/np.sqrt(1 + (f/fc)**2)


pfit, pcov = curve_fit(passe_bas, f_gain, gain, p0=[1, 500])
print(resume_parametres(("H0", "fc"), pfit, pcov, unites=("", "Hz")))

# 5. le spectre fin, avec fenêtre de Blackman, pour le tracé
freq_fin, S_fin = spectre(t, ve)

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 8))
ax1.plot(t*1e3, ve, label="EA0 : créneau")
ax1.plot(t*1e3, vs, label="EA1 : sortie du filtre")
ax1.set_xlim(0, 3e3/fondamental)
ax1.set_xlabel("$t$ (ms)")
ax1.set_ylabel("tension (V)")
ax1.grid()
ax1.legend(fontsize=8)
ax2.plot(freq, S_ve, label="entrée (TFD)")
ax2.plot(freq, S_vs, label="sortie (TFD)")
ax2.plot(freq[i_ve], S_ve[i_ve], "o", ms=4, label="harmoniques retenues")
ax2.plot(freq_fin, S_fin, lw=0.5, alpha=0.6, label="entrée (fenêtre de Blackman)")
ax2.set_xlim(0, 25*fondamental)
ax2.set_xlabel("$f$ (Hz)")
ax2.set_ylabel("amplitude (V)")
ax2.grid()
ax2.legend(fontsize=8)
ff = np.linspace(f_gain[0], f_gain[-1], 500)
ax3.plot(f_gain, gain, "o", label="gain mesuré par harmonique")
ax3.plot(ff, passe_bas(ff, *pfit), "r", label="passe-bas ajusté\n" + resume_parametres(("$H_0$", "$f_c$"), pfit, pcov, unites=("", "Hz")))
ax3.set_xlabel("$f$ (Hz)")
ax3.set_ylabel("$|H|$")
ax3.grid()
ax3.legend(fontsize=8)
plt.tight_layout()
plt.savefig("spectre_harmoniques.pdf")
plt.show()
