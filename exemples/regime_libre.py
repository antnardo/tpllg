# -*- coding: utf-8 -*-
"""
Le régime libre d'un filtre après un front du créneau qui l'attaque : repérage
du front, fenêtre, valeurs de départ, ajustement, figure.

Le montage : un créneau ±E à 100 Hz sur EA0, la sortie d'un passe-bande du
second ordre (f0 ≈ 2 kHz, Q ≈ 6) sur EA1. SIMULATION = True fabrique une
acquisition telle que la centrale la rendrait — le créneau commence n'importe
où, la sortie sonne après chaque front, bruit, offset ; False acquiert pour de
bon. Voir doc/signaux.md.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tpllg.acquisition import acquerir
from tpllg.ajustement import ecarts_types, formater, resume_parametres
from tpllg.signaux import (decrement_logarithmique, extremums, fenetre, frequence_pic,
                           front_utile, fronts_montants)

SIMULATION = True
ENTREES = [0, 1]
CALIBRE = 5
fe = 200000.0              # une centaine de points par pseudo-période
T = 0.03                   # trois périodes du créneau : au moins un front montant complet
te, N = 1/fe, int(fe*T)


def acquisition_simulee(te, N, E=1.0, f_creneau=100.0, f0=1994.6, Q=6.27, H0=-5.0,
                        bruit=0.01, offset=0.02, graine=1):
    """Un créneau ±E de phase quelconque sur EA0, et sur EA1 la sortie du
    filtre, qui sonne après chaque front. Même forme que la centrale : deux
    tableaux (2, N), temps en secondes."""
    rng = np.random.RandomState(graine)
    t = np.arange(N)*te
    w0 = 2*np.pi*f0
    wp = w0*np.sqrt(1 - 1/(4*Q**2))
    phase = rng.uniform(0, 1)
    ve = E*np.sign(np.sin(2*np.pi*f_creneau*t + 2*np.pi*phase))
    vs = np.zeros(N)
    for k in range(-3, int(f_creneau*N*te) + 3):
        tk = (k/2 - phase)/f_creneau                    # un front ; montant si k pair
        tau = np.where(t > tk, t - tk, 0.0)
        signe = 1 if k % 2 == 0 else -1
        vs += signe*2*E*(H0/Q)*(w0/wp)*np.exp(-w0*tau/(2*Q))*np.sin(wp*tau)
    ve += rng.normal(0, bruit, N)
    vs += offset + rng.normal(0, bruit, N)
    return np.array([t, t]), np.array([ve, vs])


# 1. l'acquisition
if SIMULATION:
    temps, tensions = acquisition_simulee(te, N)
else:
    temps, tensions = acquerir(ENTREES, CALIBRE, te, N, trigger=(0, 0.0, 50))
t, ve, vs = temps[0], tensions[0], tensions[1]

# 2. les fronts montants du créneau, et le premier qui laisse une demi-période derrière lui
t_fronts, (v_bas, v_haut) = fronts_montants(t, ve)
if v_haut - v_bas < 0.2:
    raise SystemExit("pas de créneau sur EA0 (niveaux %.2f et %.2f V) : le GBF est-il branché ?"
                     % (v_bas, v_haut))
t0, periode = front_utile(t, t_fronts, fraction=0.45)
print("créneau : niveaux %.2f et %.2f V, fronts montants à %s ms" % (v_bas, v_haut, np.round(t_fronts*1e3, 2)))
print("front retenu : t0 = %.5f s, période du créneau %.2f ms" % (t0, periode*1e3))

# 3. la fenêtre du régime libre et les valeurs de départ
t_demi, v_demi = fenetre(t, vs, t0, 0.45*periode)   # jusqu'au front suivant
f_pic = frequence_pic(v_demi, te)                   # le pic de la FFT
t_lib, v_lib = fenetre(t, vs, t0, min(0.45*periode, 10/f_pic))   # dix pseudo-périodes au plus
offset = v_lib[-len(v_lib)//5:].mean()              # la fin de fenêtre, où tout est amorti
pics = extremums(v_lib, fe, f_pic, offset)          # les indices des extremums
alpha = decrement_logarithmique(t_lib[pics], v_lib[pics], offset)
Q_estime = np.pi*f_pic/alpha
print("pseudo-fréquence (pic de la FFT) : %.0f Hz" % f_pic)
print("offset : %.3f V ; %d extremums, le premier à %.2f V" % (offset, len(pics), v_lib[pics[0]]))
print("décrément : alpha = %.0f 1/s, soit Q ≈ %.1f" % (alpha, Q_estime))


# 4. l'ajustement
def regime_libre(t, A, f0, Q, t0, v_off):
    tau = t - t0
    fp = f0*np.sqrt(1 - 1/(4*Q**2))
    return A*np.exp(-np.pi*f0*tau/Q)*np.sin(2*np.pi*fp*tau) + v_off


p0 = [1.2*(v_lib[pics[0]] - offset), f_pic, Q_estime, t0, offset]
pfit, pcov = curve_fit(regime_libre, t_lib, v_lib, p0=p0)
A, f0, Q, t0_fit, v_off = pfit
sig = ecarts_types(pcov)
print(resume_parametres(("A", "f0", "Q", "t0", "v_off"), pfit, pcov, unites=("V", "Hz", "", "s", "V")))
residus = v_lib - regime_libre(t_lib, *pfit)
print("résidus : écart-type %.1f mV" % (residus.std()*1e3))

# 5. la figure : l'acquisition brute, le régime libre ajusté avec son enveloppe, les résidus
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 8), gridspec_kw={"height_ratios": [2, 3, 1.2]})
ax1.plot(t*1e3, ve, label="EA0 : créneau")
ax1.plot(t*1e3, vs, label="EA1 : sortie")
ax1.axvspan(t_lib[0]*1e3, t_lib[-1]*1e3, color="orange", alpha=0.2, label="fenêtre ajustée")
for tf in t_fronts:
    ax1.axvline(tf*1e3, color="gray", ls=":", lw=0.8)
ax1.set_xlabel("$t$ (ms)")
ax1.set_ylabel("tension (V)")
ax1.grid()
ax1.legend(fontsize=8, loc="upper right")
tt = np.linspace(t_lib[0], t_lib[-1], 4000)
ax2.plot((t_lib - t0_fit)*1e3, v_lib, ".", ms=3, label="mesures")
ax2.plot((tt - t0_fit)*1e3, regime_libre(tt, *pfit), "r", lw=1,
         label="ajustement : $f_0$ = " + formater(f0, sig[1], "Hz") + ", $Q$ = " + formater(Q, sig[2]))
env = abs(A)*np.exp(-np.pi*f0*(tt - t0_fit)/Q)
ax2.plot((tt - t0_fit)*1e3, v_off + env, "--", color="gray", lw=0.8, label="enveloppe")
ax2.plot((tt - t0_fit)*1e3, v_off - env, "--", color="gray", lw=0.8)
ax2.set_xlabel("$t - t_0$ (ms)")
ax2.set_ylabel("$v_s$ (V)")
ax2.grid()
ax2.legend(fontsize=8)
ax3.plot((t_lib - t0_fit)*1e3, residus*1e3, ".", ms=3)
ax3.axhline(0, color="k", lw=0.8)
ax3.set_xlabel("$t - t_0$ (ms)")
ax3.set_ylabel("résidus (mV)")
ax3.grid()
plt.tight_layout()
plt.savefig("regime_libre.pdf")
plt.show()
