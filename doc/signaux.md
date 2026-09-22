# Signaux, Bode, spectres

Quatre modules pour ce qu'on fait d'un signal acquis : `tpllg.signaux` en
repère les fronts et les oscillations, `tpllg.bode` trace un diagramme de
Bode, `tpllg.fft` calcule un spectre, `tpllg.traitement` mesure le gain d'un
filtre et choisit l'échantillonnage — c'est lui que le Bode automatique
emploie.

## Le régime libre après un front

Cas d'usage : un filtre est attaqué par un créneau ; on veut la sonnerie qui
suit un front, l'ajuster, et en tirer la fréquence propre et le facteur de
qualité. Le créneau est sur EA0, la sortie sur EA1, et l'acquisition
commence n'importe où.

```python
import numpy as np
from scipy.optimize import curve_fit
from tpllg.acquisition import acquerir
from tpllg.ajustement import resume_parametres
from tpllg.signaux import (decrement_logarithmique, extremums, fenetre, frequence_pic,
                           front_utile, fronts_montants)

fe = 200_000.0
temps, tensions = acquerir([0, 1], 5, 1/fe, 6000, trigger=(0, 0.0, 50))
t, ve, vs = temps[0], tensions[0], tensions[1]

# 1. les fronts montants du créneau, et le premier qui laisse une demi-période derrière lui
t_fronts, (v_bas, v_haut) = fronts_montants(t, ve)
t0, periode = front_utile(t, t_fronts, fraction=0.45)

# 2. la fenêtre du régime libre, et une première estimation de ses paramètres
t_lib, v_lib = fenetre(t, vs, t0, 0.45*periode)
f_pic = frequence_pic(v_lib, 1/fe)                 # pic de la FFT
offset = v_lib[-len(v_lib)//5:].mean()             # fin de fenêtre
pics = extremums(v_lib, fe, f_pic, offset)         # indices des extremums
alpha = decrement_logarithmique(t_lib[pics], v_lib[pics], offset)
Q_estime = np.pi*f_pic/alpha

# 3. l'ajustement
def regime_libre(t, A, f0, Q, t0, v_off):
    tau = t - t0
    fp = f0*np.sqrt(1 - 1/(4*Q**2))
    return A*np.exp(-np.pi*f0*tau/Q)*np.sin(2*np.pi*fp*tau) + v_off

p0 = [v_lib[pics[0]] - offset, f_pic, Q_estime, t0, offset]
pfit, pcov = curve_fit(regime_libre, t_lib, v_lib, p0=p0)
print(resume_parametres(("A", "f0", "Q", "t0", "v_off"), pfit, pcov,
                        unites=("V", "Hz", "", "s", "V")))
```

Les fonctions, une à une :

| Fonction | Ce qu'elle rend |
| --- | --- |
| `fronts_montants(t, v)` | les instants des fronts montants d'un créneau, et ses deux niveaux. Seuil à mi-hauteur entre les percentiles 5 et 95 avec une hystérésis d'un quart de l'amplitude ; l'instant est interpolé au passage à mi-hauteur |
| `fronts_descendants(t, v)` | idem pour les fronts descendants |
| `front_utile(t, t_fronts, fraction=0.45)` | le premier front suivi d'au moins `fraction` de période avant la fin de l'acquisition, et la période entre fronts |
| `fenetre(t, v, t_debut, duree)` | les points de $[t_{debut}, t_{debut}+duree[$ |
| `frequence_pic(v, te)` | la fréquence du pic de la FFT, moyenne retirée |
| `extremums(v, fe, f, offset=0)` | les indices des extremums d'une oscillation de fréquence `f`, en valeur absolue autour de `offset`, séparés d'au moins 0,4 période |
| `decrement_logarithmique(t_pics, v_pics, offset=0)` | le taux $\alpha$ d'une enveloppe $e^{-\alpha t}$, par régression du logarithme des amplitudes |

Le déclenchement place le front vers le cinquantième point ; le repérage
dans les données ne s'y fie pas, et fonctionne aussi bien sans lui, pourvu
que l'acquisition dure plus d'une période du créneau. Un signal sans
créneau (GBF débranché) donne deux niveaux confondus : on le teste avant
d'aller plus loin, `v_haut - v_bas < 0.2` par exemple.

## Tracer un diagramme de Bode

```python
from tpllg.bode import tracer_bode

fig = tracer_bode(f, norm, phase, modele=None, pfit=None, pcov=None, noms=None,
                  unites=None, fichier=None, gain_log=True)
```

Deux graphes l'un au-dessous de l'autre, le module en haut (échelle log-log
par défaut, `gain_log=False` pour une échelle linéaire), la phase en bas, en
degrés dans $[0°, 360°[$ — ce qui évite le saut à ±180° autour d'une
résonance inverseuse. Avec `modele` et `pfit`, la courbe ajustée s'ajoute et
la légende porte les paramètres, mis en forme par `resume_parametres` avec
`noms` et `unites` ; `pcov` y ajoute les incertitudes. `fichier` enregistre
la figure.

```python
tracer_bode(f, H, phi, gain, pfit, pcov, noms=("$H_0$", "$f_0$", "$Q$"),
            unites=("", "Hz", ""), fichier="bode.pdf")
```

`phase_0_360(phase)` est la conversion employée, radians vers degrés dans
$[0, 360[$.

## Le spectre d'un signal

```python
from tpllg.fft import calcule_DFT, spectre

freq, amplitudes = calcule_DFT(t, u)      # la TFD brute, amplitudes en volts
freq, amplitudes = spectre(t, u, p=6)     # fenêtre de Blackman et zéros ajoutés
```

`calcule_DFT` rend les fréquences positives et l'amplitude de chaque
composante, normalisée pour qu'une sinusoïde d'amplitude $A$ donne un pic
de hauteur $A$ (la composante continue vaut la moyenne). `spectre` applique
d'abord une fenêtre de Blackman, qui écrase les lobes secondaires d'un signal
qui ne fait pas un nombre entier de périodes, et ajoute `p` fois la longueur
du signal en zéros, ce qui interpole le spectre finement. Il faut un temps
régulièrement échantillonné, ce que la centrale garantit.

```python
import matplotlib.pyplot as plt
from tpllg.acquisition import acquerir
from tpllg.fft import spectre

temps, tensions = acquerir([0], 1, 1/20000, 20000)
freq, amp = spectre(temps[0], tensions[0])
plt.plot(freq, amp)
plt.xlim(0, 2000)
plt.xlabel("f (Hz)")
plt.ylabel("amplitude (V)")
plt.show()
```

`tpllg.traitement.indices_plages(freq, fondamental, delta_freq)` et
`detecte_maxima_secondaires(valeurs, indices_bords, seuil)` servent ensuite à
relever les harmoniques : la première découpe l'axe des fréquences en plages
autour des multiples du fondamental, la seconde y cherche les maximums qui
dépassent un seuil.

## Mesurer le gain d'un filtre

Quand on a acquis l'entrée `e` et la sortie `s` d'un filtre attaqué par une
sinusoïde, `gain_std` mesure le gain complexe sans ajustement :

```python
from tpllg.traitement import gain_std

G, phi = gain_std(t, e, s, Np=100, ninter=0)
```

Le module est le rapport des valeurs efficaces, et la phase vient de la
moyenne de $s(t)\,\big(e(t) - j\,e(t - T/4)\big)$, qui vaut
$\tfrac12 G E^2 e^{j\varphi}$ : `Np` est le nombre de points par période,
nécessaire pour le décalage d'un quart de période. `ninter > 0` interpole les
signaux par FFT avant la mesure, plus précis et beaucoup plus long. Il faut
un grand nombre de périodes dans l'acquisition. `gain(t, e, s, freq, Np,
method="std", **kwargs)` est l'entrée générale ; seule la méthode `"std"`
existe.

## Le Bode automatique

Cas d'usage : la centrale génère elle-même la sinusoïde sur sa sortie, la
lit sur EA0 et lit la sortie du filtre sur EA1, fréquence par fréquence, et
trace le diagramme de Bode sans toucher au GBF. C'est `exemples/Bode.py`,
dont voici l'ossature :

```python
import numpy as np
from tpllg.sysam import Sysam
from tpllg.traitement import choix_echantillonnage, gain_std

VOIES = [0, 1]                 # EA0 lit la sortie S1 (un câble entre les deux), EA1 le filtre
AMPLITUDE = 1.7                # V ; attention à ce que |H| max × amplitude reste sous 10 V
frequences = np.logspace(2, 4, 20)

def mesure(can, freq):
    te, N = choix_echantillonnage(freq, temin=Sysam.TE_MIN_SORTIE, Npmin=100,
                                  permin=20, Nmax=Sysam.N_MAX, Tmax=1)
    P = int(freq*N*te)         # période en points, pour tomber juste
    freq = P/(N*te)
    Np = N/P
    e1 = AMPLITUDE*np.cos(2*np.pi*np.arange(N)/Np)
    can.config_entrees(VOIES, [2, 2])
    can.config_echantillon(te, N)
    t, (e, s) = can.acquerir_avec_sorties(e1, 0)
    n1 = int(5*Np)             # on écarte cinq périodes de transitoire
    G, phi = gain_std(t[0][n1:], e[n1:], s[n1:], Np)
    return freq, G, phi

with Sysam() as can:
    resultats = [mesure(can, f) for f in frequences]
```

`choix_echantillonnage(freq, temin, Npmin, permin, Nmax, Tmax)` rend la
période d'échantillonnage et le nombre de points pour une fréquence donnée :
au moins `Npmin` points par période sans descendre sous `temin`, au plus
`Nmax` points et `Tmax` secondes, et il prévient si l'on a moins de `permin`
périodes. Le script complet gère aussi le tracé et l'export, et fait suivre
les calibres à l'amplitude attendue.
