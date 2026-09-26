# Diagramme de Bode et gain d'un filtre

`tpllg.bode` trace un diagramme de Bode, mesures et modèle ajusté ;
`tpllg.traitement` mesure le gain complexe d'un filtre sur ses signaux
d'entrée et de sortie, sans ajustement, et choisit l'échantillonnage pour
une fréquence donnée. Ensemble, avec la centrale, ils font un Bode
automatique.

## Sommaire

- [Tracer un diagramme de Bode](#tracer-un-diagramme-de-bode)
- [La phase entre 0 et 360°](#la-phase-entre-0-et-360)
- [Des mesures à la figure](#des-mesures-à-la-figure)
- [Mesurer le gain sur les signaux](#mesurer-le-gain-sur-les-signaux)
- [Choisir l'échantillonnage](#choisir-léchantillonnage)
- [Le Bode automatique](#le-bode-automatique)

## Tracer un diagramme de Bode

```python
from tpllg.bode import tracer_bode

fig = tracer_bode(f, norm, phase, modele=None, pfit=None, err=None, noms=None,
                  unites=None, fichier=None, gain_log=True)
```

| Argument | Sens |
| --- | --- |
| `f` | les fréquences, en Hz |
| `norm` | le module de la fonction de transfert à chaque fréquence, `Vs/Ve` |
| `phase` | la phase, en **radians** |
| `modele` | facultatif : la fonction de transfert complexe, `modele(f, *pfit)`, celle qu'on a ajustée |
| `pfit`, `err` | les paramètres ajustés et leurs incertitudes-types, ce que `curve_fit_complex` rend ; `err` est facultatif, il ajoute les incertitudes à la légende (la matrice de covariance de `curve_fit` convient aussi) |
| `noms`, `unites` | les noms des paramètres pour la légende, en LaTeX si l'on veut, `("$H_0$", "$f_0$", "$Q$")`, et leurs unités, `("", "Hz", "")` |
| `fichier` | facultatif : le nom du fichier où enregistrer la figure, `.pdf` ou `.png` |
| `gain_log` | `True` pour un module en échelle logarithmique (un vrai Bode), `False` pour une échelle linéaire |

Deux graphes l'un au-dessus de l'autre, aux fréquences en abscisse
logarithmique : le module en haut, la phase en bas, en degrés dans
`[0, 360[`. Les mesures sont des points ; si `modele` et `pfit` sont
donnés, la courbe ajustée s'ajoute, calculée sur deux mille fréquences entre
la plus basse et la plus haute mesurées, et la légende du gain porte les
paramètres, mis en forme par `resume_parametres`. La figure est rendue et
reste ouverte : `plt.show()` l'affiche, et l'on peut retoucher `fig.axes`
avant.

## La phase entre 0 et 360°

```python
from tpllg.bode import phase_0_360

degres = phase_0_360(phase)
```

La phase en radians devient des degrés dans `[0, 360[`. Ce choix évite le
saut de −180° à +180° autour d'une résonance inverseuse : un passe-bande à
amplificateur inverseur a une phase qui va de −90° loin sous `f0` à +90°
loin au-dessus en passant par 180°, et sur `]-180, 180]` elle saute d'un
bord à l'autre au voisinage de `f0`, où l'on mesure justement le plus de
points. Sur `[0, 360[` elle va de 270° à 90° en passant par 180°, d'un
seul tenant.

```python
>>> phase_0_360(np.radians([-175, 175, 0, -90, 90, 200]))
array([185., 175.,   0., 270.,  90., 200.])
```

Une phase mesurée en degrés sur un oscilloscope, sur `]-180, 180]`, se
convertit en radians par `np.radians` avant tout ; `tracer_bode` fait la
conversion inverse pour l'affichage.

## Des mesures à la figure

Le cas complet, `exemples/bode_ajustement.py` : un diagramme de Bode relevé
point par point à l'oscilloscope, sur un passe-bande actif du second ordre.
À chaque fréquence on note l'amplitude d'entrée, l'amplitude de sortie, le
déphasage de la sortie sur l'entrée.

```python
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
```

```text
Least square method
H0 = -5.075 ± 0.081
f0 = 1991.0 ± 2.7 Hz
Q = 6.43 ± 0.12
chi2 réduit = 1.20
résidus relatifs sur |H| : écart-type 3.5 %, maximum 11.1 %
résidus sur phi          : écart-type 2.7°, maximum 5.6°
```

Ces mesures ont été fabriquées avec 3 % de bruit sur les amplitudes et 3°
sur les phases, autour de `H0 = −5`, `f0 = 1994,6 Hz`, `Q = 6,27`. Ce sont
ces incertitudes-là qu'on a données à l'ajustement, et les résidus les
retrouvent ; le maximum de 11 % sur le module est un point à 0,09 V lu à
0,01 V près. Le χ² réduit de 1,2 dit que les incertitudes déclarées rendent
compte des écarts, donc que les incertitudes rendues sont crédibles : `f0`
sort à 3,6 Hz de la vraie valeur pour 2,7 Hz annoncés. La fiche
[ajustement.md](ajustement.md) dit ce que valent ces incertitudes, ce qui se
passe quand on ne les donne pas, et ce qu'il advient quand `PARAM_INIT` est
loin.

Sur une vraie mesure, `Ve` n'est pas constant : le GBF a une impédance de
sortie, et un filtre actif charge peu, un filtre passif davantage. On note
`Ve` à chaque fréquence, et le module est bien `Vs/Ve`.

## Mesurer le gain sur les signaux

Quand on a acquis l'entrée `e` et la sortie `s` d'un filtre attaqué par une
sinusoïde, on n'a pas besoin d'ajuster deux sinusoïdes pour avoir le gain
complexe :

```python
from tpllg.traitement import gain_std

G, phi = gain_std(t, e, s, Np=0, ninter=0)
```

| Argument | Sens |
| --- | --- |
| `t` | les instants ; ne servent pas au calcul, gardés pour la signature |
| `e`, `s` | les signaux d'entrée et de sortie, tableaux de même longueur, sinusoïdaux, sans transitoire |
| `Np` | le nombre de points par période, nécessaire au décalage d'un quart de période |
| `ninter` | un facteur d'interpolation par FFT, 0 pour aucune |

Retour : le module `G` et la phase `phi` en radians, de la sortie sur
l'entrée. Le module est le rapport des valeurs efficaces, `s.std()/e.std()`.
La phase vient de la moyenne du produit `s(t) (e(t) - j e(t - T/4))` : pour
`e = E cos(ωt)` et `s = GE cos(ωt + φ)`, `e(t) - j e(t - T/4)` vaut
`E e^{-jωt}`, et la moyenne du produit vaut `½ G E² e^{jφ}`, dont l'argument
est la phase. D'où `Np`, qui donne le quart de période en points. La méthode
est due à Frédéric Legrand, comme `interpolation_fft` et
`choix_echantillonnage` : les trois reprennent la fonction `mesure()` de son
exemple [Diagramme de Bode](https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/pybode/pybode.html).

Il faut un grand nombre de périodes, entières de préférence, pour que les
moyennes soient bonnes, et un `Np` juste : la fréquence d'échantillonnage
divisée par la fréquence du signal. Avec `ninter > 0`, les signaux sont
d'abord interpolés par FFT (`interpolation_fft`, qui ajoute des zéros aux
hautes fréquences du spectre) pour affiner le quart de période ; c'est plus
précis et beaucoup plus long, en `O(N²)`.

```python
fs, N, Np = 20000.0, 20000, 100
t = np.arange(N)/fs
e = 1.7*np.cos(2*np.pi*np.arange(N)/Np)
s = 0.5*1.7*np.cos(2*np.pi*np.arange(N)/Np - 1.0)      # G = 0,5, phi = -1 rad
print(gain_std(t, e, s, Np=Np))
```

```text
(0.4999999999999997, -1.000376923774711)
```

`gain(t, e, s, freq, Np, method="std", **kwargs)` est l'entrée générale
vers les méthodes de mesure ; seule `"std"` existe.

## Choisir l'échantillonnage

```python
from tpllg.traitement import choix_echantillonnage

te, N = choix_echantillonnage(freq, temin, Npmin, permin, Nmax, Tmax)
```

| Argument | Sens |
| --- | --- |
| `freq` | la fréquence du signal |
| `temin` | la période d'échantillonnage minimale de la carte, `Sysam.TE_MIN_SORTIE` avec une sortie |
| `Npmin` | le nombre de points par période souhaité |
| `permin` | le nombre de périodes souhaité, pour la précision des moyennes |
| `Nmax` | le nombre de points maximal, `Sysam.N_MAX` |
| `Tmax` | la durée maximale d'une acquisition |

Retour : la période d'échantillonnage, multiple de `temin`, et le nombre de
points. La règle : `Npmin` points par période sans descendre sous `temin`,
puis autant de points que `Tmax` et `Nmax` le permettent. La fonction
prévient si l'on a moins de `permin` périodes ou moins de `Npmin` points par
période, ce qui arrive aux deux bouts du spectre :

```python
for freq in (10, 1000, 1e5):
    print(freq, choix_echantillonnage(freq, 2e-7, 100, 20, 262144, 1))
```

```text
[WARNING] : nb de périodes faible 10.0<20
10 (0.001, 1000)
1000 (9.999999999999999e-06, 100000)
[WARNING] : nb de points par période faible 5.0<100
100000.0 (2e-07, 262144)
```

## Le Bode automatique

La centrale génère elle-même la sinusoïde sur sa sortie SA1, la relit sur
EA0 (un câble entre les deux) et lit la sortie du filtre sur EA1, fréquence
par fréquence, et trace le diagramme de Bode sans toucher au GBF. C'est
`exemples/Bode.py`, script dérivé de l'exemple
[Diagramme de Bode](https://www.f-legrand.fr/scidoc/docmml/sciphys/caneurosmart/pybode/pybode.html) de
Frédéric Legrand, dont voici l'ossature :

```python
import numpy as np
from tpllg.sysam import Sysam
from tpllg.traitement import choix_echantillonnage, gain_std

VOIES = [0, 1]                 # EA0 lit la sortie SA1, EA1 la sortie du filtre
AMPLITUDE = 1.7                # V ; |H| max × amplitude doit rester sous 10 V
frequences = np.logspace(2, 4, 20)

def mesure(can, freq, amplitude, calibres):
    te, N = choix_echantillonnage(freq, temin=Sysam.TE_MIN_SORTIE, Npmin=100,
                                  permin=20, Nmax=Sysam.N_MAX, Tmax=1)
    P = int(freq*N*te)         # période en points : on ajuste la fréquence pour tomber juste
    freq = P/(N*te)
    Np = N/P
    e1 = amplitude*np.cos(2*np.pi*np.arange(N)/Np)
    can.config_entrees(VOIES, calibres)
    can.config_echantillon(te, N)
    temps, (e, s) = can.acquerir_avec_sorties(e1, 0)
    n1 = int(5*Np)             # on écarte cinq périodes de transitoire
    G, phi = gain_std(temps[0][n1:], e[n1:], s[n1:], Np)
    return freq, G, phi

resultats = []
with Sysam() as can:
    for f in frequences:
        f, G, phi = mesure(can, f, AMPLITUDE, [2, 2])          # une première mesure
        amplitude = min(AMPLITUDE, AMPLITUDE/G)                # pour ne pas saturer la sortie
        f, G, phi = mesure(can, f, amplitude, [amplitude*1.1, amplitude*G*1.1])   # calibres ajustés
        resultats.append((f, G, phi))

f, G, phi = np.array(resultats).T
phi = np.unwrap(phi)           # continuité de la phase à 2π près
```

Quatre choix à comprendre. La fréquence demandée est **ajustée** pour
qu'un nombre entier de périodes tienne dans l'acquisition, ce qui rend les
moyennes de `gain_std` exactes ; c'est la fréquence ajustée qu'on garde.
Les **cinq premières périodes** sont écartées : c'est le régime transitoire
du filtre. Chaque fréquence est mesurée **deux fois** : une première fois
avec un calibre large, pour connaître le gain, une seconde avec l'amplitude
et les calibres adaptés, pour la précision. Et `np.unwrap` recolle la phase,
que `gain_std` rend sur `]-π, π]`.

Le script complet fait suivre les tracés et l'export, `np.savetxt(...,
header="f G phi")`, et le résultat s'ajuste ensuite exactement comme des
mesures à la main, `curve_fit_complex(modele, f, G, phi, p0=...)`, puis
`tracer_bode`.
