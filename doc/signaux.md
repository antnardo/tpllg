# Signaux acquis

`tpllg.signaux` repère dans un signal acquis ce qu'on veut y mesurer : les
fronts d'un créneau, une fenêtre de temps, la fréquence d'une oscillation,
ses extremums et son décrément. Ce sont les briques de l'exploitation d'un
régime transitoire, et la dernière section les assemble sur un cas complet.
Toutes prennent des tableaux numpy, ou des listes qu'elles convertissent, et
des temps en secondes.

## Sommaire

- [Les fronts d'un créneau](#les-fronts-dun-créneau)
- [Le front utile](#le-front-utile)
- [Une fenêtre](#une-fenêtre)
- [La fréquence d'une oscillation](#la-fréquence-dune-oscillation)
- [Les extremums](#les-extremums)
- [Le décrément logarithmique](#le-décrément-logarithmique)
- [Le régime libre après un front](#le-régime-libre-après-un-front)

## Les fronts d'un créneau

```python
from tpllg.signaux import fronts_montants, fronts_descendants

instants, (v_bas, v_haut) = fronts_montants(t, v)
instants, (v_bas, v_haut) = fronts_descendants(t, v)
```

| Argument | Sens |
| --- | --- |
| `t` | les instants, en secondes, croissants |
| `v` | le signal, même longueur ; un créneau, ou tout signal à deux niveaux |

Retour : les instants des fronts, dans un tableau, et les deux niveaux du
signal.

Les niveaux sont les percentiles 5 et 95 du signal, ce qui les rend
insensibles aux dépassements et au bruit ; le seuil est à mi-hauteur. Une
**hystérésis** d'un quart de l'amplitude évite qu'un signal bruité ne
compte plusieurs fronts au même passage : on passe à l'état haut quand le
signal dépasse le seuil plus la marge, à l'état bas quand il descend sous le
seuil moins la marge, et un front montant est une transition bas vers haut.
L'instant du front est **interpolé** linéairement entre les deux points qui
encadrent le passage à mi-hauteur : il est connu à mieux qu'un point
d'échantillonnage. `fronts_descendants` fait la même chose sur le signal
changé de signe.

Sur un créneau à 100 Hz de phase quelconque, échantillonné à 200 kHz
pendant 30 ms, avec un bruit de 10 mV :

```python
import numpy as np
from tpllg.signaux import fronts_montants, fronts_descendants, front_utile

fe = 200000.0
t = np.arange(6000)/fe
rng = np.random.RandomState(3)
v = np.sign(np.sin(2*np.pi*100*t + 2.3)) + rng.normal(0, 0.01, t.size)

fronts, (v_bas, v_haut) = fronts_montants(t, v)
print(np.round(fronts*1e3, 3), "ms ; niveaux", round(v_bas, 3), round(v_haut, 3))
fronts_d, _ = fronts_descendants(t, v)
print(np.round(fronts_d*1e3, 3), "ms")
```

```text
[ 6.338 16.337 26.338] ms ; niveaux -1.013 1.013
[ 1.338 11.338 21.337] ms
```

Les fronts sont à 10,00 ms les uns des autres, à 1 µs près, soit un
cinquième de la période d'échantillonnage : c'est l'interpolation. Les
niveaux dépassent 1 V de 13 mV, la marque du bruit sur les percentiles.

Un signal **sans créneau**, GBF débranché ou éteint, donne deux niveaux
confondus et des fronts partout, puisque le bruit passe sans cesse par un
seuil qui est au milieu de lui :

```python
bruit = rng.normal(0, 0.01, t.size)
fronts, niveaux = fronts_montants(t, bruit)
print(len(fronts), "fronts ; niveaux", np.round(niveaux, 4))
```

```text
625 fronts ; niveaux [-0.0163  0.0161]
```

Un script teste donc les niveaux avant d'aller plus loin : `v_haut - v_bas
< 0.2` V, par exemple, et un message qui dit quoi vérifier.

## Le front utile

```python
from tpllg.signaux import front_utile

t0, periode = front_utile(t, t_fronts, fraction=0.45)
```

| Argument | Sens |
| --- | --- |
| `t` | les instants de l'acquisition ; seul le dernier sert |
| `t_fronts` | les instants des fronts, ceux que `fronts_montants` rend |
| `fraction` | la part de période qu'il faut avoir derrière le front avant la fin de l'acquisition |

Retour : l'instant du premier front suivi d'au moins `fraction` de période
avant la fin de l'acquisition, et la **période** entre fronts, médiane des
écarts. Si aucun front ne convient, c'est le premier ; avec un seul front, la
période est ce qui reste jusqu'à la fin. Sans front, `ValueError`.

C'est ce qui choisit le front sur lequel on va travailler : après un front
montant du créneau on a une demi-période avant le front descendant, et l'on
veut cette demi-période entière, ou presque. `fraction=0.45` la demande
presque entière ; le front qui laisse moins de 4,5 ms d'un créneau à 100 Hz
derrière lui est écarté.

```python
t0, periode = front_utile(t, fronts)
print(t0, periode)
```

```text
0.006337520255049363 0.009999994947868324
```

## Une fenêtre

```python
from tpllg.signaux import fenetre

t_fen, v_fen = fenetre(t, v, t_debut, duree)
```

Rend les points de `(t, v)` dont l'instant est dans `[t_debut, t_debut +
duree[`, en tableaux. C'est un découpage, rien de plus, mais il revient à
chaque étape : la demi-période après le front, les dix premières
pseudo-périodes d'une oscillation, la fin de fenêtre où l'on lit l'offset.

```python
t_lib, v_lib = fenetre(t, v, t0, 0.45*periode)     # du front à presque le suivant
```

## La fréquence d'une oscillation

```python
from tpllg.signaux import frequence_pic

f = frequence_pic(v, te)
```

La fréquence du pic de la transformée de Fourier de `v`, moyenne retirée,
pour une période d'échantillonnage `te`. La résolution est `1/(N te)`, la
durée du signal : sur 5 ms de signal, 200 Hz. Ce n'est donc pas une mesure,
c'est une première estimation, celle qu'on donne comme valeur de départ à un
ajustement, et elle suffit à cela. Le signal doit contenir quelques
périodes, et l'oscillation doit dominer : sur un régime libre très amorti,
le pic est large mais toujours au bon endroit.

```python
fe, f0, Q = 200000.0, 2000.0, 6.0
t = np.arange(2000)/fe                                     # 10 ms
v = -1.5*np.exp(-np.pi*f0*t/Q)*np.sin(2*np.pi*f0*t) + 0.02
print(frequence_pic(v, 1/fe), "Hz, résolution", fe/2000, "Hz")
```

```text
2000.0 Hz, résolution 100.0 Hz
```

## Les extremums

```python
from tpllg.signaux import extremums

indices = extremums(v, fe, f, offset=0.0)
```

| Argument | Sens |
| --- | --- |
| `v` | le signal |
| `fe` | la fréquence d'échantillonnage |
| `f` | la fréquence de l'oscillation, celle de `frequence_pic` |
| `offset` | la valeur autour de laquelle le signal oscille |

Retour : les **indices** des extremums, maximums et minimums confondus, dans
`v`. Ce sont les pics de `|v - offset|` séparés d'au moins 0,4 période, ce
qui interdit à un point de bruit près d'un extremum de compter pour un
second. Les valeurs sont `v[indices]`, les instants `t[indices]`.

```python
pics = extremums(v, fe, f0, offset=0.02)
print(pics[:6], "…", len(pics), "extremums")
print(np.round(v[pics[:6]], 3))
```

```text
[ 24  74 124 174 224 274] … 40 extremums
[-1.3    1.036 -0.762  0.622 -0.443  0.377]
```

Cinquante points d'un extremum à l'autre, une demi-période à 200 kHz et
2 kHz. Le premier extremum est à 24 points et non à 25, parce que
l'oscillation amortie a son maximum un peu avant le quart de période.

## Le décrément logarithmique

```python
from tpllg.signaux import decrement_logarithmique

alpha = decrement_logarithmique(t_pics, v_pics, offset=0.0)
```

Le taux d'amortissement `alpha` d'une enveloppe `exp(-alpha t)`, par
régression linéaire du logarithme des amplitudes `|v_pics - offset|` sur les
instants `t_pics`. Pour un oscillateur du second ordre, `alpha = pi f0/Q`,
d'où `Q = pi f0/alpha`.

```python
alpha = decrement_logarithmique(t[pics], v[pics], offset=0.02)
print(alpha, "1/s ; Q =", np.pi*f0/alpha)
print("sans tenir compte de l'offset : Q =", np.pi*f0/decrement_logarithmique(t[pics], v[pics]))
```

```text
1047.1975511965977 1/s ; Q = 6.0
sans tenir compte de l'offset : Q = 17.0817506753877
```

L'offset compte : 20 mV oubliés sur une oscillation qui finit à quelques
dizaines de millivolts, et les derniers extremums paraissent trois fois trop
grands, la pente s'effondre, `Q` triple. On lit l'offset à la fin de la
fenêtre, là où l'oscillation est éteinte, et l'on ne garde pour la
régression que les extremums nettement au-dessus du bruit.

## Le régime libre après un front

Le cas complet, `exemples/regime_libre.py`. Un filtre passe-bande du second
ordre est attaqué par un créneau ; on veut la sonnerie qui suit un front,
l'ajuster, et en tirer la fréquence propre et le facteur de qualité. Le
créneau est sur EA0, la sortie sur EA1, et l'acquisition commence n'importe
où dans le créneau, que le déclenchement matériel soit configuré ou non.

La chaîne tient en cinq temps, et les fonctions ci-dessus y sont chacune à
leur place :

1. **les fronts montants** du créneau, et le premier qui laisse une
   demi-période derrière lui ;
2. **une première fenêtre**, jusqu'au front suivant, pour estimer la
   pseudo-fréquence par le pic de la FFT ;
3. **la fenêtre de l'ajustement**, dix pseudo-périodes au plus, où l'on lit
   l'offset en fin de fenêtre, les extremums, le décrément ;
4. **l'ajustement** du modèle, avec pour valeurs de départ ce qu'on vient
   d'estimer, l'instant du front compris ;
5. **la figure** : l'acquisition brute avec les fronts et la fenêtre, le
   régime libre ajusté avec son enveloppe, les résidus.

```python
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

# 1. l'acquisition
if SIMULATION:
    temps, tensions = acquisition_simulee(te, N)      # définie dans l'exemple
else:
    temps, tensions = acquerir(ENTREES, CALIBRE, te, N, trigger=(0, 0.0, 50))
t, ve, vs = temps[0], tensions[0], tensions[1]

# 2. les fronts montants du créneau, et le premier qui laisse une demi-période derrière lui
t_fronts, (v_bas, v_haut) = fronts_montants(t, ve)
if v_haut - v_bas < 0.2:
    raise SystemExit("pas de créneau sur EA0 (niveaux %.2f et %.2f V) : le GBF est-il branché ?"
                     % (v_bas, v_haut))
t0, periode = front_utile(t, t_fronts, fraction=0.45)

# 3. la fenêtre du régime libre et les valeurs de départ
t_demi, v_demi = fenetre(t, vs, t0, 0.45*periode)   # jusqu'au front suivant
f_pic = frequence_pic(v_demi, te)                   # le pic de la FFT
t_lib, v_lib = fenetre(t, vs, t0, min(0.45*periode, 10/f_pic))   # dix pseudo-périodes au plus
offset = v_lib[-len(v_lib)//5:].mean()              # la fin de fenêtre, où tout est amorti
pics = extremums(v_lib, fe, f_pic, offset)          # les indices des extremums
alpha = decrement_logarithmique(t_lib[pics], v_lib[pics], offset)
Q_estime = np.pi*f_pic/alpha


# 4. l'ajustement
def regime_libre(t, A, f0, Q, t0, v_off):
    tau = t - t0
    fp = f0*np.sqrt(1 - 1/(4*Q**2))
    return A*np.exp(-np.pi*f0*tau/Q)*np.sin(2*np.pi*fp*tau) + v_off


p0 = [1.2*(v_lib[pics[0]] - offset), f_pic, Q_estime, t0, offset]
pfit, pcov = curve_fit(regime_libre, t_lib, v_lib, p0=p0)
print(resume_parametres(("A", "f0", "Q", "t0", "v_off"), pfit, pcov, unites=("V", "Hz", "", "s", "V")))
residus = v_lib - regime_libre(t_lib, *pfit)
print("résidus : écart-type %.1f mV" % (residus.std()*1e3))
```

Ce que le script imprime, sur l'acquisition simulée (créneau ±1 V à
100 Hz, filtre à 1994,6 Hz et Q = 6,27, bruit de 10 mV, offset de 20 mV) :

```text
créneau : niveaux -1.01 et 1.01 V, fronts montants à [ 5.83 15.83 25.83] ms
front retenu : t0 = 0.00583 s, période du créneau 10.00 ms
pseudo-fréquence (pic de la FFT) : 2000 Hz
offset : 0.020 V ; 18 extremums, le premier à -1.41 V
décrément : alpha = 876 1/s, soit Q ≈ 7.2
A = -1.5913 ± 0.0020 V
f0 = 1994.50 ± 0.28 Hz
Q = 6.265 ± 0.011
t0 = 0.005829567 ± 0.000000098 s
v_off = 0.02008 ± 0.00033 V
résidus : écart-type 9.8 mV
```

Trois choses à lire. Les estimations sont grossières et suffisent : 2000 Hz
à 100 Hz près, `Q ≈ 7` pour 6,27, et l'ajustement, parti de là, retrouve
`f0` à 0,3 Hz et `Q` à 0,01. L'instant du front `t0` est un **paramètre
ajusté** et non une donnée : l'interpolation le donne à une fraction de
point, l'ajustement à un dixième de microseconde, et c'est ce qui rend la
méthode indifférente au déclenchement, matériel ou non. Et l'écart-type des
résidus, 9,8 mV, est celui du bruit mis dans la simulation : le modèle
explique tout ce qui n'est pas du bruit. Sur une vraie mesure, des résidus
qui oscillent à la pseudo-fréquence disent que la fenêtre commence trop tôt
(le front du créneau n'est pas instantané) ou que le modèle est incomplet.

La suite du script fait la figure ; la voici en entier dans
`exemples/regime_libre.py`, avec la simulation d'acquisition. Le signe de
`A` vient de la phase de l'oscillation par rapport au front : sur un
passe-bande inverseur, la sortie part vers le bas après un front montant, et
le premier extremum, qui donne la valeur de départ, est négatif.

Pour comparer au diagramme de Bode du même filtre, voir
[bode.md](bode.md) ; pour ce que valent les incertitudes rendues par
`curve_fit`, [ajustement.md](ajustement.md#ce-que-curve_fit-fait).
