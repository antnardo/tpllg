# Incertitudes

`tpllg.incertitudes` estime une valeur et son incertitude sur une série de
mesures répétées, avec ou sans coefficient de Student ; `tpllg.montecarlo`
propage des incertitudes à travers un calcul par tirages aléatoires, sans
formule de dérivées partielles, selon la loi de chaque grandeur (normale,
uniforme pour une tolérance…), ajuste une droite ou un modèle sur chaque
tirage des mesures, et dit quelle grandeur pèse le plus dans l'incertitude
du résultat. Les incertitudes que rend un ajustement sont dans
[ajustement.md](ajustement.md).

## Sommaire

- [Une série de mesures](#une-série-de-mesures)
- [Le coefficient de Student](#le-coefficient-de-student)
- [La loi normale](#la-loi-normale)
- [Monte-Carlo : un Point](#monte-carlo--un-point)
- [Le nombre de tirages](#le-nombre-de-tirages)
- [D'autres lois : une tolérance, une résolution](#dautres-lois--une-tolérance-une-résolution)
- [Ce que vaut un tirage](#ce-que-vaut-un-tirage)
- [Quelle grandeur pèse le plus : les indices de Sobol](#quelle-grandeur-pèse-le-plus--les-indices-de-sobol)
- [Une droite sur tous les tirages, sans boucle](#une-droite-sur-tous-les-tirages-sans-boucle)
- [Un modèle quelconque par tirages](#un-modèle-quelconque-par-tirages)
- [Cas complets](#cas-complets)

## Une série de mesures

```python
from tpllg.incertitudes import incertitudes

m, delta, s = incertitudes(liste, sigma=1, advanced=False)
```

| Argument | Sens |
| --- | --- |
| `liste` | les mesures, au moins deux |
| `sigma` | le niveau de confiance, en nombre d'écarts-types de la loi normale : 1 pour 68,27 %, 2 pour 95,45 % ; `delta` est multipliée par `sigma` |
| `advanced` | `True` pour multiplier l'incertitude par le coefficient de Student au même niveau plutôt que par `sigma`, ce qui compte quand les mesures sont peu nombreuses |

Retour : la **moyenne** `m`, l'**incertitude** `delta` sur cette moyenne, et
l'**écart-type estimé** `s` de la série. Ce sont les trois estimateurs d'une
incertitude de **type A**, sur `N` mesures indépendantes `x_i` :

```text
m     = (1/N) somme des x_i                       la moyenne
s     = sqrt( somme des (x_i - m)² / (N - 1) )    l'écart-type de la série, estimé à N - 1
delta = s / sqrt(N)                               l'incertitude-type sur la moyenne
```

Rien qu'on ne puisse écrire soi-même en numpy, et c'est exactement ce que la
fonction fait :

```python
m = np.mean(mesures)
s = np.std(mesures, ddof=1)  # ddof=1 : la division par N - 1
delta = s / np.sqrt(len(mesures))
```

La division par `N - 1` rend sans biais la **variance** `s²` ; `s`
lui-même sous-estime un peu l'écart-type quand `N` est petit (de 6 % pour
cinq mesures), ce que le coefficient de Student prend en compte. Avec
`sigma=2`, `delta` est doublée, l'incertitude élargie à 95 % de la loi
normale (la version précédente ignorait `sigma` sans `advanced`) ; avec
`advanced`, `delta` est multipliée par le coefficient de Student au niveau
`sigma` (section suivante).

```python
import numpy as np
from tpllg.ajustement import formater, resume_parametres
from tpllg.incertitudes import incertitudes

mesures = [9.78, 9.81, 9.85, 9.79, 9.83]  # g, en m/s², cinq fois
m, delta, s = incertitudes(mesures)
print(m, delta, s)
print(formater(m, delta, "m/s²"))
m2, delta2, s2 = incertitudes(mesures, sigma=2)
print(formater(m2, delta2, "m/s²"), "à 95 %, loi normale")
m3, delta3, s3 = incertitudes(mesures, sigma=2, advanced=True)
print(formater(m3, delta3, "m/s²"), "à 95 %, avec Student")
print(resume_parametres(("g",), [m], [delta], ("m/s²",)))
```

```text
9.812 0.012806248474865799 0.028635642126552934
9.812 ± 0.013 m/s²
9.812 ± 0.026 m/s² à 95 %, loi normale
9.812 ± 0.037 m/s² à 95 %, avec Student
g = 9.812 ± 0.013 m/s²
```

`s` est la dispersion d'**une** mesure ; `delta` est ce qu'on gagne à en
faire `N`, et c'est `delta` qu'on écrit derrière le `±`. Deux fonctions de
`tpllg.ajustement` mettent en forme, avec deux chiffres significatifs sur
l'incertitude et la valeur arrondie au même rang : `formater` pour une
grandeur, `resume_parametres` pour plusieurs grandeurs nommées, une ligne
chacune — c'est la même mise en forme, la seconde appelle la première
(voir [ajustement.md](ajustement.md#présenter-un-résultat)).

`exemples/incertitudes_serie.py` trace les trois estimateurs sur deux
séries : les cinq mesures ci-dessus, puis cinquante de même dispersion — les
cinq mêmes et quarante-cinq autres, simulées. D'abord sur l'histogramme des
mesures, où `m ± s` et `m ± delta` sont deux bandes verticales autour de la
moyenne :

![L'histogramme de cinq mesures puis de cinquante, la loi normale de mêmes moyenne et écart-type, la moyenne en trait, m ± s en bande orange et m ± delta en bande rouge](images/incertitudes_serie_histogramme.png)

Puis sur les mesures dans l'ordre où elles ont été faites, les mêmes bandes
à l'horizontale :

![Les mesures en fonction de leur numéro, cinq puis cinquante, la moyenne en trait horizontal, m ± s en bande orange et m ± delta en bande rouge](images/incertitudes_serie_indices.png)

```text
5 mesures  : m = 9.8120  s = 0.0286  delta = 0.0128  soit g = 9.812 ± 0.013 m/s²
50 mesures : m = 9.8102  s = 0.0257  delta = 0.0036  soit g = 9.8102 ± 0.0036 m/s²
```

De cinq à cinquante mesures, la bande `m ± s` garde sa largeur, aux
fluctuations près : c'est là que tombe une mesure, deux fois sur trois si la
loi est normale, et en faire davantage n'y change rien. La bande `m ± delta`
rétrécit, de 0,013 à 0,004 m/s² : elle ne dit pas où tombent les mesures —
sur les cinquante, neuf seulement y sont — mais où est la moyenne, et c'est
elle le résultat.

## Le coefficient de Student

```python
from tpllg.incertitudes import student_coef

t = student_coef(sigma, n)
```

La règle : une incertitude de **type A**, estimée sur `N` mesures répétées,
s'écrit `delta = t × s/sqrt(N)`, où `t` est le coefficient de Student au
niveau de confiance voulu, à `k = N - 1` degrés de liberté. Le coefficient
ne corrige pas `s` — l'écart-type de la série reste `np.std(mesures,
ddof=1)` — il élargit l'intervalle de confiance sur la **moyenne**, parce que
`s` est lui-même estimé sur ces mêmes `N` mesures, et d'autant moins bien
connu que `N` est petit. `sigma` dit le niveau, en nombre d'écarts-types
d'une loi normale : 1 pour 68,27 %, 2 pour 95,45 % (pour 95 % tout rond,
`sigma = 1.96`). `t` tend vers `sigma` quand `N` grandit. `n` doit être un
entier au moins égal à 2 (`n` peut être un tableau) et `sigma` strictement
positif, sans quoi `ValueError` rappelle l'ordre des arguments :
`student_coef(7, 0.95)`, les deux inversés, rendait `nan` en silence.

| `N` | à 68,27 % (`sigma=1`) | à 95,45 % (`sigma=2`) |
| --- | --- | --- |
| 2 | 1,84 | 13,97 |
| 3 | 1,32 | 4,53 |
| 5 | 1,14 | 2,87 |
| 10 | 1,06 | 2,32 |
| 30 | 1,02 | 2,09 |
| 100 | 1,005 | 2,03 |

`exemples/incertitudes_student.py` trace `t` en fonction de `N` aux deux
niveaux, avec les valeurs du tableau :

![Le coefficient de Student en fonction du nombre de mesures, de 2 à 100, à 68 % et à 95 % : deux courbes qui descendent vers 1 et 2, leurs valeurs pour la loi normale](images/incertitudes_student.png)

Sur les cinq mesures de `g` ci-dessus, à 95,45 % : `t = 2.87`, et
`delta = 2.87 × 0.0286/sqrt(5) = 0.037 m/s²`, ce que `incertitudes(mesures,
sigma=2, advanced=True)` rend. Deux mesures ne disent presque rien de
l'écart-type : à 95,45 % le facteur est 14. À partir d'une dizaine de mesures,
la correction est de quelques pour cent à 68 %, et l'on peut s'en passer en
incertitude-type.

## La loi normale

```python
from tpllg.incertitudes import loi_normale, loi_normale_cumulee

y = loi_normale(x, m=0, s=1)  # la densité, de moyenne m et d'écart-type s
p = loi_normale_cumulee(t)  # la probabilité d'un tirage dans [-t s, +t s]
```

```python
>>> [round(loi_normale_cumulee(k), 4) for k in (1, 2, 3)]
[0.6827, 0.9545, 0.9973]
```

Les 68 %, 95 % et 99,7 % à un, deux et trois écarts-types. `loi_normale`
sert aux tracés, pour superposer la gaussienne à un histogramme.

`exemples/incertitudes_loi_normale.py` trace les deux sur le même graphe.
La valeur du cumul en `t` est l'aire sous la densité entre `-t` et `+t` :

![La densité de la loi normale, son aire coloriée entre -1 et 1, -2 et 2, -3 et 3 écarts-types, et le cumul qui monte de 0 à 1 en passant par 68,27 %, 95,45 % et 99,73 %](images/incertitudes_loi_normale.png)

## Monte-Carlo : un Point

La méthode (GUM, supplément 1) : on donne à chaque grandeur mesurée une
loi, gaussienne centrée sur sa valeur avec son incertitude-type pour
écart-type le plus souvent ; on en tire `N` valeurs ; on refait le calcul
sur chaque tirage ; la moyenne et l'écart-type des résultats sont la valeur
et l'incertitude cherchées. À la main, cela tient en cinq lignes :

```python
import numpy as np

rng = np.random.default_rng()
N = 100000
x = rng.normal(1.0, 0.05, N)
y = rng.normal(2.0, 0.09, N)
q = x * y**2
print(q.mean(), "±", q.std(ddof=1))
```

`Point` fait exactement cela, en gardant le tirage avec la grandeur pour que
les opérations s'écrivent comme sur des nombres :

```python
from tpllg.montecarlo import Point

X = Point(val, u, N=None)
```

| Argument | Sens |
| --- | --- |
| `val` | la valeur |
| `u` | l'incertitude-type |
| `N` | le nombre de tirages ; sans lui, celui des `Point` qu'il rencontre, ou `Point.NN` = 100 000 (section suivante) ; cent mille tirages coûtent quelques millisecondes |

Un `Point` porte une valeur, une incertitude-type et, dès qu'on en a besoin,
un **tirage** de `N` valeurs gaussiennes. Toute opération, entre `Point` ou
avec un nombre, se fait terme à terme sur les tirages ; le résultat est un
`Point` dont `val` est la moyenne des tirages et `u` leur écart-type, pris
avec `ddof=1`.

| Écriture | Ce qui est fait |
| --- | --- |
| `X + Y`, `X - Y`, `X*Y`, `X/Y`, `X**Y` | terme à terme sur les tirages, `Y` un `Point` ou un nombre ; `2*X`, `3 - X`, `2/X`, `2**X` aussi |
| `-X`, `abs(X)` | idem |
| `np.exp(X)`, `np.sqrt(X)`, `np.sin(X)`, `np.arctan2(Y, X)`… | toute fonction numpy élémentaire appliquée aux tirages : le résultat est un `Point` |
| `X.val`, `X.u`, `X.N` | la valeur, l'incertitude-type, le nombre de tirages |
| `X.tirage` | le tableau des tirages, pour un histogramme ou des quantiles |
| `X.quantiles(niveau=0.6827)` | l'intervalle `(bas, haut)` qui contient `niveau` des tirages, autant de chaque côté ; à 68,27 % c'est l'équivalent de ± un écart-type |
| `X.intervalle_le_plus_court(niveau=0.95)` | le plus court des intervalles qui contiennent `niveau` des tirages (GUM-S1) |
| `X.show(ax=None, largeur=5, nbins=1000)` | l'histogramme des tirages, la moyenne, ± l'écart-type, et la loi normale de mêmes moyenne et écart-type, dans le repère `ax` ou une figure neuve |

Pour retrouver les mêmes tirages d'une exécution à l'autre,
`fixer_graine(0)` en tête de script (un entier quelconque) ;
`np.random.seed` n'a pas d'effet sur `tpllg.montecarlo`, qui a son propre
générateur.

`g` par un pendule, `L = 1,000 ± 0,002 m`, `T = 2,007 ± 0,010 s` :

```python
import matplotlib.pyplot as plt
import numpy as np
from tpllg.ajustement import formater
from tpllg.montecarlo import Point, fixer_graine

fixer_graine(0)  # les mêmes tirages d'une fois sur l'autre
L = Point(1.000, 0.002)
T = Point(2.007, 0.010)
g = 4 * np.pi**2 * L / T**2
print(formater(g.val, g.u, "m/s²"), f"({g.N} tirages)")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, point in zip(axes, (L, T, g)):
    point.show(ax, nbins=200)  # l'histogramme d'un Point, dans le repère donné
```

```text
9.80 ± 0.10 m/s² (100000 tirages)
```

![Les histogrammes des cent mille tirages de L, de T et de g, chacun avec sa moyenne, ± son écart-type et la loi normale de mêmes paramètres : ce que show() trace](images/montecarlo_g.png)

Deux gaussiennes en entrée, le calcul refait sur chaque tirage, et
l'histogramme du résultat. `g.show()` seul trace le sien dans une figure
neuve.

La formule de propagation linéaire, `u_g/g = sqrt((u_L/L)² + (2 u_T/T)²)`,
donne `9.80 ± 0.10 m/s²` : ici les incertitudes relatives sont petites et
les deux méthodes coïncident, ce qui est la règle. Monte-Carlo apporte
quelque chose quand elles ne sont plus petites, ou que la fonction n'est pas
dérivable, ou qu'on ne veut pas dériver. Cent mille tirages coûtent
quelques millisecondes.

Deux précisions. Une opération entre deux `Point` suppose qu'ils sont
**indépendants** : chacun a son tirage. Le même `Point` employé deux fois est,
lui, parfaitement corrélé avec lui-même, et c'est juste : `X + 1 - X` a une
incertitude nulle, `X*X` la même que `X**2`, `X + X` celle de `2*X`.

## Le nombre de tirages

Un `Point` mesuré n'est tiré qu'à sa **première utilisation**, avec le
nombre de tirages des `Point` qu'il rencontre — ceux d'un résultat de
`ajuster_modele(..., N=200)`, par exemple — ou `Point.NN` s'il les
rencontre tous non tirés. Une fois tiré, son tirage ne change plus. Deux
`Point` déjà tirés, ou créés avec des `N` explicites différents, ne se
combinent pas :

```text
ValueError: des Point de [1000, 5000] tirages ne se combinent pas : donnez le même N à tous (Point.NN, ou N= à chaque Point et à ajuster_modele)
```

Retirer l'un pour l'aligner sur l'autre, comme le faisait la version
précédente, cassait les corrélations déjà construites avec lui — `(a + b) -
a` ne valait plus `b` — et plantait si c'était un résultat de calcul. Pour
changer le nombre de tirages de tout un calcul, `Point.NN = 20000` avant de
créer les `Point`.

## D'autres lois : une tolérance, une résolution

Une notice de multimètre garantit « ± (0,5 % + 5 mV) » sans en dire plus ;
un affichage numérique arrondit au digit. Ce ne sont pas des écarts-types
mais des **demi-largeurs** : la grandeur est quelque part dans l'intervalle,
également probable partout. Le GUM leur donne une loi uniforme,
d'incertitude-type `demi-largeur/√3`, et c'est ce que tirent :

| Constructeur | Loi | Incertitude-type |
| --- | --- | --- |
| `Point(val, u)`, ou `Point.normale(val, u)` | normale, d'écart-type `u` | `u` |
| `Point.uniforme(val, demi_largeur)` | uniforme sur `val ± demi_largeur` : une tolérance, une résolution (demi-largeur d'un demi-digit) | `demi_largeur/√3` |
| `Point.triangulaire(val, demi_largeur)` | triangulaire de sommet `val` | `demi_largeur/√6` |
| `Point.arcsinus(val, demi_largeur)` | celle d'une grandeur qui oscille sinusoïdalement entre les deux bornes, lue à un instant quelconque | `demi_largeur/√2` |

`X.u` est cette incertitude-type, `X.tirage` suit la loi, et les opérations
sont les mêmes. Prendre une tolérance pour un écart-type gaussien surestime
l'incertitude de 73 % (√3).

## Ce que vaut un tirage

L'écart-type des tirages est une bonne incertitude tant que leur loi
ressemble à une gaussienne. Un quotient par une grandeur incertaine à 15 %
n'en est déjà plus une :

```python
a = Point(1.0, 0.1)
b = Point(2.0, 0.3)  # 15 %
q = a / b
bas, haut = q.quantiles()
print(
    formater(q.val, q.u),
    f"; médiane {np.median(q.tirage):.3f} ; 68 % des tirages entre {bas:.3f} et {haut:.3f}",
)
court = q.intervalle_le_plus_court(0.95)
symetrique = q.quantiles(0.95)
print(
    f"à 95 % : de {symetrique[0]:.3f} à {symetrique[1]:.3f} (2,5 % de chaque côté), "
    f"ou de {court[0]:.3f} à {court[1]:.3f} (le plus court)"
)
```

```text
0.512 ± 0.098 ; médiane 0.500 ; 68 % des tirages entre 0.419 et 0.604
à 95 % : de 0.356 à 0.739 (2,5 % de chaque côté), ou de 0.336 à 0.706 (le plus court)
```

La moyenne est décalée de 0,500 à 0,512, la loi est plus longue à droite
(1/b s'envole quand b est petit), et la formule linéaire donnait ±0,090. À
30 % d'incertitude sur `b`, l'écart-type des tirages n'a plus de sens du
tout : des tirages de `b` passent près de zéro, quelques quotients énormes
le font exploser, alors que les quantiles restent raisonnables. Dans ce
cas-là, ce sont les **quantiles** qu'on reporte, et l'on dit que
l'intervalle est dissymétrique. Pour une loi dissymétrique, le GUM préfère
l'**intervalle le plus court** qui contient 95 % des tirages : ici
0,370 de large au lieu de 0,383, décalé vers le sommet de la loi.

Les histogrammes de `a`, de `b` et de leur quotient, par `show` :

![Les histogrammes de a et de b, deux gaussiennes, et celui de leur quotient, dissymétrique : la loi normale de mêmes moyenne et écart-type ne le suit pas, la médiane est à gauche de la moyenne, et la bande des 68 % des tirages n'est pas centrée](images/montecarlo_quotient.png)

Deux gaussiennes, et une loi qui ne l'est plus : la loi normale de mêmes
moyenne et écart-type passe à côté de l'histogramme, la médiane est à gauche
de la moyenne, et l'intervalle qui contient 68 % des tirages va de 0,081
sous la médiane à 0,104 au-dessus.

## Quelle grandeur pèse le plus : les indices de Sobol

```python
from tpllg.montecarlo import indices_sobol

parts = indices_sobol(fonction, entrees, N=None)
```

| Argument | Sens |
| --- | --- |
| `fonction` | le calcul, `fonction(x1, x2, …)`, sur des tableaux de tirages |
| `entrees` | la liste des `Point` mesurés `x1, x2, …`, dans l'ordre de `fonction` |
| `N` | le nombre de tirages, `Point.NN` par défaut |

Rend un tableau, un indice par grandeur : la part de la variance du
résultat que cette grandeur explique **seule**, ce qu'on gagnerait à la
mesurer parfaitement. Les indices somment à 1 quand les effets s'ajoutent,
à moins quand les grandeurs interagissent. L'estimateur « pick-freeze »
tire deux jeux indépendants et, pour chaque grandeur, recalcule le résultat
avec cette grandeur gardée et les autres retirées : la covariance des deux
résultats, rapportée à la variance, est l'indice. Repris de `mc_sobol.py`
(dataanalysis), dont l'exemple, `y = x1 x2` avec `x1 = 3,1 ± 0,05` et
`x2 = 6,8 ± 0,05` en lois plates, donne 83 % et 17 % : la grandeur la plus
petite pèse le plus, son incertitude relative étant la plus grande.

## Une droite sur tous les tirages, sans boucle

Ajuster une droite sur chaque tirage des `P` points de mesure, dans une
boucle de `polyfit`, prend 7 s pour 100 000 tirages de dix points. La
droite de York — les moindres carrés pondérés par les incertitudes sur `x`
et sur `y`, voir [ajustement.md](ajustement.md#une-droite-quand-x-et-y-sont-incertains--regression_york)
— se calcule sur les `N` tirages d'un coup, en tableaux `(N, P)`, sans
boucle : un dixième de seconde. C'est ce que fait `SerieLineaire`.

```python
from tpllg.montecarlo import SerieLineaire

serie = SerieLineaire(x, u_x, y, u_y, N=100000)
a, b = serie.ajuster()
```

| Argument | Sens |
| --- | --- |
| `x`, `y` | les mesures, `P` points |
| `u_x`, `u_y` | leurs incertitudes-types : un nombre pour tous les points, ou une liste, une par point |
| `N` | le nombre de tirages |

Les tirages sont faits à la construction, dans `serie.x_tirages` et
`serie.y_tirages`, deux tableaux `(N, P)` ; `serie.x_tirages[0]`,
`serie.y_tirages[0]` en donnent un, pour tracer un jeu de mesures comme on
aurait pu l'avoir. `ajuster()`
rend la pente et l'ordonnée à l'origine en deux `Point` : leur **valeur**
est la droite de York des mesures elles-mêmes, leur **tirage** la droite de
York de chacun des `N` tirages, recentré sur cette valeur. Ajuster des
tirages des mesures, c'est ajouter le bruit des tirages à celui qu'elles
ont déjà : avec une incertitude sur `x`, la pente moyenne des tirages est
atténuée (régression diluée : 1,90 pour 2 sur un cas simple, dans la
version précédente, qui rendait cette moyenne). Les tirages ne donnent donc
que la dispersion.

Face à `regression_york` et à `curvefit` avec la variance effective, sur
dix points de `y = 2x + 1` bruités à 0,2 en `x` et 0,5 en `y` :

```text
Monte-Carlo : a = 2.098 ± 0.065  b = 0.25 ± 0.39  (100000 tirages en 0.11 s)
York        : a = 2.098 ± 0.065  b = 0.25 ± 0.39
curvefit    : a = 2.091 ± 0.065  b = 0.28 ± 0.39  chi2 réduit = 0.97
```

Les trois méthodes s'accordent : le Monte-Carlo retrouve, par la dispersion
des tirages, les incertitudes que York calcule ; `curvefit` approche la même
droite en itérant, et rend en plus le χ² réduit.

![Les dix points avec leurs barres d'incertitude en x et en y, la droite de Monte-Carlo et celle de curvefit, confondues](images/montecarlo_droite.png)

Et ce que les tirages contiennent : cent des cent mille droites autour des
mesures, un tirage des mesures (`serie.x_tirages[0]`, `serie.y_tirages[0]`)
avec sa droite, puis les histogrammes de la pente et de l'ordonnée à
l'origine, par `a.show()` et
`b.show()` :

![Les mesures et cent droites, une par tirage, qui s'ouvrent en éventail aux deux bouts ; l'histogramme des cent mille pentes ; celui des cent mille ordonnées à l'origine](images/montecarlo_droite_tirages.png)

Le faisceau des droites est le plus serré au milieu des points et s'ouvre
aux deux bouts : l'ordonnée à l'origine, lue en `x = 0`, au bord des
mesures, est relativement bien moins connue que la pente.

## Un modèle quelconque par tirages

```python
from tpllg.montecarlo import ajuster_modele

params = ajuster_modele(modele, x, u_x, y, u_y, p0, N=1000, **kwargs)
```

| Argument | Sens |
| --- | --- |
| `modele` | `modele(x, a, b, …)`, comme pour `curve_fit` |
| `x`, `u_x`, `y`, `u_y` | les mesures et leurs incertitudes-types, comme `SerieLineaire` |
| `p0` | les valeurs de départ |
| `N` | le nombre de tirages, 1000 par défaut |
| `**kwargs` | transmis à `curve_fit` : `maxfev`, `bounds` |

Rend une liste de `Point`, un par paramètre. Comme pour `SerieLineaire`,
leur valeur est l'ajustement des mesures elles-mêmes, par `curvefit` avec
`u_y` et `u_x` ramenée en `y` par la pente du modèle, et leur tirage les `N`
ajustements des tirages, recentrés sur elle. Ici il n'y a pas de solution
analytique : c'est une boucle de `N` appels à `curve_fit`, avec les poids de
l'ajustement des mesures, quelques dixièmes de seconde pour mille tirages
d'une dizaine de points ; mille suffisent pour deux chiffres significatifs
sur une incertitude. Une exponentielle sur douze points, incertitudes de
0,01 s en `t` et 0,02 V en `u` :

```python
rng = np.random.default_rng(0)  # des mesures simulées
t = np.linspace(0, 5, 12)
u = 2 * np.exp(-t / 1.5) + rng.normal(0, 0.02, t.size)
A, tau = ajuster_modele(lambda t, A, tau: A * np.exp(-t / tau), t, 0.01, u, 0.02, p0=[1, 1], N=2000)
print("A =", formater(A.val, A.u), " tau =", formater(tau.val, tau.u, "s"))
```

```text
A = 2.013 ± 0.020  tau = 1.488 ± 0.022 s  (2000 tirages en 0.1 s)
```

![Les douze mesures et cent exponentielles ajustées, une par tirage, presque confondues ; l'histogramme des deux mille valeurs de A ; celui des deux mille valeurs de tau](images/montecarlo_modele.png)

Avec deux mille tirages au lieu de cent mille, les histogrammes sont plus
heurtés ; l'écart-type, lui, est déjà connu à 2 % près, ce qui suffit pour
ses deux chiffres significatifs.

Un tirage dont l'ajustement ne converge pas est remplacé par un autre, et
la fonction le dit ; dix échecs de suite arrêtent tout (`RuntimeError`) :
de bonnes valeurs de départ, et `maxfev` si besoin, comme pour `curve_fit`.
Pour une droite, `SerieLineaire` tire cent fois plus en autant de temps, et
sans risque d'échec.

## Cas complets

`exemples/montecarlo.py` réunit les cas ci-dessus et les deux qui suivent,
et trace toutes les figures des sections Monte-Carlo de cette fiche. Une
résistance par la loi d'Ohm, avec les tolérances des multimètres, en lois
uniformes, et la part de chacun dans l'incertitude :

```python
# la notice garantit ± (0,5 % + 5 mV) et ± (0,8 % + 0,1 mA) : des demi-largeurs
U = Point.uniforme(4.87, 0.5 * 0.01 * 4.87 + 0.005)
I = Point.uniforme(0.0213, 0.008 * 0.0213 + 0.0001)
R = U / I
print(formater(R.val, R.u, "Ω"), f"; u(U) = {U.u * 1e3:.1f} mV, u(I) = {I.u * 1e6:.0f} µA")
part_U, part_I = indices_sobol(lambda u, i: u / i, [U, I])
print(f"part de la variance de R due à U : {100 * part_U:.0f} %, à I : {100 * part_I:.0f} %")
```

```text
228.7 ± 1.9 Ω ; u(U) = 16.9 mV, u(I) = 156 µA
part de la variance de R due à U : 18 %, à I : 82 %
```

En prenant les tolérances pour des écarts-types gaussiens, on annonçait
±3,2 Ω. Et c'est l'ampèremètre qu'il faut améliorer : il fait plus des
quatre cinquièmes de la variance.

Et l'incertitude d'une fonction non dérivable, la valeur absolue d'une
différence :

```python
d = abs(Point(1.02, 0.05) - Point(1.00, 0.05))
print(formater(d.val, d.u), f" médiane {np.median(d.tirage):.3f}")
```

```text
0.059 ± 0.044  médiane 0.050
```

Ici la formule linéaire ne s'applique pas, la valeur moyenne des tirages
n'est pas `|1,02 − 1,00| = 0,02`, et seul le tirage dit à quoi s'attendre :
une différence nulle à ±0,07 près a une valeur absolue autour de 0,05.

![Les histogrammes de U et de I, deux lois plates ; celui de R, un trapèze aux bords arrondis ; celui de la valeur absolue de la différence, une loi repliée sur zéro que la loi normale de mêmes moyenne et écart-type ne décrit pas, avec la médiane à 0,050 et la valeur 0,02 calculée sans incertitudes](images/montecarlo_cas.png)

Deux lois plates en entrée, et `R` n'en est déjà plus une : un trapèze aux
bords arrondis, en route vers la gaussienne. Son écart-type reste
l'incertitude-type du GUM ; pour un intervalle à 95 %, ce sont les tirages
qu'on lit, `R.intervalle_le_plus_court()`. La valeur absolue replie la loi sur zéro : `R.show()` et
`d.show()` montrent d'un coup d'œil lequel des deux cas on a.
