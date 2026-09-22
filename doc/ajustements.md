# Ajustements et incertitudes

`tpllg.ajustement` habille `scipy.optimize.curve_fit` pour deux besoins qui
reviennent en TP : tenir compte des incertitudes de mesure, et ajuster une
grandeur complexe mesurée par son module et sa phase. `tpllg.incertitudes` et
`tpllg.montecarlo` complètent, pour une série de mesures et pour la
propagation par tirages.

## Ce que curve_fit fait

Un ajustement cherche les paramètres $p$ qui rendent minimale la somme des
carrés des écarts entre mesures et modèle,

$$\chi^2(p)=\sum_i \frac{\big(y_i-f(x_i;p)\big)^2}{\sigma_i^2}\ .$$

Pour un modèle affine le minimum est analytique ; pour un modèle quelconque,
`curve_fit` itère (Levenberg-Marquardt) à partir des valeurs de départ `p0`,
et s'arrête quand $\chi^2$ ne descend plus. Il rend les paramètres et leur
matrice de covariance `pcov`, dont la diagonale porte les variances — les
incertitudes-types sont leurs racines — et les termes croisés les
corrélations entre paramètres.

Deux conséquences à garder en tête. Les valeurs de départ comptent : loin du
bon minimum, une descente locale peut mener n'importe où, et on les lit sur
un tracé des mesures avant d'ajuster. Et sans incertitudes fournies,
`curve_fit` met `pcov` à l'échelle de la dispersion des résidus : les
incertitudes rendues supposent le modèle bon et les écarts aléatoires ; elles
ne connaissent ni une erreur systématique ni un modèle faux.

## curvefit, avec les incertitudes

```python
from tpllg.ajustement import curvefit

pfit, err, chi2 = curvefit(function, datax, datay, p0,
                           datayerrors=None, dataxerrors=None,
                           function_derivate=None,
                           n_var_method_max=10, chi_limit=0.01,
                           verbose=True, **kwargs)
```

| Argument | Sens |
| --- | --- |
| `function` | le modèle, `function(x, a, b, …)`, vectorisé en `x` |
| `datax`, `datay` | les mesures, tableaux de même longueur |
| `p0` | les valeurs de départ, une par paramètre |
| `datayerrors` | les incertitudes-types sur `y` ; sans elles, tous les points pèsent pareil, `pcov` est mise à l'échelle des résidus comme le fait `curve_fit`, et le χ² réduit rendu n'a pas de sens |
| `dataxerrors` | les incertitudes-types sur `x` ; demande `datayerrors` et `function_derivate` |
| `function_derivate` | la dérivée du modèle par rapport à `x`, mêmes arguments que `function` |
| `n_var_method_max`, `chi_limit` | la boucle de la variance effective : au plus tant d'itérations, arrêt quand le χ² réduit ne bouge plus que de tant |
| `verbose` | `False` pour taire la méthode employée |
| `**kwargs` | transmis à `curve_fit` (`maxfev`, `bounds`…) |

Retour : `pfit` les paramètres, `err` leurs incertitudes-types, `chi2` le
**χ² réduit**, c'est-à-dire $\chi^2$ divisé par le nombre de points moins le
nombre de paramètres. Quand les incertitudes fournies sont justes et le
modèle bon, il vaut 1 à quelques dixièmes près ; nettement plus grand, le
modèle ne décrit pas les mesures ou les incertitudes sont sous-estimées ;
nettement plus petit, elles sont surestimées.

Avec des incertitudes sur `x`, la méthode est celle de la **variance
effective** (Orear, 1982) : chaque incertitude en `x` est ramenée en `y` par
la pente locale du modèle, $\sigma_i^2 = \sigma_{y,i}^2 + \sigma_{x,i}^2\,f'(x_i)^2$,
et l'on itère puisque la pente dépend des paramètres. Pour une droite avec
des incertitudes constantes, les paramètres ne changent pas d'une méthode à
l'autre ; seules leurs incertitudes et le χ² bougent. Avec des incertitudes
variables d'un point à l'autre, les paramètres changent aussi.

Le cas complet, `exemples/ajustement_incertitudes.py`, sur une droite :

```python
import numpy as np
from tpllg.ajustement import curvefit, resume_parametres

def modele(x, a, b):
    return a*x + b

def modele_derivee(x, a, b):     # dérivée par rapport à x
    return a

x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])
y = np.array([-0.85, -0.42, 0.11, 0.35, 0.84, 1.15, 1.70, 1.95, 2.36, 2.85])
sigma_x = 0.05*np.ones(x.size)
sigma_y = 0.15*np.ones(x.size)

# 1. tous les points pèsent pareil ; pcov est mise à l'échelle des résidus, chi2 sans objet
pfit, err, chi2 = curvefit(modele, x, y, p0=[1, 0])
# 2. les incertitudes sur y pèsent les points ; chi2 dit si elles sont justes
pfit, err, chi2 = curvefit(modele, x, y, p0=[1, 0], datayerrors=sigma_y)
# 3. incertitudes sur x et y : variance effective, il faut la dérivée
pfit, err, chi2 = curvefit(modele, x, y, p0=[1, 0], datayerrors=sigma_y,
                           dataxerrors=sigma_x, function_derivate=modele_derivee)
print(resume_parametres(("a", "b"), pfit, sigmas=err), f"chi2 réduit = {chi2:.2f}")
```

Les trois cas, sur ces mêmes points, imprimés l'un sous l'autre (`verbose=False`) :

```text
sans incertitudes        a = 2.010 ± 0.038   b = -1.006 ± 0.044   chi2 réduit sans objet
incertitudes sur y       a = 2.010 ± 0.083   b = -1.006 ± 0.095   chi2 réduit = 0.21
incertitudes sur x et y  a = 2.010 ± 0.099   b = -1.01 ± 0.11   chi2 réduit = 0.14
```

## curve_fit_complex, module et phase ensemble

Une fonction de transfert se mesure par son module $|H|$ et sa phase
$\varphi$, à chaque fréquence. Ajuster le module seul ou la phase seule
n'utilise que la moitié des mesures ; les ajuster séparément donne deux jeux
de paramètres à réconcilier. `curve_fit_complex` écrit chaque mesure sous
forme complexe, $|H|e^{j\varphi}$, empile parties réelles et imaginaires en un
seul vecteur, et ajuste le modèle complexe dessus :

```python
from tpllg.ajustement import curve_fit_complex

pfit, pcov = curve_fit_complex(complex_func, x_data, norm, phase, **kwargs)
```

| Argument | Sens |
| --- | --- |
| `complex_func` | le modèle, `complex_func(x, *params)`, qui rend un tableau complexe |
| `x_data` | les abscisses, la fréquence en général |
| `norm` | les modules mesurés |
| `phase` | les phases mesurées, en **radians** ; le tour complet ne pose aucun problème, +175° et −175° sont le même nombre complexe |
| `**kwargs` | transmis à `curve_fit`, `p0` en premier lieu |

Retour : `pfit`, `pcov`, comme `curve_fit`. Un passe-bande du second ordre :

```python
import numpy as np
from tpllg.ajustement import curve_fit_complex, residus_complexes, resume_parametres

def gain(f, H0, f0, Q):
    return H0/(1 + 1j*Q*(f/f0 - f0/f))

f = np.array([200, 500, 1000, 1500, 1800, 1900, 2000, 2100, 2200, 2700, 5000, 10000.])
H = np.array([0.08, 0.20, 0.45, 1.10, 2.6, 4.0, 4.9, 4.2, 2.9, 1.2, 0.5, 0.22])
phi = np.radians([-95, -100, -110, -130, -155, -170, 180, 165, 150, 125, 100, 95])

pfit, pcov = curve_fit_complex(gain, f, H, phi, p0=[-5, 2000, 6])
print(resume_parametres(("H0", "f0", "Q"), pfit, pcov, unites=("", "Hz", "")))
res_H, res_phi = residus_complexes(gain, f, H, phi, pfit)
print("résidus relatifs sur |H| :", np.round(res_H, 2))
print("résidus sur phi (°)      :", np.round(res_phi, 1))
```

`residus_complexes` rend l'écart relatif sur le module et l'écart de phase en
degrés, point par point : des résidus répartis au hasard, de l'ordre des
incertitudes de mesure, valident le modèle ; une tendance est une
information.

Les valeurs de départ se lisent sur le tracé des mesures : `f0` à la
fréquence du maximum de $|H|$, `H0` ce maximum avec son signe (la phase à la
résonance dit lequel : 0° ou 180°), `Q` le rapport de `f0` à la largeur à
$|H_0|/\sqrt2$. Partir à une décade de la résonance mène à un résultat
absurde ; partir avec un `Q` faux d'un facteur dix se corrige seul.

## Présenter un résultat

```python
from tpllg.ajustement import ecarts_types, formater, resume_parametres

ecarts_types(pcov)                       # les incertitudes-types, racines de la diagonale
formater(1993.489, 1.875, "Hz")          # '1993.5 ± 1.9 Hz'
resume_parametres(("H0", "f0", "Q"), pfit, pcov, unites=("", "Hz", ""))
resume_parametres(("a", "b"), pfit, sigmas=err)     # avec le err de curvefit
```

`formater` arrondit l'incertitude à deux chiffres significatifs et la valeur
au même rang, la règle des fiches de TP. Sans incertitude, quatre chiffres.

## Une série de mesures

```python
from tpllg.incertitudes import incertitudes, student_coef

m, delta, sigma = incertitudes([9.78, 9.81, 9.85, 9.79, 9.83])
```

Rend la moyenne, l'incertitude-type **de la moyenne** ($s/\sqrt N$) et
l'écart-type estimé $s$. `incertitudes(liste, sigma=2, advanced=True)`
multiplie `delta` par le coefficient de Student pour un niveau de confiance
à `sigma` écarts-types (2 pour 95 %), ce qui compte quand $N$ est petit :
`student_coef(sigma, n)` le donne seul. `loi_normale(x, m, s)` et
`loi_normale_cumulee(t)` (la probabilité d'un tirage dans $[-t, t]$) servent
aux tracés.

## Monte-Carlo

```python
import numpy as np
from tpllg.montecarlo import Point, SerieLineaire

L = Point(1.000, 0.002)          # une longueur, en m, à ± 2 mm
T = Point(2.007, 0.010)          # une période, en s
g = 4*np.pi**2*L/T**2            # les opérations se propagent sur les tirages
print(g.val, g.u)                # moyenne et écart-type des tirages
g.show()                         # l'histogramme, avec la loi normale
```

Un `Point` porte une valeur, une incertitude-type et un tirage de 100 000
valeurs gaussiennes ; les opérations `+ - * / **` combinent les tirages,
`apply_func(np.exp)` applique n'importe quelle fonction, et `val`, `u` en
tirent la valeur et l'incertitude. C'est la propagation des incertitudes sans
formule de dérivées partielles, et sans hypothèse de petitesse.

`SerieLineaire(x, u_x, y, u_y)` fait de même pour une droite $y=ax+b$ :
`a, b = serie.ajuste()` rend deux `Point`, l'ajustement ayant été refait sur
chaque tirage des mesures. `exemples/montecarlo.py` montre les deux.
