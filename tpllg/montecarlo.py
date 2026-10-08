"""
Propagation des incertitudes par la méthode de Monte-Carlo (GUM, supplément
1) : des tirages aléatoires sur chaque grandeur mesurée, le calcul refait sur
chaque tirage, et la moyenne et l'écart-type du résultat.

- Point : une grandeur, son incertitude-type et un tirage. Point(val, u) suit
  une loi normale ; Point.uniforme(val, demi_largeur) une loi uniforme, celle
  d'une tolérance de constructeur ou d'une résolution (u = demi-largeur/√3) ;
  Point.triangulaire et Point.arcsinus les deux autres lois du GUM. Les
  opérations entre Point, et avec des nombres, se font sur les tirages, et
  les fonctions numpy (np.exp, np.sqrt, np.sin…) aussi ;
- SerieLineaire : une série de mesures (x ± u_x, y ± u_y) et la droite de
  York refaite sur chaque tirage, sans boucle ;
- ajuster_modele : la même chose pour un modèle quelconque, par une boucle de
  curve_fit, donc bien plus lent ;
- indices_sobol : la part de la variance d'un résultat due à chaque grandeur ;
- fixer_graine : des tirages reproductibles d'une exécution à l'autre.

Le nombre de tirages. Un Point mesuré n'est tiré qu'à sa première
utilisation, avec le nombre de tirages des Point qu'il rencontre (Point.NN
s'il est seul ou s'il les rencontre tous non tirés). Une fois tiré, son
tirage ne change plus : deux Point déjà tirés avec des nombres différents ne
se combinent pas (ValueError). Retirer l'un pour l'aligner sur l'autre
casserait les corrélations déjà construites avec lui — (a + b) - a ne
vaudrait plus b.

Ajuster sur des tirages des mesures, c'est ajouter le bruit des tirages à
celui que les mesures ont déjà : avec une incertitude sur x, la pente moyenne
des tirages est atténuée (régression diluée). La valeur rendue est donc
l'ajustement des mesures elles-mêmes, et les tirages n'en donnent que la
dispersion : ils sont recentrés sur elle.

L'écart-type des tirages est pris avec ddof=1 (sa variance est sans biais).
Le module montecarlo de dataanalysis (2018), refondu ; les lois et les
indices de Sobol viennent de mc_sobol.py (dataanalysis, rsc).

@author: a. marchand
"""

import math
import warnings
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from tpllg.ajustement import _tableau, _vectorisee, _york, curvefit, regression_york

__all__ = ["Point", "SerieLineaire", "ajuster_modele", "fixer_graine", "indices_sobol"]

_generateur = np.random.default_rng()


def fixer_graine(graine=None):
    """Les tirages qui suivent seront les mêmes d'une exécution à l'autre pour
    une même `graine` (un entier) ; None les rend de nouveau imprévisibles."""
    global _generateur
    _generateur = np.random.default_rng(graine)


class Point:
    """Une grandeur mesurée, `Point(val, u)`, ou le résultat d'un calcul,
    `Point(tirage=…)`. `val` et `u` sont la valeur et l'incertitude-type :
    celles données pour une grandeur mesurée, la moyenne et l'écart-type des
    tirages pour un résultat. Le tirage n'est fait qu'à la première demande."""

    NN = 100000  # nombre de tirages par défaut

    def __init__(self, val=None, u=None, N=None, tirage=None):
        if tirage is not None:
            self._tirage = np.asarray(tirage, dtype=float)
            self._N = len(self._tirage)
            self._val = self._u = self._loi = None
            return
        if val is None or u is None:
            raise ValueError("Point(val, u) : il faut la valeur et l'incertitude-type")
        if not u >= 0:
            raise ValueError(f"Point(val, u) : incertitude-type négative ou indéfinie ({u})")
        self._val = float(val)
        self._u = float(u)
        self._loi = ("normale", self._u)
        self._N = None if N is None else int(N)
        self._tirage = None

    @classmethod
    def uniforme(cls, val, demi_largeur, N=None):
        """Une grandeur également probable entre val ± demi_largeur : une
        tolérance de constructeur, la résolution d'un affichage (demi_largeur
        = un demi-digit). Son incertitude-type est demi_largeur/√3."""
        return cls._avec_loi("uniforme", val, demi_largeur, math.sqrt(3), N)

    @classmethod
    def triangulaire(cls, val, demi_largeur, N=None):
        """La loi triangulaire entre val ± demi_largeur, de sommet val :
        u = demi_largeur/√6."""
        return cls._avec_loi("triangulaire", val, demi_largeur, math.sqrt(6), N)

    @classmethod
    def arcsinus(cls, val, demi_largeur, N=None):
        """La loi d'une grandeur qui oscille sinusoïdalement entre val ±
        demi_largeur, et qu'on lit à un instant quelconque : u = demi_largeur/√2."""
        return cls._avec_loi("arcsinus", val, demi_largeur, math.sqrt(2), N)

    @classmethod
    def _avec_loi(cls, nom, val, demi_largeur, rapport, N):
        if not demi_largeur >= 0:
            raise ValueError(f"demi-largeur négative ou indéfinie ({demi_largeur})")
        point = cls(val, demi_largeur / rapport, N)
        point._loi = (nom, float(demi_largeur))
        return point

    def _tirer(self, N):
        """N valeurs tirées selon la loi du Point, indépendantes de son tirage."""
        if self._loi is None:
            raise ValueError("un résultat de calcul n'a pas de loi : il ne se tire pas de nouveau")
        nom, largeur = self._loi
        if largeur == 0:
            return np.full(N, self._val)
        if nom == "normale":
            return _generateur.normal(self._val, largeur, N)
        if nom == "uniforme":
            return _generateur.uniform(self._val - largeur, self._val + largeur, N)
        if nom == "triangulaire":
            return _generateur.triangular(self._val - largeur, self._val, self._val + largeur, N)
        return self._val - largeur * np.cos(np.pi * _generateur.random(N))

    # --- ce qu'on lit
    @property
    def tirage(self):
        """Les N valeurs tirées."""
        if self._tirage is None:
            self._tirage = self._tirer(self.N)
            self._N = len(self._tirage)
        return self._tirage

    @property
    def val(self):
        return self._val if self._val is not None else float(self.tirage.mean())

    @property
    def u(self):
        return self._u if self._u is not None else float(self.tirage.std(ddof=1))

    @property
    def N(self):
        """Le nombre de tirages : celui du tirage s'il est fait, sinon celui
        qu'il aura seul."""
        return self._N if self._N is not None else self.NN

    @property
    def i(self):
        """Une valeur tirée, la première."""
        return self.tirage[0]

    def calc_tirage(self, N=None):
        """Refait le tirage d'une grandeur mesurée, avec N valeurs. Les
        résultats déjà calculés avec l'ancien tirage n'y sont plus corrélés."""
        self._N = int(N) if N is not None else self.N
        self._tirage = self._tirer(self._N)

    def quantiles(self, niveau=0.6827):
        """L'intervalle qui contient `niveau` des tirages, centré en probabilité :
        (bas, haut). À 68,27 % c'est l'équivalent de ± un écart-type, et il
        dit si la loi est dissymétrique."""
        marge = 100 * (1 - niveau) / 2
        return tuple(np.percentile(self.tirage, [marge, 100 - marge]))

    def intervalle_le_plus_court(self, niveau=0.95):
        """Le plus court des intervalles qui contiennent `niveau` des tirages
        (GUM, supplément 1, 7.7.2) : pour une loi dissymétrique, il est plus
        étroit que celui de quantiles, et décentré vers le mode."""
        tri = np.sort(self.tirage)
        n = tri.size
        q = round(niveau * n)
        if q >= n:
            return tri[0], tri[-1]
        i = int(np.argmin(tri[q:] - tri[: n - q]))
        return tri[i], tri[i + q]

    def __repr__(self):
        return f"Point({self.val:g}, {self.u:g}, N={self.N})"

    def show(self, ax=None, largeur=5, nbins=1000):
        """L'histogramme des tirages, la moyenne, ± l'écart-type, et la loi
        normale de même moyenne et écart-type. Rend (fig, ax, n, bins, patches)."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        x1, x2 = self.val - largeur * self.u, self.val + largeur * self.u
        bins = np.linspace(x1, x2, nbins, endpoint=True)
        n, _, patches = ax.hist(self.tirage, bins=bins, density=True)
        haut = max(n) * 1.1
        ax.plot([self.val, self.val], [0, haut], "--", label=f"moyenne={self.val:.2e}")
        trait = ax.plot([self.val - self.u] * 2, [0, haut], "--", label=f"écart-type={self.u:.2e}")
        ax.plot([self.val + self.u] * 2, [0, haut], "--", color=trait[0].get_color())
        x = np.linspace(x1, x2, 500)
        ax.plot(
            x,
            np.exp(-(((x - self.val) / self.u) ** 2) / 2) / (self.u * np.sqrt(2 * np.pi)),
            label="loi normale",
        )
        ax.legend()
        return fig, ax, n, bins, patches

    # --- les opérations : toutes sur les tirages, pour garder les corrélations
    def _op(self, other, fonction):
        tirages = _tirages((self, other))
        if tirages is None:
            return NotImplemented
        return Point(tirage=fonction(*tirages))

    def __add__(self, other):
        return self._op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._op(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._op(other, lambda a, b: a / b)

    def __pow__(self, other):
        return self._op(other, lambda a, b: a**b)

    def __radd__(self, other):
        return self._op(other, lambda a, b: b + a)

    def __rsub__(self, other):
        return self._op(other, lambda a, b: b - a)

    def __rmul__(self, other):
        return self._op(other, lambda a, b: b * a)

    def __rtruediv__(self, other):
        return self._op(other, lambda a, b: b / a)

    def __rpow__(self, other):
        return self._op(other, lambda a, b: b**a)

    def __neg__(self):
        return Point(tirage=-self.tirage)

    def __pos__(self):
        return self

    def __abs__(self):
        return Point(tirage=np.abs(self.tirage))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """np.exp(X), np.sqrt(X), np.arctan2(Y, X)… : une fonction numpy
        appliquée à un Point rend un Point, calculé sur les tirages."""
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        tirages = _tirages(inputs)
        if tirages is None:
            return NotImplemented
        resultat = ufunc(*tirages, **kwargs)
        if isinstance(resultat, tuple):  # np.modf, np.divmod : plusieurs sorties
            return tuple(Point(tirage=r) for r in resultat)
        return Point(tirage=resultat)

    def apply_func(self, func, *args, **kwargs):
        """Un Point dont le tirage est func(tirage, *args, **kwargs), pour une
        fonction qui n'est pas une fonction numpy élémentaire — np.exp(X),
        np.log(X), np.sin(X) s'écrivent directement."""
        return Point(tirage=func(self.tirage, *args, **kwargs))


def _tirages(operandes):
    """Les tirages des opérandes, de même longueur : un tableau par Point, le
    nombre lui-même sinon ; None si un opérande n'est ni l'un ni l'autre."""
    if not all(isinstance(o, (Point, Number)) for o in operandes):
        return None
    points = [o for o in operandes if isinstance(o, Point)]
    tailles = {p._N for p in points if p._N is not None}
    if len(tailles) > 1:
        raise ValueError(
            f"des Point de {sorted(tailles)} tirages ne se combinent pas : donnez le même N à tous "
            "(Point.NN, ou N= à chaque Point et à ajuster_modele)"
        )
    taille = tailles.pop() if tailles else Point.NN
    for p in points:
        if p._tirage is None:
            p._N = taille
    return [o.tirage if isinstance(o, Point) else o for o in operandes]


def _tableaux(x, u_x, y, u_y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x et y doivent être deux tableaux 1D de même longueur")
    return x, _tableau(u_x, x.shape), y, _tableau(u_y, y.shape)


def _recentre(tirage, valeur):
    return tirage - tirage.mean() + valeur


class SerieLineaire:
    """Une série de P mesures (x_i ± u_x,i ; y_i ± u_y,i) et la droite
    y = a x + b par tirages : N tirages des P points, la droite de York sur
    chacun, sans boucle.

    `u_x` et `u_y` sont un nombre, la même incertitude pour tous les points,
    ou une liste, une par point. Les tirages (gaussiens) sont faits à la
    construction, dans deux tableaux (N, P)."""

    def __init__(self, x, u_x, y, u_y, N=None):
        self.x, self.u_x, self.y, self.u_y = _tableaux(x, u_x, y, u_y)
        self.P = len(self.x)
        self.N = int(N) if N is not None else Point.NN
        self.x_tirages = _generateur.normal(self.x, self.u_x, size=(self.N, self.P))
        self.y_tirages = _generateur.normal(self.y, self.u_y, size=(self.N, self.P))

    @property
    def Nt(self):
        return self.N

    @property
    def xi(self):
        """Un tirage des x, le premier : un jeu de mesures comme on aurait pu l'avoir."""
        return self.x_tirages[0]

    @property
    def yi(self):
        return self.y_tirages[0]

    @staticmethod
    def coefs(x, y):
        """y = a x + b par moindres carrés ordinaires, sans incertitudes :
        a = Cov(x, y)/V(x), b = <y> - a<x>. Sur des tableaux 1D, ou 2D (un jeu
        de mesures par ligne)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        xm, ym = x.mean(axis=-1), y.mean(axis=-1)
        dx = x - xm[..., None]  # centré avant de multiplier : stable même pour x ~ 1e8
        a = (dx * (y - ym[..., None])).sum(axis=-1) / (dx**2).sum(axis=-1)
        return a, ym - a * xm

    def ajuster(self):
        """Rend (a, b), deux Point : leur valeur est la droite de York des
        mesures (regression_york), leur tirage la droite de York de chacun des
        N tirages, recentré sur cette valeur."""
        reference = regression_york(self.x, self.u_x, self.y, self.u_y)
        pentes, ordonnees = _york(self.x_tirages, self.u_x**2, self.y_tirages, self.u_y**2)[:2]
        return (
            Point(tirage=_recentre(pentes, reference.pfit[0])),
            Point(tirage=_recentre(ordonnees, reference.pfit[1])),
        )


def ajuster_modele(modele, x, u_x, y, u_y, p0, N=1000, **kwargs):
    """Un modèle quelconque, `modele(x, *params)`, ajusté sur chacun des N
    tirages gaussiens des mesures : rend une liste de Point, un par
    paramètre, dont la valeur est l'ajustement des mesures par curvefit (avec
    u_y, et u_x ramenée en y par la pente du modèle) et le tirage les N
    ajustements des tirages, recentrés sur elle.

    Chaque tirage est ajusté par curve_fit avec les poids de l'ajustement des
    mesures ; un tirage qui ne converge pas est remplacé par un autre (et on
    le dit). Une boucle de N appels à curve_fit, donc de l'ordre d'une seconde
    pour N = 1000 : pour une droite, SerieLineaire fait la même chose cent
    fois plus vite. Les mots-clés (maxfev, bounds…) vont à curve_fit."""
    x, u_x, y, u_y = _tableaux(x, u_x, y, u_y)
    N = int(N)
    reference = curvefit(
        modele,
        x,
        y,
        p0,
        datayerrors=u_y,
        dataxerrors=u_x if np.any(u_x > 0) else None,
        verbose=False,
        **kwargs,
    )
    p_ref = reference.pfit
    f = _vectorisee(modele, x, p_ref)
    sigma = np.sqrt(u_y**2 + ((f(x + u_x, *p_ref) - f(x - u_x, *p_ref)) / 2) ** 2)
    params = np.empty((N, p_ref.size))
    remplaces = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        for i in range(N):
            for _ in range(10):
                x_t, y_t = _generateur.normal(x, u_x), _generateur.normal(y, u_y)
                try:
                    params[i] = curve_fit(f, x_t, y_t, p0=p_ref, sigma=sigma, absolute_sigma=True, **kwargs)[
                        0
                    ]
                    break
                except RuntimeError:
                    remplaces += 1
            else:
                raise RuntimeError(
                    "curve_fit n'aboutit pas sur dix tirages de suite : revoir p0 ou le modèle"
                )
    if remplaces:
        print(f"ajuster_modele : {remplaces} tirage(s) sans convergence, remplacé(s) par d'autres")
    return [Point(tirage=_recentre(params[:, k], p_ref[k])) for k in range(p_ref.size)]


def indices_sobol(fonction, entrees, N=None):
    """Les indices de Sobol du premier ordre : pour chaque grandeur d'entrée,
    la part de la variance du résultat qu'elle explique seule — ce qu'on
    gagnerait à mieux la mesurer. Ils somment à 1 pour un modèle additif,
    à moins quand les grandeurs interagissent.

    fonction(x1, x2…) calcule le résultat sur des tableaux de tirages ;
    entrees est la liste des Point mesurés (x1, x2…), dans le même ordre.
    Estimateur « pick-freeze » sur deux jeux de N tirages indépendants (N par
    défaut Point.NN) : S_i = Cov(f(A), f(B avec la colonne i de A))/Var f(A).
    Rend un tableau, un indice par entrée."""
    N = int(N) if N is not None else Point.NN
    a = [p._tirer(N) for p in entrees]
    b = [p._tirer(N) for p in entrees]
    y = np.asarray(fonction(*a), dtype=float)
    variance = y.var()
    if not variance > 0:
        raise ValueError("le résultat ne varie pas : pas d'indices de Sobol")
    indices = []
    for i in range(len(entrees)):
        y_i = np.asarray(fonction(*[a[j] if j == i else b[j] for j in range(len(entrees))]), dtype=float)
        indices.append(np.mean((y - y.mean()) * (y_i - y_i.mean())) / variance)
    return np.array(indices)
