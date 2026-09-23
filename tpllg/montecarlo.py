# -*- coding: utf-8 -*-
"""
Propagation des incertitudes par la méthode de Monte-Carlo : des tirages
aléatoires gaussiens sur chaque grandeur mesurée, le calcul refait sur chaque
tirage, et la moyenne et l'écart-type du résultat.

- Point : une valeur, une incertitude-type et un tirage ; les opérations
  entre Point, et avec des nombres, se font sur les tirages ;
- SerieLineaire : une série de mesures (x ± u_x, y ± u_y) et l'ajustement
  y = ax + b refait sur chaque tirage, sans boucle — la régression affine a
  une solution analytique, calculée sur tous les tirages à la fois ;
- ajuster_modele : la même chose pour un modèle quelconque, par une boucle
  de curve_fit, donc bien plus lent.

L'écart-type des tirages est l'estimateur sans biais (ddof=1), comme dans la
fiche de méthodologie. Le module montecarlo de dataanalysis (2018), refondu.

@author: a. marchand
"""
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

__all__ = ["Point", "SerieLineaire", "ajuster_modele"]


class Point:
    """Une grandeur mesurée, `Point(val, u)`, ou le résultat d'un calcul,
    `Point(tirage=…)`. `val` et `u` sont la moyenne et l'écart-type des
    tirages ; le tirage n'est fait qu'à la première demande."""

    NN = 100000     # nombre de tirages par défaut

    def __init__(self, val=None, u=None, N=None, tirage=None):
        if tirage is not None:
            self._tirage = np.asarray(tirage, dtype=float)
            self._N = len(self._tirage)
            self._val = None
            self._u = None
        else:
            if val is None or u is None:
                raise ValueError("Point(val, u) : il faut la valeur et l'incertitude-type")
            self._val = float(val)
            self._u = float(u)
            self._N = int(N) if N is not None else self.NN
            self._tirage = None

    # --- les trois choses qu'on lit
    @property
    def tirage(self):
        """Les N valeurs tirées, gaussiennes autour de val à u près."""
        if self._tirage is None:
            self.calc_tirage(self._N)
        return self._tirage

    @property
    def val(self):
        return self._val if self._val is not None else self.tirage.mean()

    @property
    def u(self):
        return self._u if self._u is not None else self.tirage.std(ddof=1)

    @property
    def N(self):
        return self._N

    def calc_tirage(self, N=None):
        """Refait le tirage, avec N valeurs."""
        if N is not None:
            self._N = int(N)
        self._tirage = np.random.normal(self._val, self._u, self._N)

    @property
    def i(self):
        """Une valeur tirée, la première."""
        return self.tirage[0]

    def quantiles(self, niveau=0.6827):
        """L'intervalle qui contient `niveau` des tirages, centré en probabilité :
        (bas, haut). À 68,27 % c'est l'équivalent de ± un écart-type, et il
        dit si la loi est dissymétrique."""
        marge = 100*(1 - niveau)/2
        return tuple(np.percentile(self.tirage, [marge, 100 - marge]))

    def __repr__(self):
        return "Point(%g, %g, N=%d)" % (self.val, self.u, self.N)

    def show(self, ax=None, largeur=5, nbins=1000):
        """L'histogramme des tirages, la moyenne, ± l'écart-type, et la loi
        normale de même moyenne et écart-type. Rend (fig, ax, n, bins, patches)."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        x1, x2 = self.val - largeur*self.u, self.val + largeur*self.u
        bins = np.linspace(x1, x2, nbins, endpoint=True)
        n, _, patches = ax.hist(self.tirage, bins=bins, density=True)
        M = max(n)*1.1
        ax.plot([self.val, self.val], [0, M], '--', label='moyenne=%.2e' % self.val)
        L = ax.plot([self.val - self.u, self.val - self.u], [0, M], '--',
                    label='écart-type=%.2e' % self.u)
        ax.plot([self.val + self.u, self.val + self.u], [0, M], '--', color=L[0].get_color())
        x = np.linspace(x1, x2, 500)
        y = np.exp(-((x - self.val)/self.u)**2/2)/(self.u*np.sqrt(2*np.pi))
        ax.plot(x, y, label='loi normale')
        ax.legend()
        return fig, ax, n, bins, patches

    # --- les opérations : toutes sur les tirages, pour garder les corrélations
    def _tirage_de(self, other):
        """Le tirage de l'autre opérande : un tableau pour un Point (aligné sur
        le même nombre de tirages), le nombre lui-même sinon."""
        if isinstance(other, Point):
            if self.N > other.N:
                other.calc_tirage(self.N)
            elif self.N < other.N:
                self.calc_tirage(other.N)
            return other.tirage
        if isinstance(other, Number):
            return other
        return None

    def _op(self, other, fonction):
        autre = self._tirage_de(other)
        if autre is None:
            return NotImplemented
        return Point(tirage=fonction(self.tirage, autre))

    def __add__(self, other):
        return self._op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._op(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._op(other, lambda a, b: a*b)

    def __truediv__(self, other):
        return self._op(other, lambda a, b: a/b)

    def __pow__(self, other):
        return self._op(other, lambda a, b: a**b)

    def __radd__(self, other):
        return self._op(other, lambda a, b: b + a)

    def __rsub__(self, other):
        return self._op(other, lambda a, b: b - a)

    def __rmul__(self, other):
        return self._op(other, lambda a, b: b*a)

    def __rtruediv__(self, other):
        return self._op(other, lambda a, b: b/a)

    def __rpow__(self, other):
        return self._op(other, lambda a, b: b**a)

    def __neg__(self):
        return Point(tirage=-self.tirage)

    def __pos__(self):
        return self

    def __abs__(self):
        return Point(tirage=np.abs(self.tirage))

    def apply_func(self, func, *args, **kwargs):
        """Un Point dont le tirage est func(tirage, *args, **kwargs) : n'importe
        quelle fonction numpy, np.exp, np.log, np.sin…"""
        return Point(tirage=func(self.tirage, *args, **kwargs))


def _tableaux(x, u_x, y, u_y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x et y doivent être deux tableaux 1D de même longueur")
    u_x = np.broadcast_to(np.asarray(u_x, dtype=float), x.shape)
    u_y = np.broadcast_to(np.asarray(u_y, dtype=float), y.shape)
    return x, u_x, y, u_y


class SerieLineaire:
    """Une série de P mesures (x_i ± u_x,i ; y_i ± u_y,i) et l'ajustement
    y = ax + b par tirages : N tirages des P points, une régression par
    tirage, la moyenne et l'écart-type des pentes et des ordonnées.

    `u_x` et `u_y` sont un nombre, la même incertitude pour tous les points,
    ou une liste, une par point. Les tirages sont faits à la construction,
    dans deux tableaux (N, P) ; `ajuste()` ne fait aucune boucle."""

    def __init__(self, x, u_x, y, u_y, N=Point.NN):
        self.x, self.u_x, self.y, self.u_y = _tableaux(x, u_x, y, u_y)
        self.P = len(self.x)
        self.N = int(N)
        self.x_tirages = np.random.normal(self.x, self.u_x, size=(self.N, self.P))
        self.y_tirages = np.random.normal(self.y, self.u_y, size=(self.N, self.P))

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
        """y = ax + b par moindres carrés : a = Cov(x,y)/V(x), b = <y> - a<x>.
        Sur des tableaux 1D, ou 2D (un jeu de mesures par ligne)."""
        xm, ym = x.mean(axis=-1), y.mean(axis=-1)
        cov = (x*y).mean(axis=-1) - xm*ym
        var = (x*x).mean(axis=-1) - xm*xm
        a = cov/var
        b = ym - a*xm
        return a, b

    def ajuste(self):
        """Rend (a, b), deux Point dont les tirages sont les N régressions."""
        a, b = self.coefs(self.x_tirages, self.y_tirages)
        return Point(tirage=a), Point(tirage=b)


def ajuster_modele(modele, x, u_x, y, u_y, p0, N=1000, **kwargs):
    """Un modèle quelconque, `modele(x, *params)`, ajusté par curve_fit sur
    chacun des N tirages des mesures : rend une liste de Point, un par
    paramètre, dont les tirages sont les N valeurs ajustées.

    Une boucle de N appels à curve_fit, donc de l'ordre d'une seconde pour
    N = 1000 : pour un modèle affine, SerieLineaire fait la même chose cent
    fois plus vite. Les mots-clés (maxfev, bounds…) vont à curve_fit."""
    x, u_x, y, u_y = _tableaux(x, u_x, y, u_y)
    x_tirages = np.random.normal(x, u_x, size=(N, len(x)))
    y_tirages = np.random.normal(y, u_y, size=(N, len(y)))
    params = np.zeros((N, len(p0)))
    for i in range(N):
        params[i], _ = curve_fit(modele, x_tirages[i], y_tirages[i], p0=p0, **kwargs)
    return [Point(tirage=params[:, k]) for k in range(len(p0))]
