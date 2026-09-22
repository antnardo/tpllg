# -*- coding: utf-8 -*-
"""
Propagation des incertitudes par Monte-Carlo : un Point porte une valeur, une
incertitude et un tirage, et se combine comme un nombre ; une SerieLineaire
ajuste y = ax + b tirage par tirage. Le module montecarlo de dataanalysis
(2018), fusionné ici.

@author: a. marchand
"""
import numpy as np
import matplotlib.pyplot as plt
from numbers import Number


class Point:
    NN = 100000
    NUM_TYPES = [int, float, np.integer, np.floating]

    def __init__(self, val=None, u=None, N=None, tirage=None):
        if N is None:
            N = self.NN
        self._val = val
        self._u = u
        self._tirage = tirage
        self._N = N if tirage is None else len(tirage)

    @property
    def tirage(self):
        if self._tirage is None:
            self.calc_tirage(self._N)
        return self._tirage

    @property
    def N(self):
        return self._N

    @property
    def val(self):
        if self._val is None:
            self._val = (self._tirage).mean()
        return self._val

    @property
    def u(self):
        if self._u is None:
            self._u = (self._tirage).std()
        return self._u

    def calc_tirage(self, N=N):
        self._N = N
        self._tirage = np.random.normal(self.val, self.u, N)

    @property
    def i(self):
        # instance
        return self.tirage[0]

    def show(self, ax=None, largeur=5, nbins=1000):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        x1, x2 = self.val-largeur*self.u, self.val+largeur*self.u
        bins = np.linspace(x1, x2, nbins, endpoint=True)
        n, _, patches = ax.hist(self.tirage, bins=bins, density=True)
        M = max(n)*1.1
        ax.plot([self.val, self.val], [0, M], '--', label=f'moyenne={self.val:.2e}')
        L = ax.plot([self.val-self.u, self.val-self.u], [0, M], '--', label=f'écart-type={self.u:.2e}')
        ax.plot([self.val+self.u, self.val+self.u], [0, M], '--', color=L[0].get_color())
        x = np.linspace(x1, x2, 500)
        y = np.exp(-((x-self.val)/self.u)**2/2)/(self.u*np.sqrt(2*np.pi))
        ax.plot(x, y, label='loi normale')
        plt.legend()
        return fig, ax, n, bins, patches

    def _reset_n(self, other):
        if self.N > other.N:
            other.calc_tirage(self.N)
        elif self.N < other.N:
            self.calc_tirage(other.N)

    @staticmethod
    def _check_type(other):
        return isinstance(other, Number)

    def __add__(self, other):
        if self._check_type(other):
            return Point(val=self.val+other, u=self.u, N=self.N)
        elif type(other) is Point:
            self._reset_n(other)
            return Point(tirage=self.tirage+other.tirage)
        else:
            NotImplemented

    def __mul__(self, other):
        if self._check_type(other):
            return Point(val=self.val*other, u=self.u*other, N=self.N)
        elif type(other) is Point:
            self._reset_n(other)
            return Point(tirage=self.tirage*other.tirage)
        else:
            NotImplemented

    def __truediv__(self, other):
        if self._check_type(other):
            return Point(val=self.val/other, u=self.u/other, N=self.N)
        elif type(other) is Point:
            self._reset_n(other)
            return Point(tirage=self.tirage/other.tirage)
        else:
            NotImplemented

    def __pow__(self, other):
        if self._check_type(other):
            return Point(val=self.val**other, u=self.u*other*self.val**(other-1), N=self.N)
        elif type(other) is Point:
            self._reset_n(other)
            return Point(tirage=self.tirage**other.tirage)
        else:
            NotImplemented

    def __neg__(self):
        return Point(tirage=-self.tirage)

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self-other

    def __radd__(self, other):
        return self+other

    def __rsub__(self, other):
        return -(self-other)

    def __rmul__(self, other):
        return self*other

    def __rtruediv__(self, other):
        return (self**(-1))*other

    def __rpow__(self, other):
        # other should be int or float...
        return (self*np.log(other)).apply_func(np.exp)

    def apply_func(self, func):
        return Point(tirage=func(self.tirage))


class SerieLineaire:
    def __init__(self, x, u_x, y, u_y):
        self.N = len(x)
        if isinstance(u_x, Number):
            u_x = [u_x]*self.N
        if isinstance(u_y, Number):
            u_y = [u_y]*self.N
        self.x = [Point(xi, u) for (xi, u) in zip(x, u_x)]  # liste de Points
        self.y = [Point(yi, u) for (yi, u) in zip(y, u_y)]
        self.Nt = self.x[0].N
        # long :
        self.sliced = [(np.array([xi.tirage[i] for xi in self.x]), np.array([yi.tirage[i] for yi in self.y])) for i in range(self.Nt)]

    @property
    def xi(self):
        return [x.i for x in self.x]

    @property
    def yi(self):
        return [y.i for y in self.y]

    @staticmethod
    def coefs(x, y):
        """ y=ax+b
        a = Cov(x,y)/V(x)
        b = <y> - a<x>
        """
        xm, ym = x.mean(), y.mean()
        cov = (x*y).mean()-xm*ym
        V = x.std()**2
        a = cov/V
        b = ym - a*xm
        return a, b

    def ajuste(self):
        a = np.zeros(self.Nt)
        b = np.zeros(self.Nt)
        for i, (x, y) in enumerate(self.sliced):
            a[i], b[i] = self.coefs(x, y)
        return Point(tirage=a), Point(tirage=b)
