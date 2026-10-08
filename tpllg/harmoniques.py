"""
Un signal périodique décrit par ses harmoniques : la synthèse de Fourier, le
passage dans un filtre linéaire, la valeur efficace — le module
traitementsignal (2014-2019), fusionné ici.

    s(t) = moyenne + somme pour n de 1 à nmax de A_n cos(2 pi n f0 t + phi_n)

- Signal : un tel signal, à partir d'une fonction spectre(n) -> (A_n, phi_n)
  ou de la liste de ses coefficients ; on l'évalue en t, on le fait passer
  dans un filtre, on en lit la valeur efficace ;
- spectre_carre, spectre_triangle, spectre_dent_de_scie : les spectres des
  signaux usuels, entre -1 et 1 ;
- passe_bas_1, passe_haut_1, passe_bas_2, passe_haut_2, passe_bande,
  coupe_bande : les fonctions de transfert usuelles, H(f) complexe.

Ce qu'apportait traitementsignal, corrigé : filtrage.make_filtre, appelé en
positionnel, faisait planter les cinq fabriques de filtres ; la valeur
efficace comptait moyenne²/2 au lieu de moyenne² (1,58 au lieu de 2,12 pour
2 + cos) ; le filtrage perdait le signe de la composante continue
(moyenne × |H(0)|) ; un spectre en entiers levait UFuncTypeError ;
spectre_to_func modifiait la liste de phases de l'appelant. fourier.fourier
normalisait par 2 dt, juste pour une seconde de signal : c'est
tpllg.fft.calcule_DFT qui le remplace.

Pour le spectre d'un signal échantillonné (une acquisition), voir tpllg.fft.
"""

import numpy as np

__all__ = [
    "Signal",
    "coupe_bande",
    "passe_bande",
    "passe_bas_1",
    "passe_bas_2",
    "passe_haut_1",
    "passe_haut_2",
    "spectre_carre",
    "spectre_dent_de_scie",
    "spectre_triangle",
]


class Signal:
    """Un signal périodique : sa fréquence fondamentale `f0`, sa `moyenne`,
    et ses harmoniques de rang 1 à `nmax`, données par spectre(n) -> (A_n,
    phi_n), vectorisée en n (un tableau d'entiers), multipliées par
    `amplitude`.

        carre = Signal(f0=100, spectre=spectre_carre, amplitude=2.5, moyenne=2.5, nmax=50)
        carre(t)                       # le signal à l'instant t (un tableau)
        carre.filtre(passe_bas_1(300)) # un Signal : la sortie du filtre
        carre.Veff                     # la valeur efficace, par Parseval

    Attributs : f0, moyenne, nmax, n (les rangs), frequences, pulsations,
    amplitudes (positives : un signe moins devient un demi-tour de phase),
    phases (radians, entre -pi et pi), et frequences_0, amplitudes_0,
    phases_0 qui commencent par la composante continue.
    """

    def __init__(self, *, f0, spectre, moyenne=0.0, amplitude=1.0, nmax=20):
        self.f0 = float(f0)
        self.moyenne = float(moyenne)
        self.nmax = int(nmax)
        self.n = np.arange(1, self.nmax + 1)
        A, phi = spectre(self.n)
        A = amplitude * np.broadcast_to(np.asarray(A, dtype=float), self.n.shape)
        phi = np.broadcast_to(np.asarray(phi, dtype=float), self.n.shape) + np.where(A < 0, np.pi, 0.0)
        self.amplitudes = np.abs(A)
        self.phases = np.angle(np.exp(1j * phi))
        self.frequences = self.f0 * self.n
        self.pulsations = 2 * np.pi * self.frequences
        self.frequences_0 = np.concatenate(([0.0], self.frequences))
        self.amplitudes_0 = np.concatenate(([self.moyenne], self.amplitudes))
        self.phases_0 = np.concatenate(([0.0], self.phases))

    @classmethod
    def depuis_coefficients(cls, f0, amplitudes, phases=None):
        """Le signal de coefficients donnés en liste : amplitudes[0] est la
        composante continue, amplitudes[n] et phases[n] l'harmonique n (la
        phase du rang 0 est ignorée). Les listes ne sont pas modifiées.

            s = Signal.depuis_coefficients(200, [1.59, 2.5, 1.06], [0, np.pi/2, 0])
        """
        amplitudes = np.array(amplitudes, dtype=float)
        phases = np.zeros_like(amplitudes) if phases is None else np.array(phases, dtype=float)
        if phases.shape != amplitudes.shape:
            raise ValueError("autant de phases que d'amplitudes, composante continue comprise")
        return cls(
            f0=f0,
            spectre=lambda n: (amplitudes[n], phases[n]),
            moyenne=amplitudes[0],
            nmax=amplitudes.size - 1,
        )

    def __call__(self, t):
        """Le signal aux instants t (un nombre ou un tableau)."""
        t = np.asarray(t, dtype=float)
        s = np.full(t.shape, self.moyenne)
        for A, w, phi in zip(self.amplitudes, self.pulsations, self.phases):
            if A:
                s += A * np.cos(w * t + phi)
        return s[()]

    @property
    def Veff(self):
        """La valeur efficace, par Parseval : racine de moyenne² + somme des A_n²/2."""
        return float(np.sqrt(self.moyenne**2 + np.sum(self.amplitudes**2) / 2))

    def filtre(self, fonction_transfert):
        """Le signal en sortie du filtre de fonction de transfert H(f)
        (complexe, vectorisée en f) : chaque harmonique multipliée par
        |H(f_n)| et déphasée de arg H(f_n), la moyenne multipliée par H(0),
        signe compris. Rend un Signal."""
        H = np.asarray(fonction_transfert(self.frequences), dtype=complex)
        moyenne = self.moyenne * _gain_continu(fonction_transfert, self.f0) if self.moyenne else 0.0
        amplitudes = self.amplitudes * np.abs(H)
        phases = self.phases + np.angle(H)
        return Signal(
            f0=self.f0, spectre=lambda n: (amplitudes[n - 1], phases[n - 1]), moyenne=moyenne, nmax=self.nmax
        )

    def __repr__(self):
        return f"Signal(f0={self.f0:g}, moyenne={self.moyenne:g}, nmax={self.nmax}, Veff={self.Veff:.4g})"


def _gain_continu(fonction_transfert, f0):
    """H(0), réel pour un filtre physique (H(-f) est le conjugué de H(f)). Une
    fonction écrite avec f0/f ne se calcule pas en 0 : on prend alors sa
    limite, H à un milliardième de f0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = complex(np.asarray(fonction_transfert(np.array([0.0])), dtype=complex).ravel()[0])
        if not np.isfinite(h):
            h = complex(np.asarray(fonction_transfert(np.array([1e-9 * f0])), dtype=complex).ravel()[0])
    return h.real


def spectre_carre(n):
    """Le créneau entre -1 et 1, pair, qui vaut 1 autour de t = 0 : les rangs
    impairs, d'amplitude 4/(n pi), en phase (n = 1, 5, 9…) ou en opposition
    (n = 3, 7…) ; les pairs sont nuls."""
    n = np.asarray(n)
    return np.where(n % 2 == 1, 4 / (n * np.pi), 0.0), np.where(n % 4 == 3, np.pi, 0.0)


def spectre_triangle(n):
    """Le triangle entre -1 et 1, pair, qui vaut 1 en t = 0 : les rangs
    impairs, d'amplitude 8/(n pi)², en phase."""
    n = np.asarray(n)
    return np.where(n % 2 == 1, 8 / (n * np.pi) ** 2, 0.0), np.zeros(n.shape)


def spectre_dent_de_scie(n):
    """La dent de scie qui monte de -1 à 1, nulle en t = 0 et qui retombe en
    t = T/2 : tous les rangs, d'amplitude 2/(n pi), sin(n w t) pour n impair
    et -sin(n w t) pour n pair."""
    n = np.asarray(n)
    return 2 / (n * np.pi), np.where(n % 2 == 1, -np.pi / 2, np.pi / 2)


def _x(f, f0):
    return np.asarray(f, dtype=float) / f0


def passe_bas_1(fc, H0=1.0):
    """H(f) = H0/(1 + j f/fc)."""

    def H(f):
        return H0 / (1 + 1j * _x(f, fc))

    return H


def passe_haut_1(fc, H0=1.0):
    """H(f) = H0 j(f/fc)/(1 + j f/fc)."""

    def H(f):
        x = _x(f, fc)
        return H0 * 1j * x / (1 + 1j * x)

    return H


def passe_bas_2(f0, Q, H0=1.0):
    """H(f) = H0/(1 - x² + j x/Q), x = f/f0."""

    def H(f):
        x = _x(f, f0)
        return H0 / (1 - x**2 + 1j * x / Q)

    return H


def passe_haut_2(f0, Q, H0=1.0):
    """H(f) = -H0 x²/(1 - x² + j x/Q), x = f/f0."""

    def H(f):
        x = _x(f, f0)
        return -H0 * x**2 / (1 - x**2 + 1j * x / Q)

    return H


def passe_bande(f0, Q, H0=1.0):
    """H(f) = H0/(1 + jQ(x - 1/x)), x = f/f0 ; écrite H0 x/(x + jQ(x² - 1))
    pour valoir 0, sans division par zéro, en f = 0."""

    def H(f):
        x = _x(f, f0)
        return H0 * x / (x + 1j * Q * (x**2 - 1))

    return H


def coupe_bande(f0, Q, H0=1.0):
    """H(f) = H0 (1 - x²)/(1 - x² + j x/Q), x = f/f0 : nulle en f0."""

    def H(f):
        x = _x(f, f0)
        return H0 * (1 - x**2) / (1 - x**2 + 1j * x / Q)

    return H
