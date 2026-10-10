"""
Ajustements de courbes : outils génériques autour de scipy.optimize.curve_fit.

- curvefit : curve_fit avec les incertitudes sur y, ou sur x et y (méthode de
  la variance effective), et le chi2 réduit — le curvefit de dataanalysis
  (2018), fusionné ici ;
- curve_fit_complex : ajuster une grandeur complexe mesurée par son module et
  sa phase, mêmes arguments et même retour ;
- regression_york : la droite y = a x + b quand x et y sont incertains, par
  la solution exacte de York (2004) — ce que faisait scipy.odr, retiré de
  SciPy 1.19 ;
- Ajustement : ce que rendent ces trois fonctions, qui se dépaquette en
  (pfit, err, chi2) ;
- ecarts_types : les incertitudes-types tirées de la matrice de covariance ;
- formater, resume_parametres : « valeur ± incertitude », arrondies comme il
  faut (deux chiffres significatifs sur l'incertitude) ;
- residus_complexes : l'écart des mesures au modèle, module et phase.

@author: a. marchand
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from tpllg._interne import incertitude

__all__ = [
    "Ajustement",
    "curve_fit_complex",
    "curvefit",
    "ecarts_types",
    "formater",
    "regression_york",
    "residus_complexes",
    "resume_parametres",
]


@dataclass(frozen=True, eq=False)
class Ajustement:
    """Le résultat d'un ajustement : les paramètres `pfit`, leurs
    incertitudes-types `err`, le chi2 réduit `chi2` et la matrice de
    covariance `pcov` (sa diagonale est err²).

    Il se dépaquette comme le triplet des versions précédentes :
    `pfit, err, chi2 = curvefit(...)` marche toujours, et `curvefit(...)[0]`
    est pfit ; `curvefit(...).pcov` donne en plus les covariances, utiles à la
    bande d'incertitude d'une courbe ajustée."""

    pfit: np.ndarray
    err: np.ndarray
    chi2: float
    pcov: np.ndarray

    def __iter__(self):
        return iter((self.pfit, self.err, self.chi2))

    def __getitem__(self, indice):
        return tuple(self)[indice]

    def __len__(self):
        return 3


def _vectorisee(f, datax, p0, dtype=float):
    """f(x, *p) rendue en tableau de la forme de x, quoi que f fasse d'un
    tableau : une fonction écrite avec math.exp est appelée point par point,
    une dérivée constante (`return a`) est étendue à tous les points."""

    def etendue(g):
        def h(x, *p):
            return np.broadcast_to(np.asarray(g(x, *p), dtype=dtype), np.shape(x))

        return h

    try:
        essai = np.asarray(f(datax, *p0), dtype=dtype)
    except (ValueError, TypeError):
        return etendue(np.vectorize(f, otypes=[dtype]))
    if essai.shape in ((), np.shape(datax)):
        return etendue(f)
    return etendue(np.vectorize(f, otypes=[dtype]))


def _chi2_reduit(residus, sigma, nb_parametres):
    libertes = residus.size - nb_parametres
    if libertes <= 0:
        return math.nan
    if sigma is None:
        return float(np.sum(residus**2) / libertes)
    return float(np.sum((residus / sigma) ** 2) / libertes)


def curvefit(
    function,
    datax,
    datay,
    p0,
    datayerrors=None,
    dataxerrors=None,
    function_derivate=None,
    n_var_method_max=10,
    chi_limit=0.01,
    verbose=False,
    u_y=None,
    u_x=None,
    **kwargs,
):
    """Ajuste function(x, *p) aux mesures (datax, datay) par curve_fit, avec
    leurs incertitudes, et rend un Ajustement (pfit, err, chi2, pcov).

    function : appelée function(x, a, b…) avec p0 = [a, b…] ; vectorisée en x
        ou non — une fonction écrite avec math.exp est appelée point par point ;
    p0 : les valeurs de départ, à lire sur un tracé avant d'ajuster ;
    datayerrors, ou u_y : les incertitudes-types sur y, un nombre pour tous
        les points ou un tableau ; sans elles, pcov est mise à l'échelle des
        résidus (comme le fait curve_fit) et le chi2 rendu est la variance des
        résidus, pas un chi2 ;
    dataxerrors, ou u_x : les incertitudes-types sur x. Elles sont ramenées
        en y par la pente du modèle — la variance effective,
        sigma² = sigma_y² + (f'(x) sigma_x)² (J. Orear, Am. J. Phys. 50, 912,
        1982) — que l'on recalcule à chaque ajustement, au plus
        n_var_method_max fois, jusqu'à ce que le chi2 réduit ne baisse plus de
        chi_limit. Le meilleur des ajustements est gardé ;
    function_derivate : la dérivée de function par rapport à x, mêmes
        arguments (`return a` suffit pour une droite). Sans elle, la pente est
        prise sur le modèle lui-même, (f(x + sigma_x) - f(x - sigma_x))/2 ;
    verbose : True pour imprimer la méthode employée ;
    les autres mots-clés (maxfev, bounds…) vont à scipy.optimize.curve_fit.

    `u_x` et `u_y` sont les noms courts de `dataxerrors` et `datayerrors`,
    ceux de regression_york et de montecarlo ; les deux écritures valent,
    pas ensemble.

    Pour une droite, des incertitudes constantes ne changent pas pfit, mais
    seulement err et le chi2 ; des incertitudes variables changent aussi pfit.
    La variance effective itère à poids figés et s'arrête un peu à côté du
    minimum de sum (y - f(x))²/sigma² : pour une droite, regression_york
    trouve ce minimum exactement.
    """
    datax = np.asarray(datax, dtype=float)
    datay = np.asarray(datay, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    datayerrors, dataxerrors = _un_seul_nom(datayerrors, u_y, "y"), _un_seul_nom(dataxerrors, u_x, "x")
    if isinstance(n_var_method_max, bool) or not isinstance(n_var_method_max, int) or n_var_method_max < 1:
        raise ValueError(f"n_var_method_max = {n_var_method_max!r} : un entier au moins égal à 1")
    function = _vectorisee(function, datax, p0)
    sigma_y = None if datayerrors is None else incertitude(datayerrors, datax.shape)

    if dataxerrors is None:
        if verbose:
            print("Moindres carrés" + (" pondérés par les incertitudes sur y" if sigma_y is not None else ""))
        _verifier_sigma(sigma_y)
        # sans incertitudes fournies, pcov est mise à l'échelle des résidus,
        # comme le fait curve_fit ; avec, elle est absolue
        pfit, pcov = optimize.curve_fit(
            function, datax, datay, p0=p0, sigma=sigma_y, absolute_sigma=sigma_y is not None, **kwargs
        )
        chi2 = _chi2_reduit(datay - function(datax, *pfit), sigma_y, p0.size)
        return Ajustement(pfit, ecarts_types(pcov), chi2, pcov)

    if verbose:
        print("Variance effective : les incertitudes sur x ramenées en y par la pente du modèle")
    sigma_x = incertitude(dataxerrors, datax.shape)
    variance_y = 0.0 if sigma_y is None else sigma_y**2
    if function_derivate is not None:
        derivee = _vectorisee(function_derivate, datax, p0)

        def report(p):
            return derivee(datax, *p) * sigma_x
    else:

        def report(p):
            return (function(datax + sigma_x, *p) - function(datax - sigma_x, *p)) / 2

    def sigma_effectif(p):
        return np.sqrt(variance_y + report(p) ** 2)

    # La boucle s'arrête quand le chi2 réduit ne baisse plus de chi_limit. Le
    # dernier ajustement est rendu : ses poids viennent des paramètres de
    # l'avant-dernier, les plus proches des siens. S'il est moins bon que le
    # meilleur (la méthode diverge), c'est le meilleur qui est rendu.
    meilleur = dernier = precedent = None
    pfit = p0
    for _ in range(n_var_method_max):
        sigma = sigma_effectif(pfit)
        _verifier_sigma(sigma)
        pfit, pcov = optimize.curve_fit(
            function, datax, datay, p0=pfit, sigma=sigma, absolute_sigma=True, **kwargs
        )
        chi2 = _chi2_reduit(datay - function(datax, *pfit), sigma_effectif(pfit), p0.size)
        dernier = Ajustement(pfit, ecarts_types(pcov), chi2, pcov)
        if meilleur is None or _rang(chi2) < _rang(meilleur.chi2):
            meilleur = dernier
        if precedent is not None and precedent - chi2 <= chi_limit:
            break
        precedent = chi2
    if _rang(dernier.chi2) > _rang(meilleur.chi2) + max(chi_limit, 0):
        if verbose:
            print("Attention : la variance effective diverge ; meilleur ajustement gardé, revoir p0")
        return meilleur
    return dernier


def _un_seul_nom(long, court, axe):
    """dataxerrors ou u_x, datayerrors ou u_y : la valeur donnée, et pas les deux."""
    if long is not None and court is not None:
        raise ValueError(f"data{axe}errors et u_{axe} sont la même chose : n'en donnez qu'un")
    return court if long is None else long


def _radians(phase):
    """Les phases en radians, comme np.angle les rend : une phase en degrés
    passerait en silence et fausserait tout, on refuse ce qui dépasse 2 pi."""
    phase = np.asarray(phase, dtype=float)
    if phase.size and np.nanmax(np.abs(phase)) > 2 * np.pi:
        raise ValueError(
            f"les phases doivent être en radians : max|phase| = {np.nanmax(np.abs(phase)):.3g} > 2 pi "
            "(np.radians convertit des degrés ; tpllg.bode.phase_repliee replie une phase déroulée)"
        )
    return phase


def _rang(chi2):
    """Un chi2 indéfini compte comme le pire."""
    return math.inf if math.isnan(chi2) else chi2


def _verifier_sigma(sigma):
    if sigma is not None and not np.all(sigma > 0):
        i = int(np.flatnonzero(~(sigma > 0))[0])
        raise ValueError(f"incertitude nulle, négative ou non définie au point {i} : {sigma[i]}")


def curve_fit_complex(
    complex_func,
    datax,
    norm,
    phase,
    p0,
    datayerrors=None,
    dataxerrors=None,
    function_derivate=None,
    u_y=None,
    u_x=None,
    **kwargs,
):
    """Ajuste un modèle complexe à des mesures données par leur module `norm`
    et leur phase `phase` (radians), par curvefit, dont c'est l'interface :
    mêmes arguments, même retour (pfit, err, chi2 réduit, pcov).

    Ce qui est ajusté, ce sont ln|H| et la phase, ensemble : un écart relatif
    sur le module et un écart de phase, indépendants comme le sont les deux
    mesures. (Ajuster les parties réelle et imaginaire mélangerait les deux
    incertitudes et ignorerait leur corrélation : jusqu'à 15 % d'erreur sur
    les incertitudes rendues.) La phase du modèle est prise à moins d'un
    demi-tour de chaque mesure : une phase à 2 pi près ne gêne pas. Le
    départ est d'abord amélioré par un ajustement des parties réelle et
    imaginaire, qui corrige un signe de H0 faux ou un départ un peu loin.

    complex_func(x, *params) doit rendre un tableau complexe ;
    function_derivate, sa dérivée par rapport à x, complexe aussi — ou rien,
    la pente est alors prise sur le modèle. datayerrors est le couple
    (u_norm, u_phase) des incertitudes-types sur le module et sur la phase
    (radians), chacune un nombre ou un tableau ; u_y et u_x en sont les noms
    courts. Les autres mots-clés (verbose, maxfev…) vont à curvefit.

    Sans incertitudes, un écart de 1 % sur le module pèse autant qu'un écart
    de 0,01 rad (0,6°) sur la phase. Une phase qui dépasse 2 pi en valeur
    absolue est refusée (ValueError) : elle est en degrés, np.radians la
    convertit.
    """
    datax = np.asarray(datax, dtype=float)
    norm = np.asarray(norm, dtype=float)
    phase = _radians(phase)
    p0 = np.asarray(p0, dtype=float)
    datayerrors, dataxerrors = _un_seul_nom(datayerrors, u_y, "y"), _un_seul_nom(dataxerrors, u_x, "x")
    if not np.all(norm > 0):
        raise ValueError("norm : les modules mesurés doivent être strictement positifs")
    n = datax.size
    f = _vectorisee(complex_func, datax, p0, dtype=complex)
    p0 = _depart_complexe(f, datax, norm * np.exp(1j * phase), p0, kwargs)

    def modele(x, *p):
        H = f(x[:n], *p)
        return np.hstack((np.log(np.abs(H)), phase + np.angle(H * np.exp(-1j * phase))))

    derivee = None
    if function_derivate is not None:
        d = _vectorisee(function_derivate, datax, p0, dtype=complex)

        def derivee(x, *p):
            r = d(x[:n], *p) / f(x[:n], *p)
            return np.hstack((r.real, r.imag))

    erreurs = None
    if datayerrors is not None:
        try:
            u_norm, u_phase = (incertitude(u, norm.shape) for u in datayerrors)
        except (TypeError, ValueError):
            raise ValueError(
                "datayerrors : le couple (u_norm, u_phase) des incertitudes sur le "
                "module et sur la phase, chacune un nombre ou un tableau"
            ) from None
        erreurs = np.hstack((u_norm / norm, u_phase))
    if dataxerrors is not None:
        dataxerrors = np.hstack([incertitude(dataxerrors, datax.shape)] * 2)
    return curvefit(
        modele,
        np.hstack((datax, datax)),
        np.hstack((np.log(norm), phase)),
        p0,
        datayerrors=erreurs,
        dataxerrors=dataxerrors,
        function_derivate=derivee,
        **kwargs,
    )


def _depart_complexe(f, x, mesures, p0, kwargs):
    """Le départ de curve_fit_complex : un premier ajustement des parties
    réelle et imaginaire, sans pondération. Il est lisse partout et peut
    faire passer H0 par zéro, ce que ln|H| interdit : un H0 de signe faux, un
    départ un peu loin se corrigent ici. Il ne sert que de point de départ ;
    s'il échoue, on part de p0."""
    options = {k: v for k, v in kwargs.items() if k not in ("verbose", "n_var_method_max", "chi_limit")}

    def re_im(xx, *p):
        H = f(x, *p)
        return np.hstack((H.real, H.imag))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            depart = optimize.curve_fit(
                re_im, np.hstack((x, x)), np.hstack((mesures.real, mesures.imag)), p0=p0, **options
            )[0]
    except (RuntimeError, ValueError):
        return p0
    return depart if np.all(np.isfinite(depart)) else p0


def _york(x, vx, y, vy, iterations=100, tolerance=1e-12):
    """La droite de York sur le dernier axe : x et y de forme (..., P), les
    variances vx et vy de même forme. Calcule d'un coup autant de droites que
    de lignes — une par tirage pour le Monte-Carlo.

    York, Evensen, López Martínez et De Basabe Delgado, « Unified equations
    for the slope, intercept, and standard errors of the best straight line »,
    Am. J. Phys. 72, 367 (2004), sans corrélation entre les erreurs sur x et
    sur y. Rend (pente, ordonnée, variance de la pente, variance de
    l'ordonnée, covariance, chi2)."""

    def moyenne(w, z):
        return (w * z).sum(-1, keepdims=True) / w.sum(-1, keepdims=True)

    xm, ym = x.mean(-1, keepdims=True), y.mean(-1, keepdims=True)
    pente = ((x - xm) * (y - ym)).sum(-1) / ((x - xm) ** 2).sum(-1)  # les moindres carrés pour partir
    for _ in range(iterations):
        b = pente[..., None]
        w = 1 / (vy + b**2 * vx)
        u, v = x - moyenne(w, x), y - moyenne(w, y)
        beta = w * (u * vy + b * v * vx)
        nouvelle = (w * beta * v).sum(-1) / (w * beta * u).sum(-1)
        fini = np.all(np.abs(nouvelle - pente) <= tolerance * np.abs(nouvelle))
        pente = nouvelle
        if fini:
            break
    b = pente[..., None]
    w = 1 / (vy + b**2 * vx)
    x_bar, y_bar = moyenne(w, x), moyenne(w, y)
    beta = w * ((x - x_bar) * vy + b * (y - y_bar) * vx)
    ordonnee = (y_bar - b * x_bar)[..., 0]
    x_ajuste = x_bar + beta  # les abscisses des points ramenés sur la droite
    centre = moyenne(w, x_ajuste)
    var_pente = 1 / (w * (x_ajuste - centre) ** 2).sum(-1)
    var_ordonnee = 1 / w.sum(-1) + centre[..., 0] ** 2 * var_pente
    covariance = -centre[..., 0] * var_pente
    chi2 = (w * (y - b * x - ordonnee[..., None]) ** 2).sum(-1)
    return pente, ordonnee, var_pente, var_ordonnee, covariance, chi2


def regression_york(x, u_x, y, u_y):
    """La droite y = a x + b ajustée sur des points dont x et y sont tous
    deux incertains : u_x et u_y, les incertitudes-types, un nombre pour tous
    les points ou un tableau. Rend un Ajustement : pfit = (a, b), leurs
    incertitudes-types, le chi2 réduit et la covariance.

    C'est la solution exacte des moindres carrés pondérés par les deux
    incertitudes (York et al., Am. J. Phys. 72, 367, 2004), sans corrélation
    entre les erreurs sur x et sur y. Elle remplace scipy.odr, retiré de
    SciPy 1.19, pour une droite ; sans incertitude sur x, elle se réduit aux
    moindres carrés pondérés, et sans incertitude du tout elle n'a pas de
    sens. Pour un modèle quelconque, curvefit avec dataxerrors.

    Les incertitudes rendues découlent de u_x et u_y (absolues) : un chi2
    réduit loin de 1 dit qu'elles sont mal estimées.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("x et y : deux tableaux 1D de même longueur, au moins trois points")
    u_x, u_y = incertitude(u_x, x.shape), incertitude(u_y, y.shape)
    if not (np.all(u_x >= 0) and np.all(u_y >= 0)) or np.any((u_x == 0) & (u_y == 0)):
        raise ValueError("u_x et u_y : positives, et pas nulles toutes les deux en un même point")
    vx, vy = u_x**2, u_y**2
    pente, ordonnee, var_a, var_b, cov, chi2 = _york(x, vx, y, vy)
    pfit = np.array([pente, ordonnee])
    pcov = np.array([[var_a, cov], [cov, var_b]])
    return Ajustement(pfit, ecarts_types(pcov), float(chi2 / (x.size - 2)), pcov)


def ecarts_types(pcov):
    """Les incertitudes-types des paramètres : racine de la diagonale de pcov."""
    return np.sqrt(np.diag(pcov))


_EXPOSANTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_SCIENTIFIQUE_SOUS, _SCIENTIFIQUE_DES = 1e-3, 1e5  # en deçà et à partir de là, « × 10ⁿ »


def _exposant(reference):
    """L'exposant de la notation scientifique pour une grandeur de l'ordre de
    `reference`, ou None dans [1e-3, 1e5[, où l'on écrit le nombre tel quel."""
    reference = abs(reference)
    if reference == 0 or _SCIENTIFIQUE_SOUS <= reference < _SCIENTIFIQUE_DES:
        return None
    return math.floor(math.log10(reference))


def _puissance(exposant):
    return "10" + str(exposant).translate(_EXPOSANTS)


def _arrondis(valeur, sigma):
    """La valeur et l'incertitude arrondies au rang du second chiffre
    significatif de l'incertitude, en texte, et le nombre de décimales."""
    sigma = float(f"{sigma:.2g}")  # deux chiffres : 0.0996 devient 0.10, pas 0.100
    decimales = 1 - math.floor(math.log10(sigma))  # négatif au-delà de 100
    valeur, sigma = round(valeur, decimales), round(sigma, decimales)
    decimales = max(0, decimales)
    return f"{valeur:.{decimales}f}", f"{sigma:.{decimales}f}", valeur


def formater(valeur, sigma=None, unite=""):
    """« 1993.5 ± 1.9 Hz » : l'incertitude à deux chiffres significatifs, la
    valeur arrondie au même rang. Sans incertitude (None ou 0), quatre
    chiffres ; une incertitude infinie ou indéfinie s'écrit telle quelle.

    Hors de [10⁻³, 10⁵[, la notation scientifique, avec une puissance de dix
    commune à la valeur et à l'incertitude, celle de la valeur :
    « (6.6260 ± 0.0010) × 10⁻³⁴ J s ». Pour une valeur nulle, la puissance
    est celle de l'incertitude."""
    unite = f" {unite}" if unite else ""
    valeur = float(valeur)
    if sigma is None or sigma == 0:
        exposant = _exposant(valeur)
        if exposant is None:
            return f"{valeur:.4g}{unite}"
        return f"{valeur / 10**exposant:.4g} × {_puissance(exposant)}{unite}"
    if not np.isfinite(sigma):
        return f"{valeur:.4g} ± {sigma}{unite}"
    if sigma < 0:
        raise ValueError(f"incertitude négative : {sigma}")
    exposant = _exposant(valeur if valeur != 0 else sigma)
    if exposant is None:
        v, s, _ = _arrondis(valeur, sigma)
        return f"{v} ± {s}{unite}"
    v, s, arrondie = _arrondis(valeur / 10**exposant, sigma / 10**exposant)
    if abs(arrondie) >= 10:  # 9.9999 arrondi à 10.00 : une décade de plus
        exposant += 1
        v, s, _ = _arrondis(valeur / 10**exposant, sigma / 10**exposant)
    return f"({v} ± {s}) × {_puissance(exposant)}{unite}"


def resume_parametres(noms, pfit, err=None, unites=None):
    """Une ligne par paramètre, « nom = valeur ± incertitude unité », par
    formater. `err` : les incertitudes-types, une par paramètre (le `err` de
    curvefit), ou la matrice de covariance que rend scipy.optimize.curve_fit,
    dont on prend la racine de la diagonale ; sans, les valeurs seules."""
    if err is None:
        sigmas = [None] * len(pfit)
    else:
        err = np.asarray(err, dtype=float)
        sigmas = ecarts_types(err) if err.ndim == 2 else err
    unites = unites if unites is not None else [""] * len(pfit)
    if not len(noms) == len(pfit) == len(sigmas) == len(unites):
        raise ValueError(
            f"{len(noms)} noms, {len(pfit)} valeurs, {len(sigmas)} incertitudes, {len(unites)} unités : "
            "il en faut autant de chaque"
        )
    return "\n".join(f"{nom} = {formater(v, s, u)}" for nom, v, s, u in zip(noms, pfit, sigmas, unites))


def residus_complexes(complex_func, x, norm, phase, pfit):
    """L'écart des mesures au modèle ajusté : (écart relatif sur le module,
    écart de phase en degrés, ramené entre -180 et 180). La phase mesurée est
    en radians ; au-delà de 2 pi en valeur absolue, elle est refusée."""
    y = complex_func(np.asarray(x, dtype=float), *pfit)
    norm = np.asarray(norm, dtype=float)
    phase = _radians(phase)
    res_norm = (norm - np.abs(y)) / norm
    res_phase = np.degrees(np.angle(np.exp(1j * phase) * np.abs(y) / y))
    return res_norm, res_phase
