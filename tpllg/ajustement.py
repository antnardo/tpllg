# -*- coding: utf-8 -*-
"""
Ajustements de courbes : outils génériques autour de scipy.optimize.curve_fit.

- curvefit : curve_fit avec les incertitudes sur y, ou sur x et y (méthode de
  la variance effective), et le chi2 réduit — le curvefit de dataanalysis
  (2018), fusionné ici ;
- curve_fit_complex : ajuster une grandeur complexe mesurée par son module et
  sa phase, parties réelle et imaginaire ajustées ensemble par curvefit,
  mêmes arguments et même retour ;
- ecarts_types : les incertitudes-types tirées de la matrice de covariance ;
- formater, resume_parametres : « valeur ± incertitude », arrondies comme il
  faut (deux chiffres significatifs sur l'incertitude) ;
- residus_complexes : l'écart des mesures au modèle, module et phase.

@author: a. marchand
"""
import numpy as np
from scipy import optimize

__all__ = ["curvefit", "curve_fit_complex", "ecarts_types", "formater", "resume_parametres",
           "residus_complexes"]


def _tableau(valeur, forme):
    """Une incertitude ramenée à la forme des données : un nombre vaut pour
    tous les points, un tableau est rendu tel quel."""
    return np.broadcast_to(np.asarray(valeur, dtype=float), forme)


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


def curvefit(function, datax, datay, p0, datayerrors=None, dataxerrors=None, function_derivate=None,  # noqa C901
             n_var_method_max=10, chi_limit=0.01, verbose=True, **kwargs):
    """ Curve fitting wrapped up around scipy.optimize.curve_fit with better error management

    Arguments:
        - function must be called as function(datax, a, b) where p0 = [a, b].
            Vectorized in datax or not : a function written with math.exp is
            called point by point.
        - datax : array of data
        - datay : array of data, should be same length as datax
        - p0 : array of first guess parameters to be passed to function in which respect the fitting is done
        - datayerrors : sigma values for y : a number, the same for every point,
            or an array of same length as data. Should be only positive values.
            If None, considered 1.
        - dataxerrors : sigma values for x, a number or an array of same length
            as data. Should be only positive values. If None, considered 0.
        - function_derivate : same arguments as function, returns the derivative
            with respect to x ; may return a number when it is constant
            (`return a` for a straight line)
        - n_var_method_max : max number of loops (default 10). For management of dataxerrors only
        - chi_limit : if a loop doesnot improves the chi squre reduced by more than chi_limit,
            it stops looping (default 0.01).
            For management of dataxerrors only
        - verbose : False pour taire la méthode employée
        - kwargs are passed to optimize.curve_fit

    Returns:
        - pfit
        - err : array of errors on pfit parameters
        - chi_squared_reduced

    Notes:
    - sans datayerrors : pcov est mise à l'échelle des résidus (comme curve_fit),
      et le chi2 réduit rendu, calculé avec sigma = 1, n'a pas de sens
    - least square method with only y errors or none at all (yerror=1, cf scipy)
    - effective variance method if x and y errors provided
        in this case, the derivate of the fit function with respect to x must be given

    Effective variance method : Least squares when both variables have uncertainties
    Jay Orear, Am. J. Phys. 50, 912 (1982); doi: 10.1119/1.12972
        -> same method as least squre but loop over derivate : stops at n_var_method_max iterations
        or when chi squared does not evolves much than chi_limit

    Notes on the method for LINEAR REGRESSION:
        - with CONSTANT sigma values (only y or x and y), the values of PFIT are THE SAME.
            Indeed minimization is independant of the dataxerror and datayerror when constant
            It only affects the error on those values and chi squared value
        - with VARIABLE sigma values : the methods will lead to different parameters values
            BUT if dataxerror is proportionnal to datayerror : the fit parameters will be se same
            and of course it will affect the errors and chi_squared values

    """
    datax = np.asarray(datax, dtype=float)
    datay = np.asarray(datay, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    if dataxerrors is not None and function_derivate is None:
        raise NotImplementedError('Pour utiliser des erreurs en x, il faut indiquer '
                                  'la dérivée de la fonction par rapport à x '
                                  '(effective variance method)')
    if dataxerrors is not None and datayerrors is None:
        raise NotImplementedError("Mettre les erreurs en y lorsqu'il n'y en a qu'un type")
    assert type(n_var_method_max) is int and n_var_method_max > 0

    # le modèle, la dérivée et les incertitudes ramenés à la forme des données
    function = _vectorisee(function, datax, p0)
    if datayerrors is not None:
        datayerrors = _tableau(datayerrors, datax.shape)
    if dataxerrors is not None:
        dataxerrors = _tableau(dataxerrors, datax.shape)
        function_derivate = _vectorisee(function_derivate, datax, p0)

    # intermediate calculus functions
    def errfunc(p, x, y):
        return function(x, *p) - y

    def calc_chi_sq_red(p):
        if datayerrors is None:
            return (errfunc(p, datax, datay)**2).sum()/(len(datay)-len(p0))
        elif dataxerrors is None:
            return ((errfunc(p, datax, datay)/datayerrors)**2).sum()/(len(datay)-len(p0))
        else:
            return (errfunc(p, datax, datay)**2
                    / (datayerrors**2+dataxerrors**2*function_derivate(datax, *p)**2)
                    ).sum()/(len(datay)-len(p0))

    # Fitting
    if dataxerrors is not None:  # and datayerrors is not None and function_derivate is not None
        # Effective variance method
        if verbose:
            print("Effective variance method")
        pfit = p0.copy()
        chi = calc_chi_sq_red(pfit)
        chi_prev = chi+2*chi_limit
        i = 0
        # loop where sigma is recalculted with new guesses of p
        while chi < chi_prev and chi_prev-chi > chi_limit and i < n_var_method_max:
            sigma = np.sqrt(datayerrors**2 + dataxerrors**2 * function_derivate(datax, *pfit)**2)
            pfit, pcov = optimize.curve_fit(
                function, datax, datay, p0=pfit, sigma=sigma, absolute_sigma=True, **kwargs)
            chi_prev = chi
            chi = calc_chi_sq_red(pfit)
            i += 1
        if chi_prev < chi and verbose:
            print('Warning : effective variance method not efficient. Try guessing a better p0.')
    else:
        # Least square method
        if verbose:
            print("Least square method")
        # sans incertitudes fournies, pcov est mise à l'échelle des résidus,
        # comme le fait curve_fit ; avec, elle est absolue
        pfit, pcov = optimize.curve_fit(
            function, datax, datay, p0=p0, sigma=datayerrors,
            absolute_sigma=datayerrors is not None, **kwargs)
    # curve_fit notes :
    # absolute_sigma = bool, optional :
        # If True, sigma is used in an absolute sense and the estimated parameter covariance pcov reflects these
        # absolute values.
        # If False, only the relative magnitudes of the sigma values matter. The returned parameter covariance matrix
        # pcov is based on scaling sigma by a constant factor. This constant is set by demanding that the reduced
        # chisq for the optimal parameters popt when using the scaled sigma equals unity. In other words, sigma is
        # scaled to match the sample variance of the residuals after the fit.
        # Mathematically, pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)

    # Errors on pfit parameters
    err = np.sqrt(np.diag(pcov))

    chi_sq_reduced = calc_chi_sq_red(pfit)
    return pfit, err, chi_sq_reduced


def curve_fit_complex(complex_func, datax, norm, phase, p0, datayerrors=None, dataxerrors=None,
                      function_derivate=None, **kwargs):
    """Ajuste un modèle complexe à des mesures données par leur module `norm`
    et leur phase `phase` (radians) : les parties réelle et imaginaire sont
    empilées bout à bout et ajustées d'un seul coup par curvefit, dont c'est
    l'interface — mêmes arguments, même retour (pfit, err, chi2 réduit).

    complex_func(x, *params) doit rendre un tableau complexe ;
    function_derivate, sa dérivée par rapport à x, complexe aussi.
    datayerrors est le couple (u_norm, u_phase) des incertitudes-types sur le
    module et sur la phase (radians), chacune un nombre ou un tableau : elles
    sont propagées aux parties réelle et imaginaire. Les autres mots-clés
    (verbose, maxfev…) vont à curvefit.
    """
    datax = np.asarray(datax, dtype=float)
    norm = np.asarray(norm, dtype=float)
    phase = np.asarray(phase, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    n = datax.size
    x_empile = np.hstack((datax, datax))
    y_empile = np.hstack((norm*np.cos(phase), norm*np.sin(phase)))

    def empile(f):
        f = _vectorisee(f, datax, p0, dtype=complex)

        def g(x, *p):
            y = f(x[:n], *p)
            return np.hstack((np.real(y), np.imag(y)))
        return g

    erreurs = None
    if datayerrors is not None:
        try:
            u_norm, u_phase = (_tableau(u, norm.shape) for u in datayerrors)
        except (TypeError, ValueError):
            raise ValueError("datayerrors : le couple (u_norm, u_phase) des incertitudes sur le "
                             "module et sur la phase, chacune un nombre ou un tableau")
        u_re = np.hypot(np.cos(phase)*u_norm, norm*np.sin(phase)*u_phase)
        u_im = np.hypot(np.sin(phase)*u_norm, norm*np.cos(phase)*u_phase)
        erreurs = np.hstack((u_re, u_im))
    if dataxerrors is not None:
        dataxerrors = np.hstack([_tableau(dataxerrors, datax.shape)]*2)
    derivee = empile(function_derivate) if function_derivate is not None else None
    return curvefit(empile(complex_func), x_empile, y_empile, p0, datayerrors=erreurs,
                    dataxerrors=dataxerrors, function_derivate=derivee, **kwargs)


def ecarts_types(pcov):
    """Les incertitudes-types des paramètres : racine de la diagonale de pcov."""
    return np.sqrt(np.diag(pcov))


def formater(valeur, sigma=None, unite=""):
    """« 1993.5 ± 1.9 Hz » : l'incertitude à deux chiffres significatifs, la
    valeur arrondie au même rang. Sans incertitude, quatre chiffres."""
    unite = f" {unite}" if unite else ""
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return f"{valeur:.4g}{unite}"
    decimales = 1 - int(np.floor(np.log10(sigma)))    # négatif au-delà de 100
    valeur, sigma = round(valeur, decimales), round(sigma, decimales)
    decimales = max(0, decimales)
    return f"{valeur:.{decimales}f} ± {sigma:.{decimales}f}{unite}"


def resume_parametres(noms, pfit, err=None, unites=None):
    """Une ligne par paramètre, « nom = valeur ± incertitude unité », par
    formater. `err` : les incertitudes-types, une par paramètre (le `err` de
    curvefit), ou la matrice de covariance que rend scipy.optimize.curve_fit,
    dont on prend la racine de la diagonale ; sans, les valeurs seules."""
    if err is None:
        sigmas = [None]*len(pfit)
    else:
        err = np.asarray(err, dtype=float)
        sigmas = ecarts_types(err) if err.ndim == 2 else err
    unites = unites if unites is not None else [""]*len(pfit)
    return "\n".join(f"{nom} = {formater(v, s, u)}"
                     for nom, v, s, u in zip(noms, pfit, sigmas, unites))


def residus_complexes(complex_func, x, norm, phase, pfit):
    """L'écart des mesures au modèle ajusté : (écart relatif sur le module,
    écart de phase en degrés)."""
    y = complex_func(np.asarray(x, dtype=float), *pfit)
    norm = np.asarray(norm, dtype=float)
    phase = np.asarray(phase, dtype=float)
    res_norm = (norm - np.abs(y))/norm
    res_phase = np.degrees(np.angle(np.exp(1j*phase)*np.abs(y)/y))
    return res_norm, res_phase
