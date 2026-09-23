# -*- coding: utf-8 -*-
"""
Ajustements de courbes : outils génériques autour de scipy.optimize.curve_fit.

- curvefit : curve_fit avec les incertitudes sur y, ou sur x et y (méthode de
  la variance effective), et le chi2 réduit — le curvefit de dataanalysis
  (2018), fusionné ici ;
- curve_fit_complex : ajuster une grandeur complexe mesurée par son module et
  sa phase, parties réelle et imaginaire ajustées ensemble ;
- ecarts_types : les incertitudes-types tirées de la matrice de covariance ;
- formater, resume_parametres : « valeur ± incertitude », arrondies comme il
  faut (deux chiffres significatifs sur l'incertitude) ;
- residus_complexes : l'écart des mesures au modèle, module et phase.

@author: a. marchand
"""
import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit

__all__ = ["curvefit", "curve_fit_complex", "ecarts_types", "formater", "resume_parametres",
           "residus_complexes"]


def curvefit(function, datax, datay, p0, datayerrors=None, dataxerrors=None, function_derivate=None,  # noqa C901
             n_var_method_max=10, chi_limit=0.01, verbose=True, **kwargs):
    """ Curve fitting wrapped up around scipy.optimize.curve_fit with better error management

    Arguments:
        - function must be called as function(datax, a, b) where p0 = [a, b]
            should be vectorizable in respect to datax, otherwise result non guaranteed
        - datax : array of data
        - datay : array of data, should be same length as datax
        - p0 : array of first guess parameters to be passed to function in which respect the fitting is done
        - datayerrors : sigma values for y. array of same length as data.
            Should be only positive values. If None, considered 1.
        - dataxerrors : sigma values for x. array of same length as data. Should be only positive values.
            If None, considered 0.
        - function_derivate : same arguments as function, returns derivate
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
    if type(datax) is list:
        datax = np.array(datax)
    if type(datay) is list:
        datay = np.array(datay)
    if dataxerrors is not None and function_derivate is None:
        raise NotImplementedError('Pour utiliser des erreurs en x, il faut indiquer '
                                  'la dérivée de la fonction par rapport à x '
                                  '(effective variance method)')
    if dataxerrors is not None and datayerrors is None:
        raise NotImplementedError("Mettre les erreurs en y lorsqu'il n'y en a qu'un type")
    assert type(n_var_method_max) is int and n_var_method_max > 0

    # vectorizable function
    try:
        function(datax, *p0)
    except (ValueError, TypeError):
        base_func = function

        def function_vectorized(x, *p):
            try:
                return np.array([base_func(item, *p) for item in x])
            except TypeError:
                return base_func(x, *p)

        function = function_vectorized

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


def curve_fit_complex(complex_func, x_data, norm, phase, **kwargs):
    """Ajuste un modèle complexe à des mesures données par leur module `norm`
    et leur phase `phase` (radians) : les parties réelle et imaginaire sont
    empilées bout à bout et ajustées d'un seul coup par curve_fit.

    complex_func(x, *params) doit rendre un tableau complexe.
    Même retour que curve_fit : (pfit, pcov). Les mots-clés (p0, maxfev…)
    sont transmis à curve_fit.
    """
    x_data = np.asarray(x_data, dtype=float)
    norm = np.asarray(norm, dtype=float)
    phase = np.asarray(phase, dtype=float)
    n = len(x_data)
    x_stacked = np.hstack((x_data, x_data))
    data_stacked = np.hstack((norm*np.cos(phase), norm*np.sin(phase)))

    def fit_func_stacked(x, *args):
        y = complex_func(x[n:], *args)
        return np.hstack((np.real(y), np.imag(y)))

    return curve_fit(fit_func_stacked, x_stacked, data_stacked, **kwargs)


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


def resume_parametres(noms, pfit, pcov=None, unites=None, sigmas=None):
    """Une ligne par paramètre, « nom = valeur ± incertitude unité ». Les
    incertitudes viennent de `pcov`, ou directement de `sigmas` (le `err`
    rendu par curvefit)."""
    if sigmas is None:
        sigmas = ecarts_types(pcov) if pcov is not None else [None]*len(pfit)
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
