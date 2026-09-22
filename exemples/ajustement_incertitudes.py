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
    - kwargs are passed to optimize.curve_fit

Returns:
    - pfit
    - err : array of errors on pfit parameters
    - chi_squared_reduced

Notes:
- no management of errors if datayerrors is not specified
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




TODO:
    - implémenter montecarlo dans le même chose
    - vérifier avec le quartet
    - comparer avec des incertitudes grandes selon x
    - comparer avec les inversions y/x
    - implémenter un lsq sur la direction de l'ellipse ?
"""
from tpllg.ajustement import curvefit
import numpy as np
import matplotlib.pyplot as plt


def modele(x, a, b):
    return a*x + b


def modele_derivee(x, a, b):
    return a


# création des données expérimentales
N = 10  # nb points exp
x = np.linspace(0.1, 2, N)
y = modele(x, 2, -1)
# bruit en x : constant
sigma_x = .05*np.ones(N)
x_noised = x + sigma_x*np.random.randn(N)
# bruit en x : non constant
sigma_x2 = sigma_x*(x+1)
x_noised2 = x + sigma_x2*np.random.randn(N)
# bruit en y
sigma_y = 0.15*np.ones(N)
y_noised = y + sigma_y*np.random.randn(N)
# bruit en y non constant - ne pas prendre proportionnel à x,
# sinon méthodes leastsquare et effective variance identiques
sigma_y2 = sigma_y*(x**2+1)
y_noised2 = y + sigma_y2*np.random.randn(N)


# fit et affichage
fmt = {'fmt': 'o', 'capthick': 2}
x_fit = np.linspace(min(x_noised.min(), x_noised2.min()), max(x_noised.max(), x_noised2.max()), 100)

plt.figure()
plt.title("Bruit en x et y constants")
plt.errorbar(x_noised, y_noised, xerr=sigma_x, yerr=sigma_y, **fmt)
# Fit sans prendre en compte les incertitudes
pfit, errs, chi2 = curvefit(modele, x_noised, y_noised, p0=[1, 0])
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit sans incertitudes\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en ne prenant en compte que l'incertitude sur y = c'est la même chose que précédemment
pfit, errs, chi2 = curvefit(
    modele, x_noised, y_noised, p0=[1, 0], datayerrors=sigma_y)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en prenant en compte les incertitudes sur x et y
pfit, errs, chi2 = curvefit(
    modele, x_noised, y_noised, p0=[1, 0], dataxerrors=sigma_x, datayerrors=sigma_y, function_derivate=modele_derivee)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur x et y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")

plt.legend()
plt.show(block=False)

plt.figure()
plt.title("Bruit en x et y non constants")
plt.errorbar(x_noised2, y_noised2, xerr=sigma_x2, yerr=sigma_y2, **fmt)
# Fit sans prendre en compte les incertitudes
pfit, errs, chi2 = curvefit(modele, x_noised2, y_noised2, p0=[1, 0])
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit sans incertitudes\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en ne prenant en compte que l'incertitude sur y
pfit, errs, chi2 = curvefit(
    modele, x_noised2, y_noised2, p0=[1, 0], datayerrors=sigma_y2)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")
# Fit en prenant en compte les incertitudes sur x et y
pfit, errs, chi2 = curvefit(
    modele, x_noised2, y_noised2, p0=[1, 0],
    dataxerrors=sigma_x2, datayerrors=sigma_y2, function_derivate=modele_derivee)
plt.plot(x_fit, modele(x_fit, *pfit),
         label=f"fit incertitudes sur x et y\n"
         f"a={pfit[0]:.3f}±{errs[0]:.3f} "
         f"b={pfit[1]:.3f}±{errs[1]:.3f} X²={chi2:.1e}")

plt.legend()
plt.show(block=False)

plt.show()
