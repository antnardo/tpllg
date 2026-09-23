import numpy as np

from tpllg.ajustement import curve_fit_complex, curvefit, formater, residus_complexes, resume_parametres


def gain(f, H0, f0, Q):
    return H0/(1 + 1j*Q*(f/f0 - f0/f))


def test_curve_fit_complex_retrouve_les_parametres():
    rng = np.random.RandomState(0)
    f = np.geomspace(200, 20000, 30)
    H = gain(f, -5, 2000, 6)
    norm = np.abs(H)*(1 + rng.normal(0, 0.01, f.size))
    phase = np.angle(H) + rng.normal(0, 0.01, f.size)
    pfit, pcov = curve_fit_complex(gain, f, norm, phase, p0=[-4, 1800, 5])
    assert abs(pfit[0] + 5) < 0.1
    assert abs(pfit[1] - 2000) < 5
    assert abs(pfit[2] - 6) < 0.2
    res_norm, res_phase = residus_complexes(gain, f, norm, phase, pfit)
    assert res_norm.std() < 0.02 and res_phase.std() < 1


def test_curvefit_affine_avec_incertitudes():
    rng = np.random.RandomState(1)
    x = np.linspace(0, 1, 20)
    y = 2*x - 1 + rng.normal(0, 0.05, x.size)
    pfit, err, chi2 = curvefit(lambda x, a, b: a*x + b, x, y, p0=[1, 0],
                               datayerrors=0.05*np.ones(x.size), verbose=False)
    assert abs(pfit[0] - 2) < 3*err[0]
    assert abs(pfit[1] + 1) < 3*err[1]
    assert 0.3 < chi2 < 3


def test_formater_arrondit_sur_l_incertitude():
    assert formater(1993.489, 1.875, "Hz") == "1993.5 ± 1.9 Hz"
    assert formater(-5.0823, 0.0631) == "-5.082 ± 0.063"
    assert formater(6.609, 0.156) == "6.61 ± 0.16"
    assert resume_parametres(("a",), [2.0], sigmas=[0.1]) == "a = 2.00 ± 0.10"


def test_formater_au_dela_de_cent_et_sans_incertitude():
    assert formater(1234567.0, 5432.0, "Hz") == "1234600 ± 5400 Hz"
    assert formater(12345.6, 234.0) == "12350 ± 230"
    assert formater(2.5, None, "V") == "2.5 V"


def test_curvefit_accepte_une_fonction_non_vectorisee():
    import math
    x = np.linspace(0, 5, 20)
    y = 2*np.exp(-x/1.5)
    pfit, err, chi2 = curvefit(lambda x, a, tau: a*math.exp(-x/tau), x, y, p0=[1, 1], verbose=False)
    assert abs(pfit[0] - 2) < 1e-6 and abs(pfit[1] - 1.5) < 1e-6
