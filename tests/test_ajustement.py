import math

import numpy as np
import pytest

from tpllg.ajustement import (curve_fit_complex, curvefit, ecarts_types, formater, residus_complexes,
                              resume_parametres)


def gain(f, H0, f0, Q):
    return H0/(1 + 1j*Q*(f/f0 - f0/f))


def mesures_bode(bruit_norm=0.01, bruit_phase=0.01, graine=0):
    rng = np.random.RandomState(graine)
    f = np.geomspace(200, 20000, 30)
    H = gain(f, -5, 2000, 6)
    norm = np.abs(H)*(1 + rng.normal(0, bruit_norm, f.size))
    phase = np.angle(H) + rng.normal(0, bruit_phase, f.size)
    return f, norm, phase


def test_curve_fit_complex_retrouve_les_parametres():
    f, norm, phase = mesures_bode()
    pfit, err, chi2 = curve_fit_complex(gain, f, norm, phase, p0=[-4, 1800, 5], verbose=False)
    assert abs(pfit[0] + 5) < 0.1
    assert abs(pfit[1] - 2000) < 5
    assert abs(pfit[2] - 6) < 0.2
    assert err.shape == (3,) and chi2 > 0
    res_norm, res_phase = residus_complexes(gain, f, norm, phase, pfit)
    assert res_norm.std() < 0.02 and res_phase.std() < 1


def test_curve_fit_complex_avec_les_incertitudes_du_module_et_de_la_phase():
    f, norm, phase = mesures_bode(bruit_norm=0.03, bruit_phase=np.radians(3), graine=1)
    pfit, err, chi2 = curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5],
                                        datayerrors=(0.03*norm, np.radians(3)), verbose=False)
    assert 0.4 < chi2 < 2.5                                   # les incertitudes déclarées sont les vraies
    assert abs(pfit[0] + 5) < 3*err[0] and abs(pfit[1] - 2000) < 3*err[1] and abs(pfit[2] - 6) < 3*err[2]
    # des incertitudes doublées ne changent pas les paramètres, doublent err, divisent chi2 par 4
    pfit2, err2, chi2_2 = curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5],
                                            datayerrors=(0.06*norm, np.radians(6)), verbose=False)
    assert np.allclose(pfit2, pfit, rtol=1e-6) and np.allclose(err2, 2*err, rtol=1e-4)
    assert abs(chi2_2 - chi2/4) < 1e-9
    with pytest.raises(ValueError, match="u_norm, u_phase"):
        curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5], datayerrors=0.03*norm, verbose=False)


def test_residus_complexes_nuls_sur_le_modele():
    f = np.geomspace(100, 10000, 20)
    H = gain(f, -5, 2000, 6)
    res_norm, res_phase = residus_complexes(gain, f, np.abs(H), np.angle(H), [-5, 2000, 6])
    assert np.abs(res_norm).max() < 1e-12 and np.abs(res_phase).max() < 1e-9


def test_curvefit_affine_avec_incertitudes():
    rng = np.random.RandomState(1)
    x = np.linspace(0, 1, 20)
    y = 2*x - 1 + rng.normal(0, 0.05, x.size)
    pfit, err, chi2 = curvefit(lambda x, a, b: a*x + b, x, y, p0=[1, 0], datayerrors=0.05, verbose=False)
    assert abs(pfit[0] - 2) < 3*err[0]
    assert abs(pfit[1] + 1) < 3*err[1]
    assert 0.3 < chi2 < 3
    # une incertitude constante vaut le tableau qui la répète
    pfit2, err2, chi2_2 = curvefit(lambda x, a, b: a*x + b, x, y, p0=[1, 0],
                                   datayerrors=0.05*np.ones(x.size), verbose=False)
    assert np.array_equal(pfit, pfit2) and np.array_equal(err, err2) and chi2 == chi2_2


def test_curvefit_sans_incertitudes_retrouve_les_formules_des_moindres_carres():
    rng = np.random.RandomState(4)
    x = np.linspace(0, 2, 25)
    y = 3*x + 0.5 + rng.normal(0, 0.1, x.size)
    pfit, err, _ = curvefit(lambda x, a, b: a*x + b, x, y, p0=[1, 0], verbose=False)
    a, b = np.polyfit(x, y, 1)
    s2 = ((y - (a*x + b))**2).sum()/(x.size - 2)          # la variance des résidus
    sxx = ((x - x.mean())**2).sum()
    assert np.allclose(pfit, [a, b], rtol=1e-8)
    assert abs(err[0] - np.sqrt(s2/sxx)) < 1e-8
    assert abs(err[1] - np.sqrt(s2*(1/x.size + x.mean()**2/sxx))) < 1e-8


def test_curvefit_variance_effective():
    rng = np.random.RandomState(5)
    a0, b0, sx, sy = 2.0, -1.0, 0.03, 0.05
    x_vrai = np.linspace(0, 1, 30)
    x = x_vrai + rng.normal(0, sx, x_vrai.size)
    y = a0*x_vrai + b0 + rng.normal(0, sy, x_vrai.size)
    modele = lambda x, a, b: a*x + b                    # noqa: E731
    derivee = lambda x, a, b: a                         # noqa: E731  — constante : un nombre suffit
    pfit_y, err_y, _ = curvefit(modele, x, y, p0=[1, 0], datayerrors=sy, verbose=False)
    pfit, err, chi2 = curvefit(modele, x, y, p0=[1, 0], datayerrors=sy, dataxerrors=sx,
                               function_derivate=derivee, verbose=False)
    # à incertitudes constantes, les paramètres sont ceux des moindres carrés ordinaires ;
    # seules les incertitudes changent, dans le rapport des sigmas effectifs
    assert np.allclose(pfit, pfit_y, rtol=1e-6)
    facteur = np.sqrt(sy**2 + pfit[0]**2*sx**2)/sy
    assert np.allclose(err, err_y*facteur, rtol=1e-4)
    assert abs(pfit[0] - a0) < 3*err[0] and abs(pfit[1] - b0) < 3*err[1]
    assert 0.4 < chi2 < 2.5
    # la dérivée écrite en tableau donne exactement la même chose
    pfit2, err2, chi2_2 = curvefit(modele, x, y, p0=[1, 0], datayerrors=sy*np.ones(x.size),
                                   dataxerrors=sx*np.ones(x.size),
                                   function_derivate=lambda x, a, b: a*np.ones_like(x), verbose=False)
    assert np.allclose(pfit2, pfit, rtol=1e-12) and np.allclose(err2, err, rtol=1e-12) and abs(chi2_2 - chi2) < 1e-12


def test_curvefit_refuse_les_erreurs_en_x_incompletes():
    x = np.linspace(0, 1, 5)
    with pytest.raises(NotImplementedError):
        curvefit(lambda x, a: a*x, x, 2*x, p0=[1], datayerrors=1.0, dataxerrors=1.0)
    with pytest.raises(NotImplementedError):
        curvefit(lambda x, a: a*x, x, 2*x, p0=[1], dataxerrors=1.0, function_derivate=lambda x, a: a)


def test_curvefit_accepte_une_fonction_non_vectorisee():
    x = np.linspace(0, 5, 20)
    y = 2*np.exp(-x/1.5)
    pfit, err, chi2 = curvefit(lambda x, a, tau: a*math.exp(-x/tau), x, y, p0=[1, 1], verbose=False)
    assert abs(pfit[0] - 2) < 1e-6 and abs(pfit[1] - 1.5) < 1e-6
    # et une dérivée non vectorisée, avec des incertitudes sur x
    pfit, err, chi2 = curvefit(lambda x, a, tau: a*math.exp(-x/tau), x, y, p0=[1, 1], datayerrors=0.01,
                               dataxerrors=0.01, function_derivate=lambda x, a, tau: -a/tau*math.exp(-x/tau),
                               verbose=False)
    assert abs(pfit[0] - 2) < 1e-6 and abs(pfit[1] - 1.5) < 1e-6


def test_formater_arrondit_sur_l_incertitude():
    assert formater(1993.489, 1.875, "Hz") == "1993.5 ± 1.9 Hz"
    assert formater(-5.0823, 0.0631) == "-5.082 ± 0.063"
    assert formater(6.609, 0.156) == "6.61 ± 0.16"
    assert resume_parametres(("a",), [2.0], [0.1]) == "a = 2.00 ± 0.10"


def test_formater_au_dela_de_cent_et_sans_incertitude():
    assert formater(1234567.0, 5432.0, "Hz") == "1234600 ± 5400 Hz"
    assert formater(12345.6, 234.0) == "12350 ± 230"
    assert formater(2.5, None, "V") == "2.5 V"
    assert resume_parametres(("a", "b"), [2.5, 1.0]) == "a = 2.5\nb = 1"


def test_ecarts_types_et_resume_depuis_err_ou_pcov():
    assert np.array_equal(ecarts_types([[4.0, 1.0], [1.0, 9.0]]), [2.0, 3.0])
    attendu = "a = 2.00 ± 0.10 V\nb = -1.00 ± 0.20"
    assert resume_parametres(("a", "b"), [2.0, -1.0], [0.1, 0.2], unites=("V", "")) == attendu
    assert resume_parametres(("a", "b"), [2.0, -1.0], np.diag([0.01, 0.04]), unites=("V", "")) == attendu
