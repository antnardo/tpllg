import numpy as np

from tpllg.incertitudes import incertitudes, loi_normale_cumulee, student_coef


def test_incertitudes_estimateurs():
    L = [9.78, 9.81, 9.85, 9.79, 9.83]
    m, delta, s = incertitudes(L)
    assert abs(m - 9.812) < 1e-9
    assert abs(s - np.std(L, ddof=1)) < 1e-12
    assert abs(delta - s/np.sqrt(5)) < 1e-12
    m2, delta2, s2 = incertitudes(L, sigma=2, advanced=True)
    assert abs(delta2 - student_coef(2, 5)*s/np.sqrt(5)) < 1e-12


def test_student_tend_vers_la_loi_normale():
    assert abs(student_coef(2, 1000) - 2) < 0.01
    assert student_coef(2, 5) > 2.5
    assert abs(loi_normale_cumulee(1) - 0.6827) < 1e-3
