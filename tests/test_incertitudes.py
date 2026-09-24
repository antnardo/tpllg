import numpy as np
from scipy import stats

from tpllg.incertitudes import incertitudes, loi_normale, loi_normale_cumulee, student_coef


def integrale(y, x):
    return (y.sum() - (y[0] + y[-1])/2)*(x[1] - x[0])


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


def test_student_coef_est_le_quantile_bilateral():
    for sigma, n in ((1, 5), (2, 5), (2, 30)):
        t = student_coef(sigma, n)                       # P(|T| < t) = P(|Z| < sigma), T de Student à n - 1
        assert abs(stats.t(n - 1).cdf(t) - stats.t(n - 1).cdf(-t) - loi_normale_cumulee(sigma)) < 1e-12


def test_loi_normale_est_normalisee_quel_que_soit_l_ecart_type():
    x = np.linspace(-40, 40, 400001)
    for m, s in ((0, 1), (2.0, 3.0), (-1.0, 0.5)):
        y = loi_normale(x, m, s)
        assert abs(integrale(y, x) - 1) < 1e-6
        assert abs(y.max() - 1/(s*np.sqrt(2*np.pi))) < 1e-6
        assert abs(x[np.argmax(y)] - m) < 1e-3
    t = np.linspace(-1.5, 1.5, 300001)
    assert abs(integrale(loi_normale(t), t) - loi_normale_cumulee(1.5)) < 1e-6
