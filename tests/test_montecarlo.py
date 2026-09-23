import numpy as np

from tpllg.montecarlo import Point


def test_operations_entre_points():
    np.random.seed(1)
    a, b = Point(1.0, 0.1), Point(2.0, 0.3)
    assert abs((a + b).u - np.hypot(0.1, 0.3)) < 0.01
    assert abs((a - b).val + 1) < 0.01 and abs((a - b).u - np.hypot(0.1, 0.3)) < 0.01
    assert abs((3 - a).val - 2) < 0.01 and abs((a - 1).val) < 0.01
    assert abs((a*b).u - 2*np.hypot(0.1, 0.15)) < 0.01
    assert abs(a.apply_func(np.exp).u - np.e*0.1) < 0.01
    try:
        a + "x"
    except TypeError:
        pass
    else:
        raise AssertionError("un Point plus une chaîne doit lever TypeError")


def test_serie_lineaire_retrouve_polyfit():
    from tpllg.montecarlo import SerieLineaire, ajuster_modele
    np.random.seed(2)
    x = np.linspace(0, 10, 10)
    y = 2*x + 1 + np.random.normal(0, 0.5, 10)
    serie = SerieLineaire(x, 0.2, y, 0.5, N=20000)
    a, b = serie.ajuste()
    a_ref, b_ref = np.polyfit(x, y, 1)
    assert abs(a.val - a_ref) < 0.02 and abs(b.val - b_ref) < 0.1
    assert 0.03 < a.u < 0.1
    assert serie.x_tirages.shape == (20000, 10)
    pa, pb = ajuster_modele(lambda x, a, b: a*x + b, x, 0.2, y, 0.5, p0=[1, 0], N=300)
    assert abs(pa.val - a.val) < 0.02 and abs(pa.u - a.u) < 0.02


def test_les_correlations_sont_gardees():
    np.random.seed(3)
    a = Point(1.0, 0.1)
    assert abs((a + 1 - a).u) < 1e-9          # la même grandeur des deux côtés
    assert abs((a*a).u - (a**2).u) < 1e-9
    assert abs(a.quantiles()[1] - a.quantiles()[0] - 2*0.1) < 0.01
