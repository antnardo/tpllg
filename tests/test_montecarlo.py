import matplotlib.pyplot as plt
import numpy as np
import pytest

from tpllg.montecarlo import Point, SerieLineaire, ajuster_modele


def test_operations_entre_points():
    np.random.seed(1)
    a, b = Point(1.0, 0.1), Point(2.0, 0.3)
    assert abs((a + b).u - np.hypot(0.1, 0.3)) < 0.01
    assert abs((a - b).val + 1) < 0.01 and abs((a - b).u - np.hypot(0.1, 0.3)) < 0.01
    assert abs((3 - a).val - 2) < 0.01 and abs((a - 1).val) < 0.01
    assert abs((a*b).u - 2*np.hypot(0.1, 0.15)) < 0.01
    assert abs(a.apply_func(np.exp).u - np.e*0.1) < 0.01
    assert abs((-a).val + 1) < 0.01 and abs(abs(-a).val - 1) < 0.01 and (+a) is a
    assert (Point(1.0, 0.1, N=1000) + Point(2.0, 0.1, N=5000)).N == 5000
    with pytest.raises(TypeError):
        a + "x"


def test_point_depuis_un_tirage():
    p = Point(tirage=[1.0, 2.0, 3.0, 4.0])
    assert p.val == 2.5 and abs(p.u - np.sqrt(5/3)) < 1e-12 and p.N == 4 and p.i == 1.0
    q = Point(1.0, 0.1, N=500)
    assert repr(q) == "Point(1, 0.1, N=500)" and len(q.tirage) == 500
    q.calc_tirage(2000)
    assert q.N == 2000 and len(q.tirage) == 2000
    with pytest.raises(ValueError):
        Point(1.0)


def test_serie_lineaire_retrouve_polyfit():
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


def test_serie_lineaire_retrouve_les_incertitudes_des_moindres_carres():
    np.random.seed(6)
    x = np.linspace(0, 10, 11)
    serie = SerieLineaire(x, 0.0, 2*x + 1, 0.3, N=200000)   # x exacts, y à 0,3 près
    a, b = serie.ajuste()
    sxx = ((x - x.mean())**2).sum()
    assert abs(a.val - 2) < 1e-3 and abs(b.val - 1) < 5e-3
    assert abs(a.u/(0.3/np.sqrt(sxx)) - 1) < 0.02                      # sigma_a = sigma/sqrt(Sxx)
    assert abs(b.u/(0.3*np.sqrt(1/x.size + x.mean()**2/sxx)) - 1) < 0.02
    assert np.array_equal(serie.xi, x) and serie.yi.shape == (11,) and serie.Nt == 200000
    assert np.allclose(SerieLineaire.coefs(x, 2*x + 1), (2, 1))
    a2, b2 = SerieLineaire.coefs(np.vstack([x, x]), np.vstack([2*x + 1, -x + 3]))
    assert np.allclose(a2, [2, -1]) and np.allclose(b2, [1, 3])
    with pytest.raises(ValueError):
        SerieLineaire([1, 2, 3], 0.1, [1, 2], 0.1)


def test_les_correlations_sont_gardees():
    np.random.seed(3)
    a = Point(1.0, 0.1)
    assert abs((a + 1 - a).u) < 1e-9          # la même grandeur des deux côtés
    assert abs((a*a).u - (a**2).u) < 1e-9
    assert abs(a.quantiles()[1] - a.quantiles()[0] - 2*0.1) < 0.01


def test_quantiles_suivent_une_transformation_monotone():
    np.random.seed(4)
    x = Point(1.0, 0.2, N=400000)
    bas, haut = (x**2).quantiles()
    assert abs(bas - 0.8**2) < 0.01 and abs(haut - 1.2**2) < 0.01   # (1 -+ 0,2)² : les quantiles suivent x -> x²
    assert abs((x**2).val - 1.04) < 0.005                             # E[x²] = 1 + u²
    b2, h2 = x.quantiles(0.9545)
    assert abs(b2 - 0.6) < 0.01 and abs(h2 - 1.4) < 0.01


def test_show_trace_une_densite():
    np.random.seed(5)
    fig, ax, n, bins, patches = Point(3.0, 0.5, N=50000).show()
    assert abs((n*np.diff(bins)).sum() - 1) < 1e-3         # l'histogramme est une densité, sur ± 5 écarts-types
    assert len(ax.get_lines()) == 4                         # moyenne, ± écart-type, loi normale
    plt.close(fig)
