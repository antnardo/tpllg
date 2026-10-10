import matplotlib.pyplot as plt
import numpy as np
import pytest

from tpllg import montecarlo
from tpllg.ajustement import curvefit, regression_york
from tpllg.montecarlo import Point, SerieLineaire, ajuster_modele, fixer_graine, indices_sobol


@pytest.fixture(autouse=True)
def graine():
    fixer_graine(20261008)
    yield
    fixer_graine(None)


def droite(x, a, b):
    return a * x + b


class TestOperations:
    def test_entre_points(self):
        a, b = Point(1.0, 0.1), Point(2.0, 0.3)
        assert (a + b).u == pytest.approx(np.hypot(0.1, 0.3), abs=0.01)
        assert (a - b).val == pytest.approx(-1, abs=0.01) and (a - b).u == pytest.approx(
            np.hypot(0.1, 0.3), abs=0.01
        )
        assert (3 - a).val == pytest.approx(2, abs=0.01) and (a - 1).val == pytest.approx(0, abs=0.01)
        assert (a * b).u == pytest.approx(2 * np.hypot(0.1, 0.15), abs=0.01)
        assert (
            (-a).val == pytest.approx(-1, abs=0.01)
            and abs(-a).val == pytest.approx(1, abs=0.01)
            and (+a) is a
        )
        with pytest.raises(TypeError):
            a + "x"

    def test_les_fonctions_numpy_rendent_un_point(self):
        a, b = Point(1.0, 0.1), Point(2.0, 0.3)
        e = np.exp(a)
        assert isinstance(e, Point) and e.u == pytest.approx(np.e * 0.1, abs=0.01)
        assert e.val == pytest.approx(np.e * np.exp(0.1**2 / 2), abs=0.005)  # E[exp X], loi log-normale
        assert np.log(b).u == pytest.approx(0.3 / 2, abs=0.005)
        assert np.sqrt(b).u == pytest.approx(0.3 / (2 * np.sqrt(2)), abs=0.005)
        assert isinstance(np.sqrt(a * b), Point) and isinstance(np.arctan2(b, a), Point)
        assert (np.float64(3.0) - a).val == pytest.approx(2, abs=0.01) and np.add(a, 1).val == pytest.approx(
            2, abs=0.01
        )
        assert (np.exp(a) - np.exp(a)).u == 0  # le même tirage des deux côtés
        frac, ent = np.modf(b)
        assert isinstance(frac, Point) and isinstance(ent, Point)

    def test_les_correlations_sont_gardees(self):
        a = Point(1.0, 0.1)
        assert (a + 1 - a).u == pytest.approx(0, abs=1e-12)
        assert (a * a).u == pytest.approx((a**2).u)


class TestNombreDeTirages:
    def test_un_point_non_tire_prend_le_n_de_l_autre(self):
        """Le cas de l'audit : ajuster_modele(N=200) puis un produit par un Point mesuré."""
        x = np.linspace(0, 10, 10)
        pa, _ = ajuster_modele(droite, x, 0.2, 2 * x + 1, 0.5, p0=[1, 0], N=200)
        g = Point(9.81, 0.01)
        assert (pa * g).N == 200 and g.N == 200 and (g * pa).N == 200

    def test_deux_n_differents_refuses(self):
        with pytest.raises(ValueError, match="ne se combinent pas"):
            Point(1.0, 0.1, N=1000) + Point(2.0, 0.1, N=5000)

    def test_un_point_tire_n_est_jamais_retire(self):
        L = Point(1.0, 0.1, N=1000)
        c = 2 * L
        _ = L + Point(2.0, 0.1)  # l'autre s'aligne sur L, qui garde son tirage
        assert L.N == 1000 and (c - 2 * L).u == 0
        with pytest.raises(ValueError, match="ne se combinent pas"):
            L + np.exp(Point(2.0, 0.1, N=5000))

    def test_point_depuis_un_tirage(self):
        p = Point(tirage=[1.0, 2.0, 3.0, 4.0])
        assert p.val == 2.5 and p.u == pytest.approx(np.sqrt(5 / 3)) and p.N == 4 and p.tirage[0] == 1.0
        q = Point(1.0, 0.1, N=500)
        assert repr(q) == "Point(1, 0.1, N=500)" and len(q.tirage) == 500

    @pytest.mark.parametrize("arguments", [(1.0,), (1.0, -0.1)])
    def test_point_incomplet_refuse(self, arguments):
        with pytest.raises(ValueError):
            Point(*arguments)

    def test_graine_reproductible(self):
        fixer_graine(3)
        premier = Point(1.0, 0.1, N=10).tirage
        fixer_graine(3)
        assert np.array_equal(Point(1.0, 0.1, N=10).tirage, premier)


class TestLois:
    @pytest.mark.parametrize(
        ("loi", "rapport"), [("uniforme", np.sqrt(3)), ("triangulaire", np.sqrt(6)), ("arcsinus", np.sqrt(2))]
    )
    def test_incertitude_type_de_la_loi(self, loi, rapport):
        p = getattr(Point, loi)(4.87, 0.03)
        assert p.u == pytest.approx(0.03 / rapport)
        assert p.tirage.std() == pytest.approx(p.u, rel=0.01) and p.tirage.mean() == pytest.approx(
            4.87, abs=1e-3
        )
        assert p.tirage.min() >= 4.84 and p.tirage.max() <= 4.90

    def test_tolerance_nulle(self):
        assert np.all(Point.triangulaire(2.0, 0.0, N=5).tirage == 2.0)

    def test_quantiles_suivent_une_transformation_monotone(self):
        x = Point(1.0, 0.2, N=400000)
        bas, haut = (x**2).quantiles()
        assert bas == pytest.approx(0.8**2, abs=0.01) and haut == pytest.approx(1.2**2, abs=0.01)
        assert (x**2).val == pytest.approx(1.04, abs=0.005)  # E[x²] = 1 + u²
        assert np.allclose(x.quantiles(0.9545), (0.6, 1.4), atol=0.01)

    def test_intervalle_le_plus_court(self):
        y = np.exp(Point(0.0, 0.5))  # log-normale : dissymétrique
        bas, haut = y.intervalle_le_plus_court(0.95)
        q_bas, q_haut = y.quantiles(0.95)
        assert haut - bas < 0.95 * (q_haut - q_bas) and bas < q_bas
        assert np.mean((y.tirage >= bas) & (y.tirage <= haut)) == pytest.approx(0.95, abs=1e-3)
        u = Point.uniforme(0.0, 1.0)  # loi plate : tous les intervalles se valent
        bas, haut = u.intervalle_le_plus_court(0.95)
        assert haut - bas == pytest.approx(1.9, abs=0.01)


class TestSerieLineaire:
    def test_x_exacts_les_incertitudes_des_moindres_carres(self):
        x = np.linspace(0, 10, 11)
        serie = SerieLineaire(x, 0.0, 2 * x + 1, 0.3, N=200000)
        a, b = serie.ajuster()
        sxx = ((x - x.mean()) ** 2).sum()
        assert a.val == pytest.approx(2) and b.val == pytest.approx(1)
        assert a.u == pytest.approx(0.3 / np.sqrt(sxx), rel=0.02)
        assert b.u == pytest.approx(0.3 * np.sqrt(1 / x.size + x.mean() ** 2 / sxx), rel=0.02)
        assert np.array_equal(serie.x_tirages[0], x) and serie.y_tirages.shape == (200000, 11)
        with pytest.raises(ValueError):
            SerieLineaire([1, 2, 3], 0.1, [1, 2], 0.1)

    def test_pas_de_pente_attenuee_avec_u_x(self):
        """Le cas de l'audit : u_x non négligeable atténuait la pente moyenne des tirages (1,90 pour 2)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a, b = SerieLineaire(x, 0.5, 2 * x, 0.1).ajuster()
        assert a.val == pytest.approx(2) and b.val == pytest.approx(0, abs=1e-9)

    def test_la_valeur_et_l_incertitude_de_york(self, rng):
        x = np.linspace(0, 10, 10)
        x_mes = x + rng.normal(0, 0.3, 10)
        y_mes = 2 * x + 1 + rng.normal(0, 0.5, 10)
        a, b = SerieLineaire(x_mes, 0.3, y_mes, 0.5).ajuster()
        york = regression_york(x_mes, 0.3, y_mes, 0.5)
        assert np.allclose([a.val, b.val], york.pfit) and np.allclose([a.u, b.u], york.err, rtol=0.05)


class TestAjusterModele:
    def test_valeur_de_l_ajustement_et_dispersion_des_tirages(self, rng):
        t = np.linspace(0, 5, 12)
        u = 2 * np.exp(-t / 1.5) + rng.normal(0, 0.02, t.size)

        def exponentielle(t, A, tau):
            return A * np.exp(-t / tau)

        pA, ptau = ajuster_modele(exponentielle, t, 0.01, u, 0.02, p0=[1, 1], N=400)
        reference = curvefit(exponentielle, t, u, [1, 1], 0.02, 0.01, verbose=False)
        assert np.allclose([pA.val, ptau.val], reference.pfit)
        assert np.allclose([pA.u, ptau.u], reference.err, rtol=0.15)

    def test_un_tirage_qui_ne_converge_pas_est_remplace(self, monkeypatch, capsys):
        vrai = montecarlo.curve_fit
        appels = []

        def capricieux(*args, **kwargs):
            appels.append(1)
            if len(appels) == 1:
                raise RuntimeError("Optimal parameters not found")
            return vrai(*args, **kwargs)

        monkeypatch.setattr(montecarlo, "curve_fit", capricieux)
        x = np.linspace(0, 10, 10)
        pa, _ = ajuster_modele(droite, x, 0.0, 2 * x + 1, 0.5, p0=[1, 0], N=20)
        assert pa.N == 20 and "1 tirage(s) sans convergence" in capsys.readouterr().out

    def test_un_modele_qui_ne_converge_jamais(self, monkeypatch):
        def jamais(*args, **kwargs):
            raise RuntimeError("Optimal parameters not found")

        monkeypatch.setattr(montecarlo, "curve_fit", jamais)
        x = np.linspace(0, 10, 10)
        with pytest.raises(RuntimeError, match="dix tirages"):
            ajuster_modele(droite, x, 0.0, 2 * x + 1, 0.5, p0=[1, 0], N=5)


class TestSobol:
    def test_produit_de_deux_lois_uniformes(self):
        """L'exemple de mc_sobol.py : y = x1 x2, x1 = 3,1 ± 0,05 et x2 = 6,8 ± 0,05 (lois plates)."""
        x1, x2 = Point.uniforme(3.1, 0.05), Point.uniforme(6.8, 0.05)
        s1, s2 = indices_sobol(lambda a, b: a * b, [x1, x2], N=200000)
        attendu = 6.8**2 / (6.8**2 + 3.1**2)  # Var y ≈ x2² u1² + x1² u2², avec u1 = u2
        assert s1 == pytest.approx(attendu, abs=0.01) and s2 == pytest.approx(1 - attendu, abs=0.01)

    def test_modele_additif(self):
        x1, x2 = Point(0.0, 1.0), Point(0.0, 0.5)
        s = indices_sobol(lambda a, b: a + 2 * b, [x1, x2], N=200000)
        assert np.allclose(s, [0.5, 0.5], atol=0.01)

    def test_un_resultat_de_calcul_refuse(self):
        x = Point(1.0, 0.1)
        with pytest.raises(ValueError, match="pas de loi"):
            indices_sobol(lambda a: a, [x * 2])


class TestShow:
    def test_trace_une_densite(self):
        _, ax, n, bins, _ = Point(3.0, 0.5, N=50000).show()
        assert (n * np.diff(bins)).sum() == pytest.approx(1, abs=1e-3)  # une densité, sur ± 5 écarts-types
        assert len(ax.get_lines()) == 4  # moyenne, ± écart-type, loi normale

    def test_dans_un_repere_donne_rend_sa_figure(self):
        f1, ax1 = plt.subplots()
        plt.subplots()  # une autre figure devient la courante
        fig, ax, *_ = Point(1.0, 0.1, N=1000).show(ax=ax1)
        assert fig is f1 and ax is ax1


class TestPointNormale:
    def test_normale_est_le_constructeur(self):
        a, b = Point.normale(2.0, 0.1, N=5000), Point(2.0, 0.1, N=5000)
        assert (a.val, a.u, a.N, a._loi) == (b.val, b.u, b.N, b._loi)
        assert a.tirage.std() == pytest.approx(0.1, rel=0.1)
