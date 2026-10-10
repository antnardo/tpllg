import numpy as np
import pytest
from scipy import stats

from tpllg.incertitudes import incertitudes, loi_normale, loi_normale_cumulee, student_coef

MESURES = [9.78, 9.81, 9.85, 9.79, 9.83]


def integrale(y, x):
    return (y.sum() - (y[0] + y[-1]) / 2) * (x[1] - x[0])


class TestIncertitudes:
    def test_estimateurs(self):
        m, delta, s = incertitudes(MESURES)
        assert m == pytest.approx(9.812) and s == pytest.approx(np.std(MESURES, ddof=1))
        assert delta == pytest.approx(s / np.sqrt(5))

    def test_sigma_sans_student_multiplie_l_incertitude(self):
        """sigma était ignoré sans advanced : incertitudes(L, sigma=2) rendait l'incertitude à 68 %."""
        _, delta, s = incertitudes(MESURES, sigma=2)
        assert delta == pytest.approx(2 * s / np.sqrt(5))

    def test_avec_student(self):
        _, delta, s = incertitudes(MESURES, sigma=2, advanced=True)
        assert delta == pytest.approx(student_coef(2, 5) * s / np.sqrt(5))

    def test_une_mesure_ne_suffit_pas(self):
        with pytest.raises(ValueError, match="deux mesures"):
            incertitudes([9.81])


class TestStudent:
    def test_tend_vers_la_loi_normale(self):
        assert student_coef(2, 1000) == pytest.approx(2, abs=0.01)
        assert student_coef(2, 5) > 2.5
        assert loi_normale_cumulee(1) == pytest.approx(0.6827, abs=1e-3)

    @pytest.mark.parametrize(("sigma", "n"), [(1, 5), (2, 5), (2, 30)])
    def test_est_le_quantile_bilateral(self, sigma, n):
        t = student_coef(sigma, n)  # P(|T| < t) = P(|Z| < sigma), T de Student à n - 1
        assert stats.t(n - 1).cdf(t) - stats.t(n - 1).cdf(-t) == pytest.approx(
            loi_normale_cumulee(sigma), abs=1e-9
        )

    def test_sur_un_tableau_de_n(self):
        assert np.allclose(student_coef(1, np.array([5, 30])), [student_coef(1, 5), student_coef(1, 30)])


@pytest.mark.parametrize(("m", "s"), [(0, 1), (2.0, 3.0), (-1.0, 0.5)])
def test_loi_normale_est_normalisee_quel_que_soit_l_ecart_type(m, s):
    x = np.linspace(-40, 40, 400001)
    y = loi_normale(x, m, s)
    assert integrale(y, x) == pytest.approx(1, abs=1e-6)
    assert y.max() == pytest.approx(1 / (s * np.sqrt(2 * np.pi)), abs=1e-6)
    assert x[np.argmax(y)] == pytest.approx(m, abs=1e-3)


def test_loi_normale_cumulee_est_l_integrale_de_la_densite():
    t = np.linspace(-1.5, 1.5, 300001)
    assert integrale(loi_normale(t), t) == pytest.approx(loi_normale_cumulee(1.5), abs=1e-6)


class TestStudentArguments:
    """student_coef(7, 0.95) rendait nan : les arguments inversés passaient en silence."""

    @pytest.mark.parametrize("n", [0.95, 1, 0, 2.5, -3])
    def test_n_doit_etre_un_entier_au_moins_egal_a_2(self, n):
        with pytest.raises(ValueError, match="nombre de mesures, un entier au moins égal à 2"):
            student_coef(1, n)

    def test_sigma_strictement_positif(self):
        with pytest.raises(ValueError, match="strictement positif"):
            student_coef(0, 5)

    def test_deux_mesures_et_un_tableau_passent(self):
        assert student_coef(1, 2) == pytest.approx(1.837, abs=1e-3)
        assert student_coef(1, np.array([2, 1e7]))[1] == pytest.approx(1, abs=1e-3)
