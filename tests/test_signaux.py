import time

import numpy as np
import pytest

from tpllg.signaux import (
    decrement_logarithmique,
    extremums,
    fenetre,
    frequence_pic,
    front_utile,
    fronts_descendants,
    fronts_montants,
    taux_amortissement,
)


def creneau(rng, fe=100000.0, n=3000, f=200.0, phase=1.0):
    t = np.arange(n) / fe
    return t, np.sign(np.sin(2 * np.pi * f * t + phase)) + rng.normal(0, 0.01, t.size)


def oscillation_amortie(fe=200000.0, f0=2000.0, Q=6.0, n=2000, offset=0.0):
    t = np.arange(n) / fe
    return t, offset - 1.5 * np.exp(-np.pi * f0 * t / Q) * np.sin(2 * np.pi * f0 * t)


class TestFronts:
    def test_fronts_montants_d_un_creneau(self, rng):
        t, v = creneau(rng)
        fronts, (bas, haut) = fronts_montants(t, v)
        assert bas == pytest.approx(-1, abs=0.05) and haut == pytest.approx(1, abs=0.05)
        attendus = [a for a in ((k - 1 / (2 * np.pi)) / 200 for k in range(8)) if 0 < a < t[-1]]
        assert np.allclose(fronts, attendus, atol=2e-5)
        t0, periode = front_utile(t, fronts)
        assert periode == pytest.approx(1 / 200, abs=2e-5) and t0 == fronts[0]

    def test_fronts_descendants_d_un_creneau(self, rng):
        t, v = creneau(rng)
        fronts, (bas, haut) = fronts_descendants(t, v)
        assert bas == pytest.approx(-1, abs=0.05) and haut == pytest.approx(1, abs=0.05)
        attendus = [a for a in ((k + 0.5 - 1 / (2 * np.pi)) / 200 for k in range(8)) if 0 < a < t[-1]]
        assert np.allclose(fronts, attendus, atol=2e-5)

    @pytest.mark.parametrize("rapport_cyclique", [0.02, 0.5, 0.98])
    def test_impulsions_breves(self, rng, rapport_cyclique):
        """Les percentiles 5 et 95 ne voyaient rien sous 5 % de rapport cyclique."""
        t = np.arange(100000) / 1e6
        v = 5.0 * ((t * 1000) % 1 < rapport_cyclique) + rng.normal(0, 0.02, t.size)
        fronts, (bas, haut) = fronts_montants(t, v)
        assert bas == pytest.approx(0, abs=0.01) and haut == pytest.approx(5, abs=0.01)
        assert len(fronts) == 99 and np.allclose(np.diff(fronts), 1e-3, atol=2e-6)

    def test_rapide_sur_une_acquisition_pleine(self, rng):
        t = np.arange(262144) / 1e6
        v = np.sign(np.sin(2 * np.pi * 1000 * t + 1)) + rng.normal(0, 0.01, t.size)
        debut = time.perf_counter()
        fronts, _ = fronts_montants(t, v)
        assert time.perf_counter() - debut < 0.5 and len(fronts) == 262

    def test_signal_plat_sans_front(self):
        t = np.arange(100) * 1e-3
        fronts, _ = fronts_montants(t, np.ones(100))
        assert fronts.size == 0

    def test_deux_voies_d_un_coup_refusees(self):
        with pytest.raises(ValueError, match="une seule voie"):
            fronts_montants(np.zeros((2, 10)), np.zeros((2, 10)))


def test_front_utile():
    t = np.linspace(0, 1, 1001)
    t0, periode = front_utile(t, [0.12, 0.32, 0.52, 0.73])
    assert t0 == 0.12 and periode == pytest.approx(0.2)  # la médiane des écarts : 0,2, 0,2, 0,21
    t0, periode = front_utile(t, [0.4])
    assert t0 == 0.4 and periode == pytest.approx(0.6)  # un seul front : jusqu'à la fin
    with pytest.raises(ValueError):
        front_utile(t, [])


def test_fenetre_est_un_intervalle_semi_ouvert():
    t = np.arange(10.0)
    tf, vf = fenetre(t, t**2, 2, 3)
    assert np.array_equal(tf, [2, 3, 4]) and np.array_equal(vf, [4, 9, 16])


def test_frequence_pic_exacte_sur_un_nombre_entier_de_periodes():
    t = np.arange(1000) / 10000.0
    assert frequence_pic(t, 3 + np.cos(2 * np.pi * 370 * t)) == pytest.approx(370)  # la moyenne ne compte pas


class TestExtremums:
    def test_frequence_et_amortissement_d_une_oscillation(self):
        t, v = oscillation_amortie()
        assert frequence_pic(t, v) == pytest.approx(2000, abs=150)
        pics = extremums(t, v, 2000)
        alpha = taux_amortissement(t[pics], v[pics])
        assert alpha == pytest.approx(np.pi * 2000 / 6, rel=0.05)

    def test_autour_d_un_offset(self):
        t, v = oscillation_amortie(offset=2.0)
        pics = extremums(t, v, 2000, offset=2.0)
        assert np.all(np.diff(t[pics]) > 0.4 / 2000)
        assert np.median(np.diff(t[pics])) == pytest.approx(0.5 / 2000, abs=1e-5)  # une demi-période
        assert taux_amortissement(t[pics], v[pics], offset=2.0) == pytest.approx(np.pi * 2000 / 6, rel=0.05)

    def test_seuil_ecarte_les_bosses_du_bruit(self, rng):
        t, v = oscillation_amortie(n=4000)
        v = v + rng.normal(0, 0.003, v.size)
        tous = extremums(t, v, 2000)
        francs = extremums(t, v, 2000, seuil=0.03)  # dix fois le bruit
        assert len(francs) < len(tous) and np.all(abs(v[francs]) > 0.03)
        assert taux_amortissement(t[francs], v[francs]) == pytest.approx(np.pi * 2000 / 6, rel=0.05)

    def test_te_deduit_de_t(self):
        """extremums(v, fe, f) et frequence_pic(v, te) ne prenaient pas la même chose : 509 extremums
        au lieu de 41 en confondant les deux. Les deux prennent maintenant (t, v)."""
        t, v = oscillation_amortie()  # 10 ms à 2 kHz : vingt périodes, quarante extremums
        assert len(extremums(t, v, 2000)) == 40

    def test_taux_amortissement_veut_deux_extremums(self):
        with pytest.raises(ValueError, match="deux extremums"):
            taux_amortissement([0.1], [1.0])


class TestAnciennesFormes:
    """Les formes de 2026.9, reconnues au second argument, un nombre, sans avertissement."""

    def test_frequence_pic_v_te(self):
        t, v = oscillation_amortie()
        assert frequence_pic(v, t[1] - t[0]) == frequence_pic(t, v)

    def test_extremums_v_fe_f_offset(self):
        t, v = oscillation_amortie(offset=0.3)
        anciens = extremums(v, 1 / (t[1] - t[0]), 2000.0, 0.3)
        assert np.array_equal(anciens, extremums(t, v, 2000.0, 0.3))

    def test_decrement_logarithmique(self):
        t, v = oscillation_amortie()
        pics = extremums(t, v, 2000.0)
        alpha = decrement_logarithmique(t[pics], v[pics])
        assert alpha == taux_amortissement(t[pics], v[pics]) == pytest.approx(np.pi * 2000 / 6, rel=1e-3)
