import warnings

import numpy as np
import pytest

from tpllg.harmoniques import (
    Signal,
    coupe_bande,
    passe_bande,
    passe_bas_1,
    passe_bas_2,
    passe_haut_1,
    passe_haut_2,
    spectre_carre,
    spectre_dent_de_scie,
    spectre_triangle,
)

T = np.linspace(0, 2, 4001)


class TestSynthese:
    def test_creneau(self):
        carre = Signal(f0=1, spectre=spectre_carre, nmax=2001)
        loin_des_fronts = abs(np.cos(2 * np.pi * T)) > 0.05
        assert np.allclose(
            carre(T)[loin_des_fronts], np.sign(np.cos(2 * np.pi * T))[loin_des_fronts], atol=0.01
        )
        assert carre.amplitudes[1] == 0 and carre.amplitudes[2] == pytest.approx(4 / (3 * np.pi))

    def test_triangle(self):
        triangle = Signal(f0=1, spectre=spectre_triangle, nmax=999)
        attendu = 1 - 4 * abs(((T + 0.5) % 1) - 0.5)  # 1 en t = 0, -1 en t = 1/2
        assert np.allclose(triangle(T), attendu, atol=1e-3)

    def test_dent_de_scie(self):
        scie = Signal(f0=1, spectre=spectre_dent_de_scie, nmax=4000)
        attendu = 2 * (((T + 0.5) % 1) - 0.5)  # de -1 à 1, nulle en t = 0
        loin_du_saut = abs(((T + 0.5) % 1) - 0.5) < 0.45
        assert np.allclose(scie(T)[loin_du_saut], attendu[loin_du_saut], atol=0.01)

    def test_moyenne_et_amplitude(self):
        carre = Signal(f0=170, spectre=spectre_carre, moyenne=2.5, amplitude=2.5, nmax=100)
        assert carre.amplitudes_0[0] == 2.5 and carre.frequences_0[2] == 340
        assert carre(0.0) == pytest.approx(5.0, abs=0.05) and np.isscalar(carre(0.0))

    def test_amplitudes_positives(self):
        s = Signal(f0=1, spectre=lambda n: (-np.ones(n.shape), np.zeros(n.shape)), nmax=3)
        assert np.all(s.amplitudes == 1) and np.allclose(abs(s.phases), np.pi)
        assert s(0.0) == pytest.approx(-3)

    def test_spectre_en_entiers(self):
        """amplitudes *= amplitude levait UFuncTypeError pour un spectre en entiers."""
        s = Signal(f0=1, spectre=lambda n: (np.ones_like(n), np.zeros_like(n)), amplitude=0.5, nmax=4)
        assert s(0.0) == pytest.approx(2.0)

    def test_depuis_coefficients_ne_modifie_pas_les_listes(self):
        """spectre_to_func mettait phases[0] à zéro dans la liste de l'appelant."""
        amplitudes = [1.59, 2.5, 1.06, 0, 0.212]
        phases = [1.0, np.pi / 2, 0, 0, 0.3]
        s = Signal.depuis_coefficients(200, amplitudes, phases)
        assert phases[0] == 1.0
        t = np.linspace(0, 0.01, 50)
        attendu = 1.59 + sum(
            A * np.cos(2 * np.pi * 200 * n * t + p)
            for n, A, p in zip(range(1, 5), amplitudes[1:], phases[1:])
        )
        assert np.allclose(s(t), attendu)
        with pytest.raises(ValueError, match="autant de phases"):
            Signal.depuis_coefficients(200, [1, 2], [0])


class TestValeurEfficace:
    def test_composante_continue_comptee_entiere(self):
        """2 + cos : 2,12, et non 1,58 (moyenne²/2)."""
        s = Signal.depuis_coefficients(1, [2.0, 1.0])
        assert s.Veff == pytest.approx(np.sqrt(4.5))
        assert s.Veff == pytest.approx(np.sqrt(np.mean(s(np.linspace(0, 1, 10000, endpoint=False)) ** 2)))

    def test_parseval_sur_le_creneau(self):
        assert Signal(f0=1, spectre=spectre_carre, nmax=20001).Veff == pytest.approx(1, abs=1e-4)


class TestFiltrage:
    def test_sinusoide_dans_un_passe_bas(self):
        sortie = Signal.depuis_coefficients(1000, [0, 2.0]).filtre(passe_bas_1(1000))
        assert sortie.amplitudes[0] == pytest.approx(2 / np.sqrt(2)) and sortie.phases[0] == pytest.approx(
            -np.pi / 4
        )

    def test_signe_de_la_composante_continue(self):
        """moyenne × |H(0)| perdait le signe d'un inverseur."""
        sortie = Signal.depuis_coefficients(1000, [1.0, 1.0]).filtre(passe_bas_1(100, H0=-2))
        assert sortie.moyenne == pytest.approx(-2)

    def test_h_qui_ne_se_calcule_pas_en_zero(self):
        def avec_f0_sur_f(f):
            return 1 / (1 + 1j * 6 * (f / 1000 - 1000 / f))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sortie = Signal.depuis_coefficients(1000, [3.0, 1.0]).filtre(avec_f0_sur_f)
        assert sortie.moyenne == pytest.approx(0, abs=1e-6) and sortie.amplitudes[0] == pytest.approx(1)

    def test_selection_d_une_harmonique(self):
        carre = Signal(f0=170, spectre=spectre_carre, moyenne=2.5, amplitude=2.5, nmax=100)
        sortie = carre.filtre(passe_bande(3 * 170, 100, H0=5))
        assert sortie.moyenne == 0 and sortie.amplitudes[2] == pytest.approx(5 * 2.5 * 4 / (3 * np.pi))
        assert sortie.Veff == pytest.approx(sortie.amplitudes[2] / np.sqrt(2), rel=0.01)


class TestFiltres:
    @pytest.mark.parametrize(
        ("H", "attendu"),
        [
            (passe_bas_1(1000, 2), 2 / (1 + 1j)),
            (passe_haut_1(1000, 2), 2 * 1j / (1 + 1j)),
            (passe_bas_2(1000, 5, 2), -10j),
            (passe_haut_2(1000, 5, 2), 10j),
            (passe_bande(1000, 5, 2), 2),
            (coupe_bande(1000, 5, 2), 0),
        ],
    )
    def test_valeur_a_la_frequence_caracteristique(self, H, attendu):
        assert H(1000.0) == pytest.approx(attendu)

    @pytest.mark.parametrize(
        "H", [passe_bas_1(1000), passe_bas_2(1000, 0.7), passe_bande(1000, 5), passe_haut_2(1000, 1)]
    )
    def test_en_zero_sans_division_par_zero(self, H):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isfinite(H(np.array([0.0, 10.0]))).all()
