import numpy as np
import pytest

from tpllg.ajustement import formater
from tpllg.bode import phase_0_360, phase_continue, phase_repliee, tracer_bode

F = np.geomspace(100, 20000, 60)


def passe_bande(f, H0, f0, Q):
    return H0 / (1 + 1j * Q * (f / f0 - f0 / f))


class TestPhaseContinue:
    @pytest.mark.parametrize(
        ("H", "debut", "fin"),
        [
            (passe_bande(F, 5, 2000, 6), 90, -90),  # repliée sur [0, 360[, elle sautait de 55° à 280°
            (passe_bande(F, -5, 2000, 6), 270, 90),  # comme le TP le montre : 270° → 180° → 90°
            (1 / (1 + 1j * F / 2000), 0, -90),
            (1 / (1 - (F / 2000) ** 2 + 1j * F / 2000 / 0.7), 0, -180),
        ],
    )
    def test_sans_saut(self, H, debut, fin):
        phase = phase_continue(F, np.angle(H))
        assert np.abs(np.diff(phase)).max() < 100  # 56° au plus près de la résonance, 360° pour un saut
        assert phase[0] == pytest.approx(debut, abs=10) and phase[-1] == pytest.approx(fin, abs=10)

    @pytest.mark.parametrize("bruit", [-3.0, 0.0, 3.0])
    def test_inverseur_mesure_sans_modele(self, bruit):
        """Une première mesure à -90° ± le bruit ne fait pas basculer la courbe d'un tour."""
        phase = phase_continue([200, 2000, 20000], np.radians([-90 + bruit, 180, 90]))
        assert np.allclose(phase, [270 + bruit, 180, 90])

    def test_mesures_voisines_de_zero(self):
        """± 0,5° autour de 0 se dispersaient entre 0 et 359,5."""
        phase = phase_continue([100, 200, 300, 400], np.radians([-0.5, 0.3, -1.0, 0.4]))
        assert np.allclose(phase, [-0.5, 0.3, -1.0, 0.4])

    def test_au_tour_le_plus_proche_de_la_reference(self):
        phase = phase_continue([1, 2, 3], np.radians([10, -170, 175]), reference=[370, 190, 180])
        assert np.allclose(phase, [370, 190, 175])

    def test_dans_le_desordre_des_frequences(self):
        ordre = np.random.default_rng(1).permutation(F.size)
        phase = np.angle(passe_bande(F, -5, 2000, 6))
        assert np.allclose(phase_continue(F[ordre], phase[ordre]), phase_continue(F, phase)[ordre])


class TestTracerBode:
    def test_ecrit_la_figure_et_la_legende(self, tmp_path):
        f = np.geomspace(100, 10000, 15)
        H = passe_bande(f, -5, 2000, 6)
        pfit, err = [-5.0, 2000.0, 6.0], [0.1, 2.0, 0.2]
        fichier = tmp_path / "bode.png"
        fig = tracer_bode(
            f, np.abs(H), np.angle(H), passe_bande, pfit, err, ("H0", "f0", "Q"), ("", "Hz", ""), str(fichier)
        )
        ax1, ax2 = fig.axes
        assert fichier.exists() and fichier.stat().st_size > 0
        assert ax1.get_xscale() == "log" and ax1.get_yscale() == "log"
        textes = [t.get_text() for t in ax1.get_legend().get_texts()]
        assert any("f0 = " + formater(2000.0, 2.0, "Hz") in t for t in textes)
        x_ajuste, y_ajuste = ax1.get_lines()[1].get_data()
        assert np.allclose(y_ajuste, np.abs(passe_bande(x_ajuste, *pfit)))
        _, phase_tracee = ax2.get_lines()[0].get_data()
        assert np.allclose(
            phase_tracee, np.degrees(np.unwrap(np.angle(H))) + 360
        )  # l'inverseur : de 270° à 90°
        assert phase_tracee[0] == pytest.approx(270, abs=1)

    def test_mesures_au_tour_du_modele(self):
        f = np.geomspace(100, 10000, 15)
        H = passe_bande(f, 5, 2000, 6)
        # repliées n'importe comment, un tour sur deux à l'autre tour, sous 2 pi en valeur absolue
        phase_mesuree = np.angle(H) + np.where(np.angle(H) > 0, -2 * np.pi, 2 * np.pi) * (np.arange(15) % 2)
        fig = tracer_bode(f, np.abs(H), phase_mesuree, passe_bande, [5, 2000, 6])
        _, phase_tracee = fig.axes[1].get_lines()[0].get_data()
        _, phase_modele = fig.axes[1].get_lines()[1].get_data()
        assert np.allclose(phase_tracee, np.degrees(np.angle(H)))
        assert np.abs(np.diff(phase_modele)).max() < 5

    def test_covariance_au_lieu_des_incertitudes(self):
        f = np.geomspace(100, 10000, 15)
        H = passe_bande(f, -5, 2000, 6)
        fig = tracer_bode(
            f,
            np.abs(H),
            np.angle(H),
            passe_bande,
            [-5.0, 2000.0, 6.0],
            np.diag([0.01, 4.0, 0.04]),
            ("H0", "f0", "Q"),
        )
        assert any(
            "f0 = " + formater(2000.0, 2.0) in t.get_text() for t in fig.axes[0].get_legend().get_texts()
        )

    def test_sans_modele(self):
        f = np.geomspace(100, 10000, 15)
        H = passe_bande(f, -5, 2000, 6)
        fig = tracer_bode(f, np.abs(H), np.angle(H), gain_log=False)
        assert fig.axes[0].get_yscale() == "linear" and [len(ax.get_lines()) for ax in fig.axes] == [1, 1]


class TestPhaseRepliee:
    def test_inverse_de_phase_continue(self):
        phase = np.angle(passe_bande(F, -5, 2000, 6))
        continue_ = phase_continue(F, phase)
        assert continue_.max() > 180  # déroulée : de 270° à 90°
        assert np.allclose(phase_repliee(continue_), phase)

    def test_dans_moins_pi_pi(self):
        attendu = [0, np.pi, np.radians(-170), np.pi, np.radians(-1)]
        assert np.allclose(phase_repliee([0, 180, -170, 540, 359.0]), attendu)
        assert phase_repliee(90.0) == pytest.approx(np.pi / 2)


class TestPhasesEnRadians:
    def test_tracer_bode_refuse_des_degres(self):
        H = passe_bande(F, -5, 2000, 6)
        with pytest.raises(ValueError, match="les phases doivent être en radians"):
            tracer_bode(F, np.abs(H), np.degrees(np.angle(H)))


class TestAnciensNoms:
    def test_phase_0_360_comme_avant(self):
        with pytest.warns(DeprecationWarning, match="phase_continue"):
            degres = phase_0_360(np.radians([-95, 176, 10, 370]))
        assert np.allclose(degres, [265, 176, 10, 10])
