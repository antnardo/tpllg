import numpy as np
import pytest

from tpllg.ajustement import formater
from tpllg.bode import phase_continue, tracer_bode

F = np.geomspace(100, 20000, 60)


def passe_bande(f, H0, f0, Q):
    return H0 / (1 + 1j * Q * (f / f0 - f0 / f))


class TestPhaseContinue:
    @pytest.mark.parametrize(
        ("H", "debut", "fin"),
        [
            (passe_bande(F, 5, 2000, 6), 90, -90),  # repliée sur [0, 360[, elle sautait de 55° à 280°
            (passe_bande(F, -5, 2000, 6), -90, -270),
            (1 / (1 + 1j * F / 2000), 0, -90),
            (1 / (1 - (F / 2000) ** 2 + 1j * F / 2000 / 0.7), 0, -180),
        ],
    )
    def test_sans_saut(self, H, debut, fin):
        phase = phase_continue(F, np.angle(H))
        assert np.abs(np.diff(phase)).max() < 100  # 56° au plus près de la résonance, 360° pour un saut
        assert phase[0] == pytest.approx(debut, abs=10) and phase[-1] == pytest.approx(fin, abs=10)

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
        assert np.allclose(phase_tracee, np.degrees(np.unwrap(np.angle(H))))  # l'inverseur : de -90° à -270°
        assert phase_tracee[0] == pytest.approx(-90, abs=1)

    def test_mesures_au_tour_du_modele(self):
        f = np.geomspace(100, 10000, 15)
        H = passe_bande(f, 5, 2000, 6)
        phase_mesuree = np.angle(H) + 2 * np.pi * (np.arange(15) % 3 - 1)  # repliées n'importe comment
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
