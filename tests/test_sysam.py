import numpy as np
import pytest

from tpllg import sysam_factice
from tpllg.sysam import Sysam


@pytest.fixture
def silence(monkeypatch):
    monkeypatch.setattr(sysam_factice, "VERBOSE", False)


def test_acquerir_rend_la_forme_de_pycanum(silence):
    """Deux tableaux 2D de double, une ligne par voie, le temps en secondes."""
    with Sysam([0, 1], 5) as can:
        can.config_echantillon(1e-5, 200)
        temps, tensions = can.acquerir()
    assert temps.shape == (2, 200) and tensions.shape == (2, 200)
    assert temps.dtype == np.float64 and tensions.dtype == np.float64
    assert temps[0][1] == pytest.approx(1e-5) and temps[1][-1] == pytest.approx(199e-5)
    assert abs(tensions).max() < 0.1


class TestCalibres:
    @pytest.mark.parametrize(
        ("valeur", "calibre"),
        [(0.1, 0.2), (0.2, 0.2), (0.5, 1), (1, 1), (2, 5), (5, 5), (7, 10), (10, 10), (12, 10)],
    )
    def test_get_calibre_prend_le_calibre_immediatement_superieur(self, valeur, calibre):
        assert Sysam.get_calibre(valeur) == calibre

    def test_un_calibre_pour_toutes_les_voies(self, silence):
        with Sysam([0, 1], 5) as can:
            assert can.calibres == [5.0, 5.0] and can.voies == [0, 1]

    def test_illisible_ou_negatif_donne_10_v(self, silence):
        with Sysam([0, 4], [np.nan, -1]) as can:
            assert can.calibres == [10.0, 10.0]

    def test_au_dela_de_10_v_ramene_a_10_v_en_le_disant(self, silence, capsys):
        with Sysam([0], 12) as can:
            assert can.calibres == [10.0] and can._calibres == [10.0]
        assert "ramené à 10 V" in capsys.readouterr().out

    def test_reconfigurer_en_cours_de_route(self, silence):
        with Sysam([2]) as can:
            can.config_entrees([1, 5], 0.2)
            assert can.voies == [1, 5] and can.calibres == [0.2, 0.2]

    def test_un_calibre_par_voie_exige(self, silence):
        with pytest.raises(ValueError, match="2 calibres pour 3 voies"):
            Sysam([0, 1, 2], [1, 5])


class TestVoies:
    def test_calibres_suivent_leur_voie_dans_le_desordre(self, silence):
        with Sysam([1, 0], [10, 0.2]) as can:
            assert can._voies == [0, 1] and can._calibres == [0.2, 10.0]  # ce que reçoit le pilote
            can.config_echantillon(1e-5, 500)
            _, tensions = can.acquerir()
        # les lignes dans l'ordre demandé : EA1 (10 V, pas de 4,9 mV), puis EA0 (0,2 V, pas de 0,1 mV)
        pas_10_v = 20 / 4096
        assert np.allclose(tensions[0] / pas_10_v, np.round(tensions[0] / pas_10_v))
        assert abs(tensions[1]).max() < pas_10_v <= abs(tensions[0]).max()

    def test_voie_en_double_refusee(self, silence):
        with pytest.raises(ValueError, match="deux fois"):
            Sysam([0, 0])

    def test_entree_moins_d_un_differentiel_refusee(self, silence):
        with pytest.raises(ValueError, match="entrée moins"):
            Sysam([0, 4], diff=[0])

    def test_te_min_selon_les_modules(self):
        assert Sysam.te_min([0, 1, 2, 3]) == Sysam.TE_MIN_DIRECT
        assert Sysam.te_min([0, 4]) == Sysam.TE_MIN_MULTIPLEX
        assert Sysam.te_min([0, 4], diff=[0]) == Sysam.TE_MIN_DIRECT


class TestEchantillonnage:
    @pytest.mark.parametrize("te", [0.7e-6, 1.4e-6, 2.5e-6, 1e-5, 3e-6])
    def test_periode_arrondie_au_dixieme_de_microseconde(self, silence, te):
        with Sysam([0], 5) as can:
            can.config_echantillon(te, 300)
            assert can._te_us == pytest.approx(te * 1e6) and can.te == pytest.approx(te)
            t, _ = can.acquerir()
        assert np.diff(t[0]) == pytest.approx(te)

    def test_trop_de_points_signale(self, silence, capsys):
        with Sysam([0, 1], 5) as can:
            can.config_echantillon(1e-6, 200_000)
        assert "la centrale en rendra moins" in capsys.readouterr().out

    def test_reduction(self, silence):
        with Sysam([0], 5) as can:
            can.config_echantillon(1e-5, 100)
            t, u = can.acquerir(reduction=4)
        assert t.shape == u.shape == (1, 25)


class TestSorties:
    def test_n_max_tient_dans_la_memoire(self, silence):
        n = Sysam.n_max(2, 1)
        with Sysam([0, 1], 5) as can:
            can.config_echantillon(2e-7, n)
            t, _ = can.acquerir_avec_sorties(np.zeros(n))
        assert t.shape == (2, n)

    def test_n_max_sans_la_marge_ne_tient_pas(self, silence):
        n = Sysam.MEMOIRE // 3
        with Sysam([0, 1], 5) as can:
            can.config_echantillon(2e-7, n)
            with pytest.raises(ValueError, match="memoire insuffisante"):
                can.acquerir_avec_sorties(np.zeros(n))

    def test_une_liste_compte_comme_une_sortie(self, silence):
        n = 100_000
        with Sysam([0, 1], 5) as can:
            can.config_echantillon(2e-7, n)
            with pytest.raises(ValueError, match="memoire insuffisante"):
                can.acquerir_avec_sorties([0.0] * n)

    def test_un_nombre_est_une_tension_constante(self, silence):
        with Sysam([0], 5) as can:
            can.config_echantillon(1e-6, 100)
            can.acquerir_avec_sorties(2.5, 0.0)
            assert can._ram_sorties == [1, 1]

    def test_periode_non_multiple_de_0_2_us_refusee(self, silence):
        with Sysam([0], 5) as can:
            can.config_echantillon(7e-7, 100)
            with pytest.raises(ValueError, match="multiple de 0,2"):
                can.acquerir_avec_sorties(np.zeros(10))
            can.acquerir()  # sans sortie, c'est permis

    def test_sortie_2d_refusee(self, silence):
        with Sysam([0], 5) as can:
            can.config_echantillon(1e-6, 100)
            with pytest.raises(ValueError, match="1D"):
                can.acquerir_avec_sorties(np.zeros((2, 10)))


class TestDeclenchement:
    @pytest.mark.parametrize(
        ("voies", "calibres", "voie"), [([1], [1], 1), ([2, 3], [10, 0.2], 3), ([0], [5], 0)]
    )
    def test_seuil_applique_est_le_seuil_demande(self, silence, voies, calibres, voie):
        with Sysam(voies, calibres) as can:
            can.config_trigger(voie, 0.15)
            assert can.seuil_applique == pytest.approx(0.15, abs=2 * max(calibres) / 4096)


class TestAnciensNoms:
    """Ce que 2026.9 écrivait fonctionne, sans avertissement."""

    def test_n_max_est_la_memoire_entiere(self, silence):
        assert Sysam.N_MAX == Sysam.MEMOIRE == 0x3FFFF
        with Sysam([0]) as can:
            assert can.N_MAX == 0x3FFFF

    def test_l_entier_0_voulait_dire_pas_de_sortie(self, silence):
        """pycanum ignorait tout ce qui n'était pas un ndarray : 0 ne générait rien."""
        with Sysam([0], 5) as can:
            can.config_echantillon(1e-6, 100)
            can.acquerir_avec_sorties(np.zeros(10), 0)
            assert can._ram_sorties == [10, 0]
            can.config_echantillon(7e-7, 100)  # pas un multiple de 0,2 µs : permis sans sortie
            temps, _ = can.acquerir_avec_sorties(0, 0)
            assert temps.shape == (1, 100) and can._ram_sorties == [10, 0]  # comptés jusqu'à la fermeture

    def test_modules_analog_est_un_dict(self):
        assert Sysam.MODULES_ANALOG[0] == (0, 4) and list(Sysam.MODULES_ANALOG.items())[3] == (3, (3, 7))
