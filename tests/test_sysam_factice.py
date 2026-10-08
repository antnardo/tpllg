import numpy as np
import pytest

from tpllg.sysam_factice import MEMOIRE, POINTS_SORTIE_MAX, Sysam


@pytest.fixture
def can():
    can = Sysam("SP5")
    can.verbose = False
    return can


class TestEntrees:
    def test_calibre_au_dessus_de_10_v_refuse(self, can):
        with pytest.raises(ValueError, match="10 V"):
            can.config_entrees([0], [12.0])

    def test_un_calibre_par_voie_exige(self, can):
        with pytest.raises(ValueError, match="nombre de calibres"):
            can.config_entrees([0, 1], [5.0])

    @pytest.mark.parametrize(
        ("demande", "applique"), [(0.1, 0.2), (0.2, 0.2), (0.5, 1.0), (1.0, 1.0), (2.0, 5.0), (7.0, 10.0)]
    )
    def test_le_plus_petit_calibre_qui_contient(self, can, demande, applique):
        can.config_entrees([0], [demande])
        assert can._calibres == [applique]

    def test_voies_rangees_et_calibres_dans_l_ordre_donne(self, can):
        can.config_entrees([1, 0], [10.0, 0.2])  # ce que fait le pilote : EA0 prend 10 V
        assert can._voies == [0, 1] and can._calibres == [10.0, 0.2]

    def test_entree_moins_d_un_differentiel_retiree(self, can):
        with pytest.raises(ValueError, match="nombre de calibres"):
            can.config_entrees([0, 4], [5.0, 5.0], diff=[0])
        can.config_entrees([0, 4], [5.0], diff=[0])
        assert can._voies == [0]


class TestEchantillonnage:
    @pytest.mark.parametrize(("voies", "diff", "te_us"), [([0], [], 0.05), ([0, 4], [], 1.0)])
    def test_periode_sous_le_minimum_refusee(self, can, voies, diff, te_us):
        can.config_entrees(voies, [10.0] * len(voies), diff)
        with pytest.raises(ValueError, match="trop faible"):
            can.config_echantillon(te_us, 100)

    def test_deux_modules_differents_en_mode_direct(self, can):
        can.config_entrees([0, 1, 2, 3], [10.0] * 4)
        can.config_echantillon(0.1, 100)
        assert can._te_us == pytest.approx(0.1)

    @pytest.mark.parametrize(("demande", "applique"), [(0.7, 0.6), (1.4, 1.3), (2.5, 2.5), (10.0, 10.0)])
    def test_periode_en_flottant_32_bits_tronquee(self, can, demande, applique):
        can.config_entrees([0], [10.0])
        can.config_echantillon(demande, 100)
        assert can._te_us == pytest.approx(applique)

    def test_sans_entrees_refuse(self, can):
        with pytest.raises(ValueError, match="entrees non configurees"):
            can.config_echantillon(10.0, 100)

    def test_reduction_rend_floor_n_sur_r_points(self, can):
        can.config_entrees([0], [5.0])
        can.config_echantillon(10.0, 100)
        can.acquerir()
        assert can.temps(3).shape == (1, 33) and can.entrees(3).shape == (1, 33)
        assert np.allclose(can.temps(3)[0, :3], [0, 30e-6, 60e-6])

    def test_points_plafonnes_en_silence_par_la_memoire(self, can):
        can.config_entrees([0, 1], [5.0, 5.0])
        can.config_echantillon(0.2, 200_000)
        can.acquerir()
        assert can.temps().shape[1] <= MEMOIRE // 2 < 200_000

    def test_lire_avant_acquisition_refuse(self, can):
        can.config_entrees([0], [5.0])
        with pytest.raises(ValueError, match="aucune acquisition"):
            can.temps()


class TestSorties:
    def test_memoire_partagee_entre_entrees_et_sorties(self, can):
        can.config_entrees([0, 1], [5.0, 5.0])
        n = MEMOIRE // 3  # 2 voies + 1 sortie, sans la marge de l'arrondi par paquets
        can.config_echantillon(0.2, n)
        with pytest.raises(ValueError, match="memoire insuffisante"):
            can.acquerir_avec_sorties(np.zeros(n), None)

    def test_points_des_sorties_comptes_jusqu_a_la_fermeture(self, can):
        can.config_entrees([0], [5.0])
        can.config_echantillon(0.2, 80_000)
        can.acquerir_avec_sorties(np.zeros(80_000), np.zeros(80_000))
        can.config_echantillon(0.2, 110_000)  # seule, elle tiendrait
        with pytest.raises(ValueError, match="memoire insuffisante"):
            can.acquerir()

    def test_sortie_trop_longue_refusee(self, can):
        can.config_entrees([0], [5.0])
        can.config_echantillon(0.2, 100)
        with pytest.raises(ValueError, match="131071"):
            can.acquerir_avec_sorties(np.zeros(POINTS_SORTIE_MAX + 1), None)

    def test_sortie_plus_rapide_que_0_2_us_refusee(self, can):
        can.config_entrees([0], [5.0])
        can.config_echantillon(0.1, 100)
        with pytest.raises(ValueError, match=r"0\.2 microsecondes"):
            can.acquerir_avec_sorties(np.zeros(10), None)

    def test_tableau_d_entiers_refuse(self, can):
        can.config_entrees([0], [5.0])
        can.config_echantillon(1.0, 100)
        with pytest.raises(ValueError, match="format de l'echantillon 1"):
            can.acquerir_avec_sorties(np.zeros(10, dtype=int), None)

    def test_liste_ignoree_comme_par_pycanum(self, can):
        can.config_entrees([0], [5.0])
        can.config_echantillon(1.0, 100)
        can.acquerir_avec_sorties([1.0] * 10, None)
        assert can._ram_sorties == [0, 0]

    def test_periode_non_multiple_de_0_2_us_signalee(self, can, capsys):
        can.config_entrees([0], [5.0])
        can.config_echantillon(0.5, 100)  # les sorties ne tournent qu'à 0,4 ou 0,6 µs
        can.acquerir_avec_sorties(np.zeros(10), None)
        assert "désynchronise" in capsys.readouterr().out


class TestDeclenchement:
    def test_voie_non_configuree_refusee(self, can):
        can.config_entrees([0], [5.0])
        with pytest.raises(ValueError, match="voie de trigger"):
            can.config_trigger(1, 0.5)

    def test_seuil_converti_avec_le_calibre_de_la_position(self, can):
        can.config_entrees([1], [1.0])  # EA1 seule : la position 1 du tableau du pilote vaut 10 V
        can.config_trigger(1, 0.5)
        assert can.seuil_applique == pytest.approx(0.05, abs=1e-3)

    def test_seuil_juste_quand_la_voie_est_a_sa_place(self, can):
        can.config_entrees([0, 1], [5.0, 1.0])
        can.config_trigger(1, 0.5)
        assert can.seuil_applique == pytest.approx(0.5, abs=1e-3)


class TestNonSimule:
    @pytest.mark.parametrize(
        "methode", ["acquerir_permanent", "lire", "compteur", "chrono", "desactiver_lecture"]
    )
    def test_leve_not_implemented(self, can, methode):
        with pytest.raises(NotImplementedError, match="ne simule pas"):
            getattr(can, methode)()


def test_fermee_refuse_tout(can):
    can.fermer()
    with pytest.raises(ValueError, match="non ouvert"):
        can.config_entrees([0], [5.0])


def test_verbeux_des_la_creation(capsys):
    Sysam("SP5")
    assert "centrale simulée" in capsys.readouterr().out
