import numpy as np

from tpllg.acquisition import acquerir
from tpllg.sysam import Sysam


def test_acquerir_sans_centrale_rend_la_forme_de_pycanum():
    """Sans pycanum, la centrale simulée répond comme pycanum : deux tableaux
    2D de double, une ligne par voie, le temps en secondes."""
    temps, tensions = acquerir([0, 1], calibre=5, te=1e-5, nbpoints=200, trigger=(0, 0.0, 10))
    assert temps.shape == (2, 200) and tensions.shape == (2, 200)
    assert temps.dtype == np.float64 and tensions.dtype == np.float64
    assert abs(temps[0][1] - 1e-5) < 1e-12 and abs(temps[1][-1] - 199e-5) < 1e-12
    assert abs(tensions).max() < 0.1


def test_te_min_selon_les_modules():
    assert Sysam.te_min([0, 1, 2, 3]) == Sysam.TE_MIN_DIRECT
    assert Sysam.te_min([0, 4]) == Sysam.TE_MIN_MULTIPLEX


def test_get_calibre_prend_le_calibre_immediatement_superieur():
    assert [Sysam.get_calibre(v) for v in (0.1, 0.2, 0.5, 1, 2, 5, 7, 10, 12)] == [0.2, 0.2, 1, 1, 5, 5, 10, 10, 10]


def test_config_entrees_normalise_les_calibres():
    with Sysam([0, 1], 5) as can:
        assert can.calibres == [5.0, 5.0] and can.voies == [0, 1]
    with Sysam([0, 4], [np.nan, -1]) as can:                      # illisible ou négatif : le défaut, 10 V
        assert can.calibres == [10.0, 10.0]
    with Sysam([2]) as can:
        assert can.calibres == [10.0]
        can.config_entrees([1, 5], 0.2, diff=[1])                    # reconfigurer en cours de route
        assert can.voies == [1, 5] and can.calibres == [0.2, 0.2] and can.diff == [1]


def test_config_echantillon_convertit_en_microsecondes(capsys):
    with Sysam([0], 5) as can:
        can.config_echantillon(2.5e-6, 8)
        assert abs(can.techant - 2.5) < 1e-12 and can.nbpoints == 8   # pycanum compte en µs
        t, u = can.acquerir()
        assert np.allclose(t[0], np.arange(8)*2.5e-6) and abs(u).max() <= 5
        t2, u2 = can.acquerir_avec_sorties(np.zeros(8), 0)
        assert t2.shape == (1, 8) and u2.shape == (1, 8) and u2.dtype == np.float64
        can.verbose = True
        can.config_quantification(8)
        assert "8 bits" in capsys.readouterr().out


def test_reduction():
    with Sysam([0], 5) as can:
        can.config_echantillon(1e-5, 100)
        can.acquerir()
        assert can.temps(reduction=4).shape == (1, 25)
