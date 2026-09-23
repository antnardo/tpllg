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


def test_reduction():
    with Sysam([0], 5) as can:
        can.config_echantillon(1e-5, 100)
        can.acquerir()
        assert can.temps(reduction=4).shape == (1, 25)
