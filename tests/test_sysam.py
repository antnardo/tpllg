from tpllg.acquisition import acquerir


def test_acquerir_sans_centrale_rend_les_deux_voies():
    """Sans pycanum, la centrale simulée répond avec la même forme de données."""
    temps, tensions = acquerir([0, 1], calibre=5, te=1e-5, nbpoints=200, trigger=(0, 0.0, 10))
    assert len(temps) == 2 and len(tensions) == 2
    assert len(temps[0]) == 200 and len(tensions[1]) == 200
