import numpy as np

from tpllg import sysam_factice
from tpllg.acquisition import acquerir, sauvegarder


def test_sauvegarder_puis_relire(tmp_path):
    temps = np.array([[0, 1e-5, 2e-5], [0, 1e-5, 2e-5]])
    tensions = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]])
    sauvegarder(str(tmp_path/"essai"), [0, 3], temps, tensions)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["essai_EA0.txt", "essai_EA3.txt"]
    t, u = np.loadtxt(str(tmp_path/"essai_EA3.txt"))
    assert np.array_equal(t, temps[1]) and np.array_equal(u, tensions[1])


def test_acquerir_regle_le_declenchement(capsys, monkeypatch):
    monkeypatch.setattr(sysam_factice, "VERBOSE", True)
    temps, tensions = acquerir([1], 1, 2e-6, 50, trigger=(1, 0.5, 5, 0))
    sortie = capsys.readouterr().out
    assert "EA1" in sortie and "seuil 0.5 V" in sortie and "descendant" in sortie and "5 points avant" in sortie
    assert temps.shape == (1, 50) and abs(temps[0, -1] - 49*2e-6) < 1e-15
    acquerir([0], 1, 2e-6, 50, trigger=(0, -0.5, 10))
    assert "front montant, 10 points avant" in capsys.readouterr().out
