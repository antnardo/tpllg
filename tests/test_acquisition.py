import numpy as np
import pytest

from tpllg import sysam_factice
from tpllg.acquisition import acquerir, sauvegarder


def test_sauvegarder_puis_relire(tmp_path):
    temps = np.array([[0, 1e-5, 2e-5], [0, 1e-5, 2e-5]])
    tensions = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]])
    sauvegarder(str(tmp_path / "essai"), [0, 3], temps, tensions)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["essai_EA0.txt", "essai_EA3.txt"]
    t, u = np.loadtxt(str(tmp_path / "essai_EA3.txt"))
    assert np.array_equal(t, temps[1]) and np.array_equal(u, tensions[1])


def test_sauvegarder_refuse_des_voies_en_trop(tmp_path):
    with pytest.raises(ValueError, match="3 voies"):
        sauvegarder(str(tmp_path / "essai"), [0, 1, 2], np.zeros((2, 3)), np.zeros((2, 3)))


def test_acquerir_regle_le_declenchement(capsys, monkeypatch):
    monkeypatch.setattr(sysam_factice, "VERBOSE", True)
    temps, _ = acquerir([1], 1, 2e-6, 50, trigger=(1, 0.5, 5, 0))
    sortie = capsys.readouterr().out
    assert all(texte in sortie for texte in ("EA1", "seuil 0.5 V", "descendant", "5 points avant"))
    assert temps.shape == (1, 50) and temps[0, -1] == pytest.approx(49 * 2e-6)
    acquerir([0], 1, 2e-6, 50, trigger=(0, -0.5, 10))
    assert "front montant, 10 points avant" in capsys.readouterr().out


def test_acquerir_rend_les_voies_dans_l_ordre_demande(monkeypatch):
    monkeypatch.setattr(sysam_factice, "VERBOSE", False)
    _, tensions = acquerir([3, 0], [0.2, 10], 1e-5, 400)
    pas_10_v = 20 / 4096  # EA3 sur 0,2 V : un bruit cinquante fois plus fin que EA0 sur 10 V
    assert abs(tensions[0]).max() < pas_10_v <= abs(tensions[1]).max()


def test_acquerir_refuse_un_declenchement_mal_forme():
    with pytest.raises(ValueError, match="trigger"):
        acquerir([0], 1, 1e-5, 10, trigger=(0, 0.5))
