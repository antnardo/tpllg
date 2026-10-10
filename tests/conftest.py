"""Les tests passent toujours par la centrale simulée : pycanum est rendu
introuvable avant tout import de tpllg.sysam, même sur un poste qui l'a. Le
simulateur n'attend pas la durée des acquisitions, et matplotlib dessine hors
écran."""

import sys

import matplotlib
import pytest

sys.modules["pycanum"] = None
sys.modules["pycanum.main"] = None
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def sans_attente(monkeypatch):
    from tpllg import sysam_factice

    monkeypatch.setattr(sysam_factice.time, "sleep", lambda duree: None)


@pytest.fixture(autouse=True)
def fermer_les_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def rng():
    """Des tirages reproductibles, les mêmes à chaque test (un RandomState
    sous un numpy sans default_rng : il a normal et permutation aussi)."""
    from tpllg._interne import generateur

    return generateur(20261008)
