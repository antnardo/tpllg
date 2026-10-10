import numpy as np
import pytest

from tpllg.fft import calcule_DFT
from tpllg.traitement import (
    choix_echantillonnage,
    detecte_maxima_secondaires,
    fonction_transfert,
    gain,
    gain_std,
    indices_plages,
    interpolation_fft,
    valeurs_correspondantes,
)

G0, PHI0 = 0.5, -1.2


def entree_sortie(Np, n_periodes=50.0, offsets=(0.0, 0.0), fe=100000.0):
    f = fe / Np
    t = np.arange(round(n_periodes * Np)) / fe
    e = 2 * np.cos(2 * np.pi * f * t) + offsets[0]
    s = G0 * 2 * np.cos(2 * np.pi * f * t + PHI0) + offsets[1]
    return f, t, e, s


class TestFonctionTransfert:
    @pytest.mark.parametrize("Np", [10.0, 13.0, 50.6, 81.04, 101.3])
    def test_gain_et_phase_quel_que_soit_np(self, Np):
        """La phase de gain_std se trompait de 0,15° (Np ≈ 100) à 3° (Np = 10), d = Np/4 étant tronqué."""
        f, t, e, s = entree_sortie(Np)
        H = fonction_transfert(t, e, s, f)
        assert abs(H) == pytest.approx(G0, rel=1e-6) and np.degrees(np.angle(H) - PHI0) == pytest.approx(
            0, abs=1e-4
        )

    def test_signaux_non_centres(self):
        """Des offsets de 10 % et 5 % biaisaient la phase de 0,63°."""
        f, t, e, s = entree_sortie(100.0, offsets=(0.2, 0.05))
        H = fonction_transfert(t, e, s, f)
        assert abs(H) == pytest.approx(G0, rel=1e-6) and np.angle(H) == pytest.approx(PHI0, abs=1e-6)

    def test_nombre_non_entier_de_periodes(self):
        f, t, e, s = entree_sortie(37.0, n_periodes=20.37)
        assert fonction_transfert(t, e, s, f) == pytest.approx(G0 * np.exp(1j * PHI0), abs=1e-5)

    def test_sans_la_frequence(self):
        _, t, e, s = entree_sortie(77.7, n_periodes=30.4)
        assert fonction_transfert(t, e, s) == pytest.approx(G0 * np.exp(1j * PHI0), abs=1e-4)

    def test_dans_le_bruit(self, rng):
        f, t, e, s = entree_sortie(100.0)
        e = e + rng.normal(0, 0.05, e.size)
        s = s + rng.normal(0, 0.05, s.size)
        H = fonction_transfert(t, e, s, f)
        assert abs(H) == pytest.approx(G0, rel=0.01) and np.angle(H) == pytest.approx(PHI0, abs=0.01)

    def test_tableaux_mal_formes_refuses(self):
        with pytest.raises(ValueError, match="1D"):
            fonction_transfert(np.zeros(10), np.zeros(10), np.zeros(9), 100.0)


def test_interpolation_fft_est_exacte_sur_un_signal_a_bande_limitee():
    N, k = 64, 3
    n = np.arange(N)
    y = interpolation_fft(np.cos(2 * np.pi * 3 * n / N) + 0.5 * np.sin(2 * np.pi * 7 * n / N), k)
    m = np.arange(N * (k + 1)) / (k + 1)  # les mêmes instants, quatre fois plus fins
    attendu = np.cos(2 * np.pi * 3 * m / N) + 0.5 * np.sin(2 * np.pi * 7 * m / N)
    assert y.shape == (N * (k + 1),) and np.allclose(y, attendu, atol=1e-10)


class TestChoixEchantillonnage:
    @pytest.mark.parametrize("freq", [50.0, 1000.0, 12345.0])
    def test_respecte_les_contraintes(self, freq, capsys):
        temin, Npmin, permin, Nmax, Tmax = 2e-7, 100, 20, 87296, 1.0
        te, n = choix_echantillonnage(freq, temin, Npmin, permin, Nmax, Tmax)
        assert te >= temin and te / temin == pytest.approx(round(te / temin))  # un multiple du pas minimal
        assert 1 / (freq * te) >= Npmin - 1e-9 and n <= Nmax and n * te <= Tmax + 1e-12
        assert n * te * freq >= permin and "[WARNING]" not in capsys.readouterr().out

    @pytest.mark.parametrize(("freq", "te"), [(1000.0, 1e-5), (500.0, 2e-5), (2500.0, 4e-6), (250.0, 4e-5)])
    def test_pas_juste_malgre_les_arrondis(self, freq, te):
        """1/(100 × 1000 × 2e-7) vaut 49,999… : int() prenait 49 pas de 0,2 µs au lieu de 50."""
        assert choix_echantillonnage(freq, 2e-7, 100, 20, 87296, 1.0)[0] == pytest.approx(te)

    def test_previent_quand_il_ne_peut_pas(self, capsys):
        te, n = choix_echantillonnage(2e6, 2e-7, 100, 20, 262144, 1.0)  # 2,5 points par période au mieux
        assert te == 2e-7 and n == 262144 and "points par période faible" in capsys.readouterr().out
        te, n = choix_echantillonnage(10.0, 2e-7, 100, 20, 262144, 1.0)  # dix périodes en une seconde
        assert n * te == pytest.approx(1, abs=2e-3) and "périodes faible" in capsys.readouterr().out


class TestHarmoniques:
    def test_harmoniques_d_un_creneau_echantillonne(self):
        Np, n_periodes = 100, 100  # créneau de 100 Hz à 10 kHz : 50 points hauts, 50 bas
        x = np.tile(np.r_[np.ones(Np // 2), -np.ones(Np // 2)], n_periodes)
        freq, a = calcule_DFT(np.arange(x.size) * 1e-4, x)
        n = np.arange(1, 50)
        attendu = np.where(
            n % 2 == 1, 4 / (Np * np.sin(np.pi * n / Np)), 0
        )  # l'harmonique n d'un créneau échantillonné
        assert np.allclose(a[100 * n], attendu, atol=1e-9)
        plages = indices_plages(freq, 100, delta_freq=20)
        assert len(plages) == 49 and plages[0] == (90, 110) and plages[-1] == (4890, 4910)
        pics = detecte_maxima_secondaires(a, plages, seuil=0.06)
        assert pics == [100 * k for k in range(1, 25, 2)]  # les impairs jusqu'à 23 : au-delà, sous le seuil
        assert np.allclose(freq[pics], 100 * np.arange(1, 25, 2))

    def test_plages_centrees_sur_chaque_harmonique(self):
        """f1 = 437,3 Hz, df = 1 Hz : les plages dérivaient, 16 impaires sur 28 manquées dès n = 25."""
        t = np.arange(50000) / 50000.0
        freq, a = calcule_DFT(t, np.sign(np.sin(2 * np.pi * 437.3 * t)))
        plages = indices_plages(freq, 437.3, delta_freq=0.3 * 437.3)
        assert len(plages) == int(freq[-1] / 437.3 + 0.5)
        for rang, (debut, fin) in enumerate(plages, start=1):
            assert debut <= round(rang * 437.3) < fin
        pics = detecte_maxima_secondaires(a, plages, seuil=0.02)
        assert np.allclose(freq[pics] / 437.3, np.round(freq[pics] / 437.3), atol=0.01)

    def test_fondamental_hors_du_spectre(self):
        freq = np.arange(1000.0)
        with pytest.raises(ValueError, match="hors du spectre"):
            indices_plages(freq, 30000.0, 10)

    def test_plage_qui_deborderait_sous_zero(self):
        assert indices_plages(np.arange(1000.0), 3.0, delta_freq=10)[0] == (0, 8)


def test_valeurs_correspondantes_apparie_a_delta_pres():
    i1, i2 = valeurs_correspondantes([100, 300, 500, 700, 900], [101, 299, 700, 901, 1100], 2)
    assert np.array_equal(i1, [100, 300, 700, 900]) and np.array_equal(i2, [101, 299, 700, 901])


class TestAnciensNoms:
    """gain et gain_std rendent (G, phi) comme en 2026.9, par fonction_transfert."""

    def test_gain_std(self):
        _, t, e, s = entree_sortie(Np=40)
        with pytest.warns(DeprecationWarning, match="fonction_transfert"):
            g, phi = gain_std(t, e, s, Np=40)
        H = fonction_transfert(t, e, s)
        assert g == abs(H) and phi == np.angle(H)
        assert g == pytest.approx(G0, abs=1e-6) and phi == pytest.approx(PHI0, abs=1e-6)

    def test_gain(self):
        f, t, e, s = entree_sortie(Np=40)
        with pytest.warns(DeprecationWarning, match="fonction_transfert"):
            g, phi = gain(t, e, s, f, 40, "std")
        H = fonction_transfert(t, e, s, f)
        assert (g, phi) == (abs(H), np.angle(H))
        with pytest.warns(DeprecationWarning), pytest.raises(NotImplementedError):
            gain(t, e, s, f, 40, "fit")
