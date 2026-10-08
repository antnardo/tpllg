import numpy as np
import pytest

from tpllg.fft import calcule_DFT, spectre


class TestCalculeDFT:
    @pytest.mark.parametrize(("fe", "N"), [(20000.0, 20000), (100000.0, 10000), (50000.0, 12500)])
    def test_amplitude_en_volts_quelle_que_soit_la_duree(self, fe, N):
        t = np.arange(N) / fe  # 440 Hz : un nombre entier de périodes
        f, a = calcule_DFT(t, 1.5 * np.sin(2 * np.pi * 440 * t) + 0.5)
        k = np.argmax(a[1:]) + 1
        assert f[k] == pytest.approx(440) and a[k] == pytest.approx(1.5, abs=1e-9)
        assert a[0] == pytest.approx(0.5, abs=1e-9) and np.delete(a, [0, k]).max() < 1e-9
        assert f[-1] < fe / 2 and f[1] == pytest.approx(fe / N)

    def test_nombre_impair_de_points(self):
        fe, N = 10000.0, 1001
        t = np.arange(N) / fe
        f, a = calcule_DFT(t, 2 * np.cos(2 * np.pi * (fe / N) * 7 * t))  # sept périodes exactes
        assert len(f) == 501 and f[-1] < fe / 2
        assert a[7] == pytest.approx(2) and np.delete(a, 7).max() < 1e-9

    def test_phases(self):
        """Le cas de traitementsignal/exemple_fourier : c'est l'argument qui donne la phase."""
        amplitudes = [1.59, 2.5, 1.06, 0, 0.212, 0, 0.0909]
        phases = [0, np.pi / 2, 0.3, 0, -1.2, 0, 2.0]
        t = np.arange(4000) / 4000
        s = sum(A * np.cos(2 * np.pi * 200 * n * t + p) for n, (A, p) in enumerate(zip(amplitudes, phases)))
        f, a, phi = calcule_DFT(t, s, phases=True)
        assert np.allclose(a[::200][:7], amplitudes) and np.allclose(f[::200][:7], 200 * np.arange(7))
        presentes = [1, 2, 4, 6]
        assert np.allclose(phi[::200][presentes], np.array(phases)[presentes])

    def test_une_voie_a_la_fois(self):
        with pytest.raises(ValueError, match="une seule voie"):
            calcule_DFT(np.zeros((2, 10)), np.zeros((2, 10)))


class TestSpectre:
    def test_fenetre_reduit_les_fuites(self):
        t = np.arange(20000) / 20000.0
        u = np.sin(2 * np.pi * 437.3 * t)  # pas un nombre entier de périodes
        _, a = calcule_DFT(t, u)
        f2, a2 = spectre(t, u)
        moitie = len(a2) // 2
        assert a.max() < 0.9  # la TFD brute perd de l'amplitude
        assert a2[:moitie].max() == pytest.approx(1, abs=0.005)
        assert f2[np.argmax(a2[:moitie])] == pytest.approx(437.3, abs=0.3)

    def test_composante_continue_est_la_moyenne(self):
        """spectre la doublait : 1,0 au lieu de 0,5 pour un offset de 0,5 V."""
        t = np.arange(20000) / 20000.0
        _, a = spectre(t, 0.5 + np.sin(2 * np.pi * 437 * t))
        assert a[0] == pytest.approx(0.5, abs=1e-9)

    def test_pas_de_frequence_et_miroir(self):
        fe, N, p = 20000.0, 4000, 3
        t = np.arange(N) / fe
        f, a = spectre(t, 0.8 * np.sin(2 * np.pi * 500 * t), p=p)
        M = (p + 1) * N
        assert len(f) == M and f[1] == pytest.approx(fe / M) and f[-1] == pytest.approx(fe - fe / M)
        moitie = M // 2
        assert a[:moitie].max() == pytest.approx(0.8, abs=0.004)
        assert f[np.argmax(a[:moitie])] == pytest.approx(500, abs=f[1])
        assert np.allclose(a[1:moitie], a[M - 1 : moitie : -1])  # la seconde moitié est le miroir
