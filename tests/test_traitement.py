import numpy as np

from tpllg.fft import calcule_DFT
from tpllg.traitement import (choix_echantillonnage, detecte_maxima_secondaires, gain, gain_std,
                              indices_plages, interpolation_fft, valeurs_correspondantes)


def test_gain_std_retrouve_le_gain_et_la_phase():
    fe, f, N = 100000.0, 1000.0, 10000                 # 100 périodes entières, 100 points par période
    Np = int(fe/f)
    t = np.arange(N)/fe
    G0, phi0 = 0.5, -1.2
    e = 2*np.cos(2*np.pi*f*t)
    s = G0*2*np.cos(2*np.pi*f*t + phi0)
    G, phi = gain_std(t, e, s, Np)
    assert abs(G - G0) < 1e-12                          # rapport des valeurs efficaces, exact
    assert abs(phi - phi0) < 5e-3                       # la moyenne du produit, sur 99,75 périodes
    G, phi = gain_std(t, e, s, Np, ninter=1)            # interpolé : le même résultat
    assert abs(G - G0) < 1e-9 and abs(phi - phi0) < 5e-3
    assert gain(t, e, s, f, Np, "std") == gain_std(t, e, s, Np)
    assert gain(t, e, s, f, Np, "fit") is NotImplemented


def test_interpolation_fft_est_exacte_sur_un_signal_a_bande_limitee():
    N, k = 64, 3
    n = np.arange(N)
    x = np.cos(2*np.pi*3*n/N) + 0.5*np.sin(2*np.pi*7*n/N)
    y = interpolation_fft(x, k)
    m = np.arange(N*(k + 1))/(k + 1)                    # les mêmes instants, quatre fois plus fins
    attendu = np.cos(2*np.pi*3*m/N) + 0.5*np.sin(2*np.pi*7*m/N)
    assert y.shape == (N*(k + 1),) and np.allclose(y, attendu, atol=1e-10)


def test_choix_echantillonnage_respecte_les_contraintes(capsys):
    temin, Npmin, permin, Nmax, Tmax = 2e-7, 100, 20, 262144, 1.0
    for freq in (50.0, 1000.0, 12345.0):
        te, n = choix_echantillonnage(freq, temin, Npmin, permin, Nmax, Tmax)
        assert te >= temin - 1e-15
        assert abs(te/temin - round(te/temin)) < 1e-6   # un multiple du pas minimal
        assert 1/(freq*te) >= Npmin - 1e-9              # au moins Npmin points par période
        assert n <= Nmax and n*te <= Tmax + 1e-12
        assert n*te*freq >= permin
    assert "[WARNING]" not in capsys.readouterr().out


def test_choix_echantillonnage_previent_quand_il_ne_peut_pas(capsys):
    te, n = choix_echantillonnage(2e6, 2e-7, 100, 20, 262144, 1.0)   # 2 MHz : 2,5 points par période au mieux
    assert te == 2e-7 and n == 262144
    assert "points par période faible" in capsys.readouterr().out
    te, n = choix_echantillonnage(10.0, 2e-7, 100, 20, 262144, 1.0)  # 10 Hz : dix périodes en une seconde
    assert abs(n*te - 1) < 2e-3
    assert "périodes faible" in capsys.readouterr().out


def test_harmoniques_d_un_creneau_echantillonne():
    Np, n_periodes = 100, 100                           # créneau de 100 Hz à 10 kHz : 50 points hauts, 50 bas
    x = np.tile(np.r_[np.ones(Np//2), -np.ones(Np//2)], n_periodes)
    t = np.arange(x.size)*1e-4
    freq, a = calcule_DFT(t, x)
    n = np.arange(1, 50)
    attendu = np.where(n % 2 == 1, 4/(Np*np.sin(np.pi*n/Np)), 0)   # l'harmonique n d'un créneau échantillonné
    assert np.allclose(a[100*n], attendu, atol=1e-9)
    plages = indices_plages(freq, 100, delta_freq=20)
    assert len(plages) == 49 and plages[0] == (90, 110) and plages[-1] == (4890, 4910)
    pics = detecte_maxima_secondaires(a, plages, seuil=0.06)
    assert pics == [100*k for k in range(1, 25, 2)]       # les impairs jusqu'à 23 : au-delà, sous le seuil
    assert np.allclose(freq[pics], 100*np.arange(1, 25, 2))


def test_valeurs_correspondantes_apparie_a_delta_pres():
    i1, i2 = valeurs_correspondantes([100, 300, 500, 700, 900], [101, 299, 700, 901, 1100], 2)
    assert np.array_equal(i1, [100, 300, 700, 900]) and np.array_equal(i2, [101, 299, 700, 901])
