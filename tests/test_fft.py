import numpy as np

from tpllg.fft import calcule_DFT, spectre


def test_calcule_DFT_rend_l_amplitude_en_volts_quelle_que_soit_la_duree():
    for fe, N in ((20000.0, 20000), (100000.0, 10000), (50000.0, 12500)):   # nombres entiers de périodes
        t = np.arange(N)/fe
        u = 1.5*np.sin(2*np.pi*440*t) + 0.5
        f, a = calcule_DFT(t, u)
        assert abs(f[np.argmax(a[1:]) + 1] - 440) <= fe/N
        assert abs(a[1:].max() - 1.5) < 0.05
        assert abs(a[0] - 0.5) < 1e-6
        assert f[-1] < fe/2 and abs(f[1] - fe/N) < 1e-9


def test_spectre_fenetre_reduit_les_fuites():
    fe, N = 20000.0, 20000
    t = np.arange(N)/fe
    u = np.sin(2*np.pi*437.3*t)              # pas un nombre entier de périodes
    f, a = calcule_DFT(t, u)
    f2, a2 = spectre(t, u)
    assert a.max() < 0.9                     # la TFD brute perd de l'amplitude
    assert abs(a2[:len(a2)//2].max() - 1) < 0.005
    assert abs(f2[np.argmax(a2[:len(a2)//2])] - 437.3) < 0.3
