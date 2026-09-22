import numpy as np

from tpllg.signaux import decrement_logarithmique, extremums, fronts_montants, frequence_pic, front_utile


def test_fronts_montants_d_un_creneau():
    fe = 100000.0
    t = np.arange(3000)/fe
    v = np.sign(np.sin(2*np.pi*200*t + 1.0)) + np.random.RandomState(2).normal(0, 0.01, t.size)
    fronts, (bas, haut) = fronts_montants(t, v)
    assert abs(bas + 1) < 0.05 and abs(haut - 1) < 0.05
    attendus = [(k - 1.0/(2*np.pi))/200 for k in range(0, 8)]
    attendus = [a for a in attendus if 0 < a < t[-1]]
    assert len(fronts) == len(attendus)
    assert np.allclose(fronts, attendus, atol=2/fe)
    t0, periode = front_utile(t, fronts)
    assert abs(periode - 1/200) < 2/fe and t0 == fronts[0]


def test_frequence_et_decrement_d_une_oscillation_amortie():
    fe, f0, Q = 200000.0, 2000.0, 6.0
    t = np.arange(2000)/fe
    v = -1.5*np.exp(-np.pi*f0*t/Q)*np.sin(2*np.pi*f0*t)
    assert abs(frequence_pic(v, 1/fe) - f0) < 150
    pics = extremums(v, fe, f0)
    alpha = decrement_logarithmique(t[pics], v[pics])
    assert abs(np.pi*f0/alpha - Q) < 0.3
