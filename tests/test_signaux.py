import numpy as np
import pytest

from tpllg.signaux import (decrement_logarithmique, extremums, fenetre, fronts_descendants,
                           fronts_montants, frequence_pic, front_utile)


def creneau(fe=100000.0, n=3000, f=200.0, phase=1.0):
    t = np.arange(n)/fe
    v = np.sign(np.sin(2*np.pi*f*t + phase)) + np.random.RandomState(2).normal(0, 0.01, t.size)
    return t, v


def test_fronts_montants_d_un_creneau():
    fe = 100000.0
    t, v = creneau(fe)
    fronts, (bas, haut) = fronts_montants(t, v)
    assert abs(bas + 1) < 0.05 and abs(haut - 1) < 0.05
    attendus = [(k - 1.0/(2*np.pi))/200 for k in range(0, 8)]       # sin(2 pi 200 t + 1) = 0, montant
    attendus = [a for a in attendus if 0 < a < t[-1]]
    assert len(fronts) == len(attendus)
    assert np.allclose(fronts, attendus, atol=2/fe)
    t0, periode = front_utile(t, fronts)
    assert abs(periode - 1/200) < 2/fe and t0 == fronts[0]


def test_fronts_descendants_d_un_creneau():
    fe = 100000.0
    t, v = creneau(fe)
    fronts, (bas, haut) = fronts_descendants(t, v)
    assert abs(bas + 1) < 0.05 and abs(haut - 1) < 0.05
    attendus = [(k + 0.5 - 1.0/(2*np.pi))/200 for k in range(0, 8)]  # sin(2 pi 200 t + 1) = 0, descendant
    attendus = [a for a in attendus if 0 < a < t[-1]]
    assert len(fronts) == len(attendus)
    assert np.allclose(fronts, attendus, atol=2/fe)


def test_front_utile():
    t = np.linspace(0, 1, 1001)
    t0, periode = front_utile(t, [0.12, 0.32, 0.52, 0.73])
    assert t0 == 0.12 and abs(periode - 0.2) < 1e-12                 # la médiane des écarts : 0,2, 0,2, 0,21
    t0, periode = front_utile(t, [0.4])
    assert t0 == 0.4 and abs(periode - 0.6) < 1e-12                  # un seul front : jusqu'à la fin
    with pytest.raises(ValueError):
        front_utile(t, [])


def test_fenetre_est_un_intervalle_semi_ouvert():
    t = np.arange(10.0)
    tf, vf = fenetre(t, t**2, 2, 3)
    assert np.array_equal(tf, [2, 3, 4]) and np.array_equal(vf, [4, 9, 16])


def test_frequence_pic_exacte_sur_un_nombre_entier_de_periodes():
    fe, N = 10000.0, 1000
    t = np.arange(N)/fe
    assert abs(frequence_pic(3 + np.cos(2*np.pi*370*t), 1/fe) - 370) < 1e-6   # la moyenne ne compte pas


def test_frequence_et_decrement_d_une_oscillation_amortie():
    fe, f0, Q = 200000.0, 2000.0, 6.0
    t = np.arange(2000)/fe
    v = -1.5*np.exp(-np.pi*f0*t/Q)*np.sin(2*np.pi*f0*t)
    assert abs(frequence_pic(v, 1/fe) - f0) < 150
    pics = extremums(v, fe, f0)
    alpha = decrement_logarithmique(t[pics], v[pics])
    assert abs(np.pi*f0/alpha - Q) < 0.3


def test_extremums_autour_d_un_offset():
    fe, f0, Q = 200000.0, 2000.0, 6.0
    t = np.arange(2000)/fe
    v = 2.0 + 1.5*np.exp(-np.pi*f0*t/Q)*np.sin(2*np.pi*f0*t)
    pics = extremums(v, fe, f0, offset=2.0)
    assert np.all(np.diff(t[pics]) > 0.4/f0)
    assert abs(np.median(np.diff(t[pics])) - 0.5/f0) < 2/fe          # une demi-période entre deux extremums
    alpha = decrement_logarithmique(t[pics], v[pics], offset=2.0)
    assert abs(np.pi*f0/alpha - Q) < 0.3
