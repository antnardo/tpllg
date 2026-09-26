import matplotlib.pyplot as plt
import numpy as np

from tpllg.ajustement import formater
from tpllg.bode import phase_0_360, tracer_bode


def gain(f, H0, f0, Q):
    return H0/(1 + 1j*Q*(f/f0 - f0/f))


def test_phase_0_360():
    phases = [0, np.pi/2, np.pi, -np.pi/2, -np.pi, 2*np.pi, 3*np.pi, np.radians(-175)]
    assert np.allclose(phase_0_360(np.array(phases)), [0, 90, 180, 270, 180, 0, 180, 185])


def test_tracer_bode_ecrit_la_figure_et_la_legende(tmp_path):
    f = np.geomspace(100, 10000, 15)
    H = gain(f, -5, 2000, 6)
    pfit, err = [-5.0, 2000.0, 6.0], [0.1, 2.0, 0.2]
    fichier = tmp_path/"bode.png"
    fig = tracer_bode(f, np.abs(H), np.angle(H), gain, pfit, err, noms=("H0", "f0", "Q"),
                      unites=("", "Hz", ""), fichier=str(fichier))
    ax1, ax2 = fig.axes
    assert fichier.exists() and fichier.stat().st_size > 0
    assert ax1.get_xscale() == "log" and ax1.get_yscale() == "log"
    textes = [t.get_text() for t in ax1.get_legend().get_texts()]
    assert any("f0 = " + formater(2000.0, 2.0, "Hz") in t for t in textes)   # « f0 = 2000.0 ± 2.0 Hz »
    x_ajuste, y_ajuste = ax1.get_lines()[1].get_data()
    assert np.allclose(y_ajuste, np.abs(gain(x_ajuste, *pfit)))
    _, phase_tracee = ax2.get_lines()[0].get_data()
    assert np.allclose(phase_tracee, np.degrees(np.angle(H)) % 360)
    plt.close(fig)
    fig = tracer_bode(f, np.abs(H), np.angle(H), gain, pfit, np.diag(np.square(err)), noms=("H0", "f0", "Q"))
    assert any("f0 = " + formater(2000.0, 2.0) in t.get_text() for t in fig.axes[0].get_legend().get_texts())
    plt.close(fig)
    fig = tracer_bode(f, np.abs(H), np.angle(H), gain_log=False)
    assert fig.axes[0].get_yscale() == "linear" and [len(ax.get_lines()) for ax in fig.axes] == [1, 1]
    plt.close(fig)
