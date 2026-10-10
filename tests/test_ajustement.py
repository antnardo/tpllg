import math

import numpy as np
import pytest
import scipy

from tpllg.ajustement import (
    Ajustement,
    curve_fit_complex,
    curvefit,
    ecarts_types,
    formater,
    regression_york,
    residus_complexes,
    resume_parametres,
)


def gain(f, H0, f0, Q):
    return H0 / (1 + 1j * Q * (f / f0 - f0 / f))


def droite(x, a, b):
    return a * x + b


def mesures_bode(rng, bruit_norm=0.01, bruit_phase=0.01, H0=-5.0):
    f = np.geomspace(200, 20000, 30)
    H = gain(f, H0, 2000, 6)
    norm = np.abs(H) * (1 + rng.normal(0, bruit_norm, f.size))
    phase = np.angle(H) + rng.normal(0, bruit_phase, f.size)
    return f, norm, phase


class TestAjustement:
    def test_se_depaquette_comme_le_triplet(self):
        x = np.linspace(0, 1, 10)
        resultat = curvefit(droite, x, 2 * x + 1, p0=[1, 0], datayerrors=0.1, verbose=False)
        pfit, err, chi2 = resultat
        assert isinstance(resultat, Ajustement) and len(resultat) == 3
        assert resultat[0] is pfit and resultat[2] == chi2 == resultat.chi2
        assert np.allclose(np.sqrt(np.diag(resultat.pcov)), err)


class TestCurvefit:
    def test_affine_avec_incertitudes(self, rng):
        x = np.linspace(0, 1, 20)
        y = 2 * x - 1 + rng.normal(0, 0.05, x.size)
        pfit, err, chi2 = curvefit(droite, x, y, p0=[1, 0], datayerrors=0.05, verbose=False)
        assert abs(pfit[0] - 2) < 3 * err[0] and abs(pfit[1] + 1) < 3 * err[1]
        assert 0.3 < chi2 < 3
        # une incertitude constante vaut le tableau qui la répète
        pfit2, err2, chi2_2 = curvefit(
            droite, x, y, p0=[1, 0], datayerrors=0.05 * np.ones(x.size), verbose=False
        )
        assert np.array_equal(pfit, pfit2) and np.array_equal(err, err2) and chi2 == chi2_2

    def test_sans_incertitudes_retrouve_les_formules_des_moindres_carres(self, rng):
        x = np.linspace(0, 2, 25)
        y = 3 * x + 0.5 + rng.normal(0, 0.1, x.size)
        pfit, err, chi2 = curvefit(droite, x, y, p0=[1, 0], verbose=False)
        a, b = np.polyfit(x, y, 1)
        s2 = ((y - (a * x + b)) ** 2).sum() / (x.size - 2)  # la variance des résidus
        sxx = ((x - x.mean()) ** 2).sum()
        assert np.allclose(pfit, [a, b], rtol=1e-8) and chi2 == pytest.approx(s2)
        assert err[0] == pytest.approx(np.sqrt(s2 / sxx)) and err[1] == pytest.approx(
            np.sqrt(s2 * (1 / x.size + x.mean() ** 2 / sxx))
        )

    def test_variance_effective(self, rng):
        a0, b0, sx, sy = 2.0, -1.0, 0.03, 0.05
        x_vrai = np.linspace(0, 1, 30)
        x = x_vrai + rng.normal(0, sx, x_vrai.size)
        y = a0 * x_vrai + b0 + rng.normal(0, sy, x_vrai.size)
        pfit_y, err_y, _ = curvefit(droite, x, y, p0=[1, 0], datayerrors=sy, verbose=False)
        pfit, err, chi2 = curvefit(
            droite,
            x,
            y,
            p0=[1, 0],
            datayerrors=sy,
            dataxerrors=sx,
            function_derivate=lambda x, a, b: a,
            verbose=False,
        )
        # à incertitudes constantes, les paramètres sont ceux des moindres carrés ordinaires ;
        # seules les incertitudes changent, dans le rapport des sigmas effectifs
        assert np.allclose(pfit, pfit_y, rtol=1e-6)
        assert np.allclose(err, err_y * np.sqrt(sy**2 + pfit[0] ** 2 * sx**2) / sy, rtol=1e-4)
        assert abs(pfit[0] - a0) < 3 * err[0] and abs(pfit[1] - b0) < 3 * err[1]
        assert 0.4 < chi2 < 2.5

    def test_variance_effective_sans_derivee_prend_la_pente_du_modele(self, rng):
        x = np.linspace(0.3, 5, 12)
        y = 2 * np.exp(-x / 1.5) + rng.normal(0, 0.03, x.size)
        modele = lambda x, A, tau: A * np.exp(-x / tau)  # noqa: E731
        derivee = lambda x, A, tau: -A / tau * np.exp(-x / tau)  # noqa: E731
        avec = curvefit(modele, x, y, [1, 1], 0.03, 0.05, derivee, verbose=False)
        sans = curvefit(modele, x, y, [1, 1], 0.03, 0.05, verbose=False)
        assert np.allclose(sans.pfit, avec.pfit, rtol=1e-3) and np.allclose(sans.err, avec.err, rtol=1e-3)
        # pour une droite, la pente prise sur le modèle est exacte
        d1 = curvefit(droite, x, y, [1, 0], 0.03, 0.05, lambda x, a, b: a, verbose=False)
        d2 = curvefit(droite, x, y, [1, 0], 0.03, 0.05, verbose=False)
        assert np.allclose(d1.pfit, d2.pfit, rtol=1e-9) and np.allclose(d1.err, d2.err, rtol=1e-9)

    def test_incertitudes_sur_x_seules(self):
        x = np.linspace(1, 5, 8)
        pfit, err, _ = curvefit(droite, x, 2 * x + 1, [1, 0], dataxerrors=0.1, verbose=False)
        assert np.allclose(pfit, [2, 1]) and np.all(err > 0)

    @pytest.mark.parametrize("chi_limit", [0.0, -1.0])
    def test_variance_effective_au_moins_une_iteration(self, chi_limit):
        x = np.linspace(1, 5, 10)
        y = 2 * np.log(x) + 0.1
        pfit, _, _ = curvefit(
            lambda x, a, b: a * np.log(b * x), x, y, [1, 1], 0.1, 0.05, lambda x, a, b: a / x,
            chi_limit=chi_limit, verbose=False,
        )  # fmt: skip
        assert pfit[0] == pytest.approx(2, rel=1e-6)

    def test_variance_effective_garde_le_meilleur(self, rng):
        x = np.linspace(0.3, 5, 12)
        y = 2 * np.exp(-x / 1.5) + rng.normal(0, 0.03, x.size)
        modele = lambda x, A, tau: A * np.exp(-x / tau)  # noqa: E731
        une = curvefit(modele, x, y, [1, 1], 0.03, 0.15, n_var_method_max=1, verbose=False)
        dix = curvefit(modele, x, y, [1, 1], 0.03, 0.15, verbose=False)
        assert dix.chi2 <= une.chi2

    @pytest.mark.parametrize(("datayerrors", "dataxerrors"), [(0.0, None), (-0.1, None), (np.nan, None)])
    def test_incertitude_nulle_refusee(self, datayerrors, dataxerrors):
        x = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="incertitude nulle"):
            curvefit(droite, x, 2 * x, [1, 0], datayerrors, dataxerrors, verbose=False)

    @pytest.mark.parametrize("n", [0, 2.5, True])
    def test_n_var_method_max_entier(self, n):
        x = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="n_var_method_max"):
            curvefit(droite, x, 2 * x, [1, 0], 0.1, 0.1, n_var_method_max=n, verbose=False)

    def test_accepte_une_fonction_non_vectorisee(self):
        x = np.linspace(0, 5, 20)
        y = 2 * np.exp(-x / 1.5)
        pfit, _, _ = curvefit(lambda x, a, tau: a * math.exp(-x / tau), x, y, p0=[1, 1], verbose=False)
        assert np.allclose(pfit, [2, 1.5], atol=1e-6)
        pfit, _, _ = curvefit(
            lambda x, a, tau: a * math.exp(-x / tau), x, y, [1, 1], 0.01, 0.01,
            lambda x, a, tau: -a / tau * math.exp(-x / tau), verbose=False,
        )  # fmt: skip
        assert np.allclose(pfit, [2, 1.5], atol=1e-6)

    def test_se_tait_par_defaut_et_annonce_la_methode_en_francais(self, capsys):
        x = np.linspace(0, 1, 5)
        curvefit(droite, x, 2 * x, [1, 0])
        curvefit(droite, x, 2 * x, [1, 0], 0.1, 0.1)
        assert capsys.readouterr().out == ""
        curvefit(droite, x, 2 * x, [1, 0], verbose=True)
        curvefit(droite, x, 2 * x, [1, 0], 0.1, verbose=True)
        curvefit(droite, x, 2 * x, [1, 0], 0.1, 0.1, verbose=True)
        lignes = capsys.readouterr().out.splitlines()
        assert lignes[0] == "Moindres carrés" and lignes[1].startswith("Moindres carrés pondérés")
        assert lignes[2].startswith("Variance effective") and "Least" not in lignes[2]

    def test_u_x_et_u_y_sont_dataxerrors_et_datayerrors(self):
        x = np.linspace(0, 1, 8)
        y = 2 * x + 1 + 0.02 * np.sin(7 * x)
        longs = curvefit(droite, x, y, [1, 0], datayerrors=0.05, dataxerrors=0.03)
        courts = curvefit(droite, x, y, [1, 0], u_y=0.05, u_x=0.03)
        assert np.allclose(longs.pfit, courts.pfit) and np.allclose(longs.err, courts.err)
        assert curvefit(droite, x, y, [1, 0], u_y=0.05).chi2 == curvefit(droite, x, y, [1, 0], 0.05).chi2
        with pytest.raises(ValueError, match="datayerrors et u_y"):
            curvefit(droite, x, y, [1, 0], datayerrors=0.05, u_y=0.05)
        with pytest.raises(ValueError, match="dataxerrors et u_x"):
            curvefit(droite, x, y, [1, 0], 0.05, dataxerrors=0.03, u_x=0.03)


class TestCurveFitComplex:
    def test_retrouve_les_parametres(self, rng):
        f, norm, phase = mesures_bode(rng)
        pfit, err, chi2 = curve_fit_complex(gain, f, norm, phase, p0=[-4, 1800, 5], verbose=False)
        assert np.allclose(pfit, [-5, 2000, 6], rtol=0.01) and err.shape == (3,) and chi2 > 0
        res_norm, res_phase = residus_complexes(gain, f, norm, phase, pfit)
        assert res_norm.std() < 0.02 and res_phase.std() < 1

    def test_une_phase_a_2_pi_pres_ne_change_rien(self, rng):
        """Chaque phase prise à l'autre tour, en restant sous 2 pi en valeur absolue."""
        f, norm, phase = mesures_bode(rng)
        a = curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5], verbose=False)
        # uniform plutôt qu'integers : RandomState (numpy 1.16) n'a pas integers
        autre_tour = phase + 2 * np.pi * np.where(phase > 0, -1, 1) * (rng.uniform(0, 1, f.size) < 0.5)
        b = curve_fit_complex(gain, f, norm, autre_tour, [-4, 1800, 5], verbose=False)
        assert np.allclose(a.pfit, b.pfit, rtol=1e-6) and np.allclose(a.err, b.err, rtol=1e-6)

    def test_avec_les_incertitudes_du_module_et_de_la_phase(self, rng):
        f, norm, phase = mesures_bode(rng, bruit_norm=0.03, bruit_phase=np.radians(3))
        pfit, err, chi2 = curve_fit_complex(
            gain, f, norm, phase, [-4, 1800, 5], datayerrors=(0.03 * norm, np.radians(3)), verbose=False
        )
        assert 0.4 < chi2 < 2.5
        assert np.all(abs(pfit - [-5, 2000, 6]) < 3 * err)
        # des incertitudes doublées ne changent pas les paramètres, doublent err, divisent chi2 par 4
        pfit2, err2, chi2_2 = curve_fit_complex(
            gain, f, norm, phase, [-4, 1800, 5], datayerrors=(0.06 * norm, np.radians(6)), verbose=False
        )
        assert np.allclose(pfit2, pfit, rtol=1e-6) and np.allclose(err2, 2 * err, rtol=1e-4)
        assert chi2_2 == pytest.approx(chi2 / 4)
        with pytest.raises(ValueError, match="u_norm, u_phase"):
            curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5], datayerrors=0.03 * norm, verbose=False)

    @pytest.mark.skipif(
        tuple(int(x) for x in scipy.__version__.split(".")[:2]) < (1, 5),
        reason="curve_fit de scipy < 1.5 converge moins finement : un biais de 0,3 sigma apparaît",
    )
    @pytest.mark.parametrize(("u_norm", "u_phase_deg"), [(0.10, 1.0), (0.01, 10.0)])
    def test_incertitudes_justes_et_sans_biais(self, rng, u_norm, u_phase_deg):
        """Ajuster Re et Im ensemble faussait err de 15 % et biaisait H0 et Q de 0,7 sigma :
        la dispersion des paramètres sur 200 jeux de mesures doit être err, autour des vraies valeurs."""
        vrais = np.array([5, 2000, 6])
        p, e = [], []
        for _ in range(200):
            f, norm, phase = mesures_bode(rng, u_norm, np.radians(u_phase_deg), H0=5)
            r = curve_fit_complex(
                gain,
                f,
                norm,
                phase,
                vrais,
                datayerrors=(u_norm * norm, np.radians(u_phase_deg)),
                verbose=False,
            )
            p.append(r.pfit)
            e.append(r.err)
        p, e = np.array(p), np.array(e)
        assert np.allclose(p.std(axis=0) / e.mean(axis=0), 1, atol=0.12)
        assert np.all(abs(p.mean(axis=0) - vrais) < 0.25 * p.std(axis=0))

    def test_incertitudes_sur_f_sans_derivee(self, rng):
        f, norm, phase = mesures_bode(rng)

        def derivee(f, H0, f0, Q):
            return -H0 * 1j * Q * (1 / f0 + f0 / f**2) / (1 + 1j * Q * (f / f0 - f0 / f)) ** 2

        avec = curve_fit_complex(
            gain, f, norm, phase, [-4, 1800, 5], (0.01, 0.01), 1.0, derivee, verbose=False
        )
        sans = curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5], (0.01, 0.01), 1.0, verbose=False)
        assert np.allclose(avec.pfit, sans.pfit, rtol=1e-4) and np.allclose(avec.err, sans.err, rtol=1e-3)

    def test_signe_de_h0_faux_corrige(self, rng):
        """ln|H| interdit à H0 de passer par zéro : le départ par Re et Im le corrige."""
        f, norm, phase = mesures_bode(rng)
        bon = curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5], verbose=False)
        signe_faux = curve_fit_complex(gain, f, norm, phase, [4, 1800, 5], verbose=False)
        assert np.allclose(signe_faux.pfit, bon.pfit, rtol=1e-6)

    def test_module_nul_refuse(self):
        with pytest.raises(ValueError, match="strictement positifs"):
            curve_fit_complex(gain, [1, 2, 3], [1, 0, 1], [0, 0, 0], [1, 2, 3], verbose=False)


class TestYork:
    def test_donnees_de_pearson_et_york(self):
        """Le cas d'épreuve de York et al. (2004) : a = -0,4805, b = 5,4799."""
        x = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
        y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
        w_x = np.array([1000, 1000, 500, 800, 200, 80, 60, 20, 1.8, 1])
        w_y = np.array([1, 1.8, 4, 8, 20, 20, 70, 70, 100, 500])
        r = regression_york(x, 1 / np.sqrt(w_x), y, 1 / np.sqrt(w_y))
        assert np.allclose(r.pfit, [-0.4805334, 5.4799102], atol=1e-6)
        assert np.allclose(r.err, [0.057985, 0.294971], atol=1e-6)
        assert r.pcov[0, 1] == r.pcov[1, 0] < 0

    def test_sans_incertitude_sur_x_les_moindres_carres_ponderes(self, rng):
        x = np.linspace(0, 10, 12)
        u_y = 0.1 + 0.05 * x
        y = 2 * x + 1 + rng.normal(0, u_y)
        york = regression_york(x, 0, y, u_y)
        mc = curvefit(droite, x, y, [1, 0], datayerrors=u_y, verbose=False)
        assert np.allclose(york.pfit, mc.pfit, rtol=1e-9) and np.allclose(york.err, mc.err, rtol=1e-6)
        assert york.chi2 == pytest.approx(mc.chi2)

    def test_incertitudes_justes(self, rng):
        x_vrai = np.linspace(0, 10, 10)
        u_x, u_y = 0.3 + 0.03 * x_vrai, 0.5
        pentes, err = [], []
        for _ in range(400):
            r = regression_york(
                x_vrai + rng.normal(0, u_x), u_x, 2 * x_vrai + 1 + rng.normal(0, u_y, 10), u_y
            )
            pentes.append(r.pfit[0])
            err.append(r.err[0])
        assert abs(np.mean(pentes) - 2) < 0.15 * np.std(pentes)
        assert np.std(pentes) / np.mean(err) == pytest.approx(1, abs=0.1)

    def test_symetrique_en_x_et_y(self, rng):
        x = np.linspace(1, 10, 8)
        y = 0.5 * x + 2 + rng.normal(0, 0.2, 8)
        directe = regression_york(x, 0.3, y, 0.2)
        inverse = regression_york(y, 0.2, x, 0.3)
        assert directe.pfit[0] == pytest.approx(1 / inverse.pfit[0])

    @pytest.mark.parametrize(("u_x", "u_y"), [(0.0, 0.0), (-0.1, 0.1)])
    def test_incertitudes_impossibles_refusees(self, u_x, u_y):
        with pytest.raises(ValueError, match="u_x et u_y"):
            regression_york([1, 2, 3], u_x, [2, 4, 6], u_y)

    def test_trop_peu_de_points_refuse(self):
        with pytest.raises(ValueError, match="trois points"):
            regression_york([1, 2], 0.1, [2, 4], 0.1)


def test_residus_complexes_nuls_sur_le_modele():
    f = np.geomspace(100, 10000, 20)
    H = gain(f, -5, 2000, 6)
    res_norm, res_phase = residus_complexes(gain, f, np.abs(H), np.angle(H), [-5, 2000, 6])
    assert np.abs(res_norm).max() < 1e-12 and np.abs(res_phase).max() < 1e-9


class TestFormater:
    @pytest.mark.parametrize(
        ("valeur", "sigma", "unite", "texte"),
        [
            (1993.489, 1.875, "Hz", "1993.5 ± 1.9 Hz"),
            (-5.0823, 0.0631, "", "-5.082 ± 0.063"),
            (6.609, 0.156, "", "6.61 ± 0.16"),
            (1.0, 0.0996, "", "1.00 ± 0.10"),
            (1.0, 9.96, "", "1 ± 10"),
            (12345.6, 234.0, "", "12350 ± 230"),
            (99999.6, 1.0, "", "99999.6 ± 1.0"),
            (0.001234, 0.000047, "s", "0.001234 ± 0.000047 s"),
            (2.5, None, "V", "2.5 V"),
            (2.5, 0, "", "2.5"),
            (2.5, math.inf, "V", "2.5 ± inf V"),
            (2.5, math.nan, "", "2.5 ± nan"),
        ],
    )
    def test_deux_chiffres_sur_l_incertitude(self, valeur, sigma, unite, texte):
        assert formater(valeur, sigma, unite) == texte

    @pytest.mark.parametrize(
        ("valeur", "sigma", "unite", "texte"),
        [
            (6.626e-34, 1e-37, "J s", "(6.6260 ± 0.0010) × 10⁻³⁴ J s"),
            (1234567.0, 5432.0, "Hz", "(1.2346 ± 0.0054) × 10⁶ Hz"),
            (0.000123456, 0.0000047, "s", "(1.235 ± 0.047) × 10⁻⁴ s"),
            (-3.2e8, 4e6, "", "(-3.200 ± 0.040) × 10⁸"),
            (9.99999e5, 1000.0, "", "(1.0000 ± 0.0010) × 10⁶"),
            (0.0, 1e-5, "", "(0.0 ± 1.0) × 10⁻⁵"),
            (6.626e-34, None, "J s", "6.626 × 10⁻³⁴ J s"),
            (123456.0, 0, "", "1.235 × 10⁵"),
            (100000.0, 0.5, "", "(1.0000000 ± 0.0000050) × 10⁵"),
        ],
    )
    def test_notation_scientifique_hors_de_1e_3_a_1e5(self, valeur, sigma, unite, texte):
        """formater(6.626e-34, 1e-37) imprimait quarante zéros."""
        assert formater(valeur, sigma, unite) == texte

    def test_incertitude_negative_refusee(self):
        with pytest.raises(ValueError, match="négative"):
            formater(1.0, -0.1)


class TestResumeParametres:
    def test_depuis_err_ou_pcov(self):
        assert np.array_equal(ecarts_types([[4.0, 1.0], [1.0, 9.0]]), [2.0, 3.0])
        attendu = "a = 2.00 ± 0.10 V\nb = -1.00 ± 0.20"
        assert resume_parametres(("a", "b"), [2.0, -1.0], [0.1, 0.2], unites=("V", "")) == attendu
        assert resume_parametres(("a", "b"), [2.0, -1.0], np.diag([0.01, 0.04]), unites=("V", "")) == attendu

    def test_sans_incertitude(self):
        assert resume_parametres(("a", "b"), [2.5, 1.0]) == "a = 2.5\nb = 1"

    def test_longueurs_differentes_refusees(self):
        with pytest.raises(ValueError, match="autant de chaque"):
            resume_parametres(("H0", "f0", "Q"), [1, 2000, 6], [0.1, 2, 0.2], unites=("", "Hz"))


class TestPhasesEnRadians:
    """Une phase en degrés passait en silence et rendait H0 = 7,3 au lieu de -5,26."""

    def test_curve_fit_complex_refuse_des_degres(self, rng):
        f, norm, phase = mesures_bode(rng)
        with pytest.raises(ValueError, match="les phases doivent être en radians"):
            curve_fit_complex(gain, f, norm, np.degrees(phase), [-4, 1800, 5])
        with pytest.raises(ValueError, match="u_y"):
            curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5], datayerrors=(0.1, 0.1), u_y=(0.1, 0.1))

    def test_residus_complexes_refusent_des_degres(self):
        f = np.array([500, 1000, 2000.0])
        H = gain(f, -5, 2000, 6)
        with pytest.raises(ValueError, match="radians"):
            residus_complexes(gain, f, np.abs(H), np.degrees(np.angle(H)), [-5, 2000, 6])

    def test_u_x_et_u_y_dans_curve_fit_complex(self, rng):
        f, norm, phase = mesures_bode(rng)
        longs = curve_fit_complex(
            gain, f, norm, phase, [-4, 1800, 5], datayerrors=(0.02, 0.02), dataxerrors=1.0
        )
        courts = curve_fit_complex(gain, f, norm, phase, [-4, 1800, 5], u_y=(0.02, 0.02), u_x=1.0)
        assert np.allclose(longs.pfit, courts.pfit) and np.allclose(longs.err, courts.err)
