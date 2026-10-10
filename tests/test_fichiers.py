import warnings
from pathlib import Path

import numpy as np
import pytest

from tpllg.fichiers import (
    ecrire_csv,
    fpointformat,
    import_latispro,
    import_regressi,
    lire_csv,
    lire_latispro,
    lire_regressi,
    readcsv,
)


def ecrire(chemin, texte, encoding="utf8"):
    chemin.write_text(texte, encoding=encoding)
    return chemin


class TestLireCsv:
    def test_devine_delimiteur_et_virgule(self, tmp_path):
        freq, ve = lire_csv(str(ecrire(tmp_path / "m.csv", "f;Ve\n500;1,00\n2000;1,50\n")))
        assert np.allclose(freq, [500, 2000]) and np.allclose(ve, [1.0, 1.5])

    def test_accepte_un_path(self, tmp_path):
        """'{:s}'.format(Path) levait TypeError."""
        freq, _ = lire_csv(ecrire(tmp_path / "m.csv", "f;Ve\n500;1,00\n"))
        assert np.allclose(freq, [500])

    def test_sans_en_tete_point_virgule_et_virgule_decimale(self, tmp_path):
        """Le Sniffer prenait la virgule pour délimiteur."""
        x, y = lire_csv(ecrire(tmp_path / "n.csv", "0,1;2,5\n0,2;3,5\n0,3;4,5\n"), entete=0)
        assert np.allclose(x, [0.1, 0.2, 0.3]) and np.allclose(y, [2.5, 3.5, 4.5])

    def test_une_seule_colonne(self, tmp_path):
        """csv.Error : le Sniffer ne trouvait pas de délimiteur."""
        (x,) = lire_csv(ecrire(tmp_path / "c.csv", "x\n1,5\n2,5\n"))
        assert np.allclose(x, [1.5, 2.5])

    @pytest.mark.parametrize("delimiteur", [",", "\t"])
    def test_virgule_ou_tabulation(self, tmp_path, delimiteur):
        x, y = lire_csv(ecrire(tmp_path / "v.csv", f"x{delimiteur}y\n1.5{delimiteur}2\n2.5{delimiteur}3\n"))
        assert np.allclose(x, [1.5, 2.5]) and np.allclose(y, [2, 3])

    def test_lignes_vides_sautees(self, tmp_path):
        x, _ = lire_csv(ecrire(tmp_path / "v.csv", "x;y\n1;2\n\n3;4\n\n"))
        assert np.allclose(x, [1, 3])

    def test_avec_types_et_erreur_situee(self, tmp_path):
        x, n, nom = lire_csv(
            ecrire(tmp_path / "t.csv", "x;n;nom\n1,5;2;a\n2,5;3;b\n"), dtypes=[float, int, str]
        )
        assert (
            np.allclose(x, [1.5, 2.5])
            and n.dtype.kind == "i"
            and list(n) == [2, 3]
            and list(nom) == ["a", "b"]
        )
        with pytest.raises(ValueError, match="line 3, column 2"):
            lire_csv(ecrire(tmp_path / "t.csv", "x;n\n1,5;2\n2,5;trois\n"))

    def test_ligne_incomplete_situee(self, tmp_path):
        with pytest.raises(ValueError, match="line 3 : 1 colonnes au lieu de 2"):
            lire_csv(ecrire(tmp_path / "t.csv", "x;n\n1,5;2\n2,5\n"), entete=1)

    def test_types_en_nombre_faux(self, tmp_path):
        with pytest.raises(ValueError, match="dtypes"):
            lire_csv(ecrire(tmp_path / "t.csv", "x;n\n1;2\n"), dtypes=[float])

    def test_encodage_windows_devine_ou_donne(self, tmp_path):
        f = ecrire(tmp_path / "w.csv", "température (°C);x\n1,5;2\n", encoding="cp1252")
        assert np.allclose(lire_csv(f)[0], [1.5]) and np.allclose(lire_csv(f, encoding="cp1252")[0], [1.5])

    def test_se_tait_par_defaut_et_tronque_l_echo_a_80_caracteres(self, tmp_path, capsys):
        """readcsv imprimait l'en-tête entier : cinquante mille caractères pour un fichier de sauvegarder."""
        f = ecrire(tmp_path / "e.csv", "x;" + "y" * 200 + "\n1;2\n")
        lire_csv(f)
        assert capsys.readouterr().out == ""
        lire_csv(f, verbose=True)
        lignes = capsys.readouterr().out.splitlines()
        assert lignes[0] == f"Lecture de {f}" and lignes[2] == "2 colonne(s), types float, float"
        assert lignes[1].startswith("En-tête écarté : x;yyy") and lignes[1].endswith("…")
        assert len(lignes[1]) == len("En-tête écarté : ") + 80


class TestEcrireCsv:
    def test_lire_csv_relit_ce_qu_ecrire_csv_ecrit(self, tmp_path):
        t, u = np.array([0, 1e-5, 2e-5]), np.array([0.1, -0.2, 1 / 3])
        f = tmp_path / "s.csv"
        ecrire_csv(f, [t, u], noms=["t (s)", "u (V)"])
        assert f.read_text(encoding="utf8").splitlines()[0] == "t (s);u (V)"
        t2, u2 = lire_csv(f)
        assert np.array_equal(t, t2) and np.array_equal(u, u2)

    def test_virgule_decimale_tabulation_sans_titres_et_textes(self, tmp_path):
        f = tmp_path / "v.csv"
        ecrire_csv(f, [[1.5, 2.5], [2, 3], ["a", "b"]], delimiter="\t", decimale=",")
        assert f.read_text(encoding="utf8") == "1,5\t2\ta\n2,5\t3\tb\n"
        x, n, nom = lire_csv(f, entete=0, dtypes=[float, int, str])
        assert np.allclose(x, [1.5, 2.5]) and list(n) == [2, 3] and list(nom) == ["a", "b"]

    def test_colonnes_mal_formees_refusees(self, tmp_path):
        with pytest.raises(ValueError, match="même longueur"):
            ecrire_csv(tmp_path / "m.csv", [[1, 2], [1, 2, 3]])
        with pytest.raises(ValueError, match="2 noms pour 1 colonnes"):
            ecrire_csv(tmp_path / "m.csv", [[1, 2]], noms=["a", "b"])
        with pytest.raises(ValueError, match="délimiteur"):
            ecrire_csv(tmp_path / "m.csv", [[1, 2]], delimiter=",", decimale=",")


def test_fpointformat():
    assert fpointformat("1,5", float) == "1.5" and fpointformat("1.5", float) == "1.5"
    assert fpointformat("15", int) == "15" and fpointformat("1,5", str) == "1,5"
    with pytest.raises(ValueError):
        fpointformat("1,234.5", float)


class TestRegressi:
    def test_rend_une_liste_le_temps_en_tete(self, tmp_path):
        f = ecrire(tmp_path / "r.txt", "Regressi\nt\tVs\tVe\ns\tV\tV\n0\t0.5\t1\n0.001\t0.6\t1\n")
        t, vs, ve = lire_regressi(str(f))
        assert np.allclose(t, [0, 0.001]) and np.allclose(vs, [0.5, 0.6]) and np.allclose(ve, [1, 1])
        assert len(lire_regressi(f, colonnes=2)) == 2

    def test_virgule_decimale_et_fichier_windows(self, tmp_path):
        f = ecrire(
            tmp_path / "r.txt", "Regressi\nt\tTempérature\ns\t°C\n0\t0,5\n0,1\t0,6\n", encoding="cp1252"
        )
        t, theta = lire_regressi(f)
        assert np.allclose(t, [0, 0.1]) and np.allclose(theta, [0.5, 0.6])


class TestLatisPro:
    def test_nan_sur_cellule_vide_avec_avertissement(self, tmp_path):
        f = ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1,2\n0,001;\n", encoding="iso8859")
        with pytest.warns(UserWarning, match="colonne 1 contient des cellules vides"):
            t, ea0 = lire_latispro(str(f))
        assert np.allclose(t, [0, 0.001]) and np.isnan(ea0[1]) and ea0[0] == 1.2

    def test_ligne_courte_et_ligne_vide(self, tmp_path):
        """IndexError non rattrapée sur une ligne à une colonne ou vide."""
        f = ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1,2\n0,001\n\n0,002;1,3\n", encoding="iso8859")
        with pytest.warns(UserWarning, match="colonne 1"):
            t, ea0 = lire_latispro(f)
        assert np.allclose(t, [0, 0.001, 0.002]) and np.isnan(ea0[1]) and ea0[2] == 1.3

    def test_toutes_les_colonnes_par_defaut(self, tmp_path):
        """import_latispro en rendait 2 sur 4 sans rien dire."""
        f = ecrire(tmp_path / "l.csv", "Temps;EA0;Temps;EA1\n0;1,2;0;0,5\n0,001;1,3;0,001;0,4\n")
        colonnes = lire_latispro(f)
        assert len(colonnes) == 4 and np.allclose(colonnes[3], [0.5, 0.4])
        assert len(lire_latispro(f, colonnes=2)) == 2

    def test_echo_demande(self, tmp_path, capsys):
        f = ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1,2\n")
        lire_latispro(f, verbose=True)
        assert capsys.readouterr().out == f"Lecture de {f}\nEn-tête écarté : Temps;EA0\n2 colonne(s)\n"


class TestErreurs:
    @pytest.mark.parametrize("lecteur", [lire_latispro, lire_regressi, lire_csv])
    def test_fichier_absent(self, tmp_path, lecteur):
        with pytest.raises(ValueError, match="n'existe pas"):
            lecteur(str(tmp_path / "rien.csv"))

    def test_zero_colonne_refuse_sur_un_fichier_qui_existe(self, tmp_path):
        """Le test d'avant passait sur un fichier absent : toujours vert."""
        f = ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1\n")
        assert Path(f).exists()
        with pytest.raises(ValueError, match="colonnes doit être"):
            lire_latispro(f, colonnes=0)
        with pytest.raises(ValueError, match="colonnes doit être"):
            lire_regressi(f, colonnes=-1)


class TestAnciensNoms:
    """Les noms de 2026.9 fonctionnent et rendent ce qu'ils rendaient, sans avertissement."""

    def test_readcsv(self, tmp_path):
        f = ecrire(tmp_path / "m.csv", "f;Ve\n500;1,00\n2000;1,50\n")
        freq, ve = readcsv(str(f), "utf8", 1)
        assert np.allclose(freq, [500, 2000]) and np.allclose(ve, [1.0, 1.5])

    def test_import_latispro_deux_colonnes_par_defaut(self, tmp_path):
        f = ecrire(tmp_path / "l.csv", "Temps;EA0;Temps;EA1\n0;1,2;0;0,5\n")
        colonnes = import_latispro(f)
        assert len(colonnes) == 2 and colonnes[1][0] == 1.2
        assert len(import_latispro(f, colonnes=4)) == 4

    def test_import_regressi_rend_le_temps_puis_la_liste(self, tmp_path):
        f = ecrire(tmp_path / "r.txt", "Regressi\nt\tVs\tVe\ns\tV\tV\n0\t0.5\t1\n0.001\t0.6\t1\n")
        t, cols = import_regressi(str(f), colonnes=2)
        assert isinstance(cols, list) and len(cols) == 2
        assert np.allclose(t, [0, 0.001]) and np.allclose(cols[0], [0.5, 0.6])
        assert np.allclose(cols[1], [1, 1])
        t, (theta,) = import_regressi(f, colonnes=1)
        assert np.allclose(theta, [0.5, 0.6])
        with pytest.raises(ValueError, match="colonnes doit être"):
            import_regressi(f, colonnes=-1)

    def test_aucun_avertissement(self, tmp_path):
        """2026.10.1 levait un DeprecationWarning ; les élèves ne doivent rien voir."""
        f = ecrire(tmp_path / "m.csv", "f;Ve\n500;1,00\n")
        with warnings.catch_warnings(record=True) as emis:
            warnings.simplefilter("always")
            readcsv(f)
            import_latispro(ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1,2\n"))
        assert emis == []
