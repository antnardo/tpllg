from pathlib import Path

import numpy as np
import pytest

from tpllg.fichiers import fpointformat, import_latispro, import_regressi, readcsv


def ecrire(chemin, texte, encoding="utf8"):
    chemin.write_text(texte, encoding=encoding)
    return chemin


class TestReadcsv:
    def test_devine_delimiteur_et_virgule(self, tmp_path):
        freq, ve = readcsv(str(ecrire(tmp_path / "m.csv", "f;Ve\n500;1,00\n2000;1,50\n")))
        assert np.allclose(freq, [500, 2000]) and np.allclose(ve, [1.0, 1.5])

    def test_accepte_un_path(self, tmp_path):
        """'{:s}'.format(Path) levait TypeError."""
        freq, _ = readcsv(ecrire(tmp_path / "m.csv", "f;Ve\n500;1,00\n"))
        assert np.allclose(freq, [500])

    def test_sans_en_tete_point_virgule_et_virgule_decimale(self, tmp_path):
        """Le Sniffer prenait la virgule pour délimiteur."""
        x, y = readcsv(ecrire(tmp_path / "n.csv", "0,1;2,5\n0,2;3,5\n0,3;4,5\n"), entete=0)
        assert np.allclose(x, [0.1, 0.2, 0.3]) and np.allclose(y, [2.5, 3.5, 4.5])

    def test_une_seule_colonne(self, tmp_path):
        """csv.Error : le Sniffer ne trouvait pas de délimiteur."""
        (x,) = readcsv(ecrire(tmp_path / "c.csv", "x\n1,5\n2,5\n"))
        assert np.allclose(x, [1.5, 2.5])

    @pytest.mark.parametrize("delimiteur", [",", "\t"])
    def test_virgule_ou_tabulation(self, tmp_path, delimiteur):
        x, y = readcsv(ecrire(tmp_path / "v.csv", f"x{delimiteur}y\n1.5{delimiteur}2\n2.5{delimiteur}3\n"))
        assert np.allclose(x, [1.5, 2.5]) and np.allclose(y, [2, 3])

    def test_lignes_vides_sautees(self, tmp_path):
        x, _ = readcsv(ecrire(tmp_path / "v.csv", "x;y\n1;2\n\n3;4\n\n"))
        assert np.allclose(x, [1, 3])

    def test_avec_types_et_erreur_situee(self, tmp_path):
        x, n, nom = readcsv(
            ecrire(tmp_path / "t.csv", "x;n;nom\n1,5;2;a\n2,5;3;b\n"), dtypes=[float, int, str]
        )
        assert (
            np.allclose(x, [1.5, 2.5])
            and n.dtype.kind == "i"
            and list(n) == [2, 3]
            and list(nom) == ["a", "b"]
        )
        with pytest.raises(ValueError, match="line 3, column 2"):
            readcsv(ecrire(tmp_path / "t.csv", "x;n\n1,5;2\n2,5;trois\n"))

    def test_ligne_incomplete_situee(self, tmp_path):
        with pytest.raises(ValueError, match="line 3 : 1 colonnes au lieu de 2"):
            readcsv(ecrire(tmp_path / "t.csv", "x;n\n1,5;2\n2,5\n"), entete=1)

    def test_types_en_nombre_faux(self, tmp_path):
        with pytest.raises(ValueError, match="dtypes"):
            readcsv(ecrire(tmp_path / "t.csv", "x;n\n1;2\n"), dtypes=[float])


def test_fpointformat():
    assert fpointformat("1,5", float) == "1.5" and fpointformat("1.5", float) == "1.5"
    assert fpointformat("15", int) == "15" and fpointformat("1,5", str) == "1,5"
    with pytest.raises(ValueError):
        fpointformat("1,234.5", float)


class TestRegressi:
    def test_rend_une_liste(self, tmp_path):
        f = ecrire(tmp_path / "r.txt", "Regressi\nt\tVs\tVe\ns\tV\tV\n0\t0.5\t1\n0.001\t0.6\t1\n")
        t, cols = import_regressi(str(f), colonnes=2)
        assert isinstance(cols, list) and len(cols) == 2
        assert (
            np.allclose(t, [0, 0.001]) and np.allclose(cols[0], [0.5, 0.6]) and np.allclose(cols[1], [1, 1])
        )

    def test_virgule_decimale_et_fichier_windows(self, tmp_path):
        f = ecrire(
            tmp_path / "r.txt", "Regressi\nt\tTempérature\ns\t°C\n0\t0,5\n0,1\t0,6\n", encoding="cp1252"
        )
        t, (theta,) = import_regressi(f, colonnes=1)
        assert np.allclose(t, [0, 0.1]) and np.allclose(theta, [0.5, 0.6])


class TestLatisPro:
    def test_nan_sur_cellule_vide(self, tmp_path):
        f = ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1,2\n0,001;\n", encoding="iso8859")
        t, ea0 = import_latispro(str(f), colonnes=2)
        assert np.allclose(t, [0, 0.001]) and np.isnan(ea0[1]) and ea0[0] == 1.2

    def test_ligne_courte_et_ligne_vide(self, tmp_path, capsys):
        """IndexError non rattrapée sur une ligne à une colonne ou vide."""
        f = ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1,2\n0,001\n\n0,002;1,3\n", encoding="iso8859")
        t, ea0 = import_latispro(f)
        assert np.allclose(t, [0, 0.001, 0.002]) and np.isnan(ea0[1]) and ea0[2] == 1.3
        assert "colonne 1 contient des valeurs non numériques" in capsys.readouterr().out


class TestErreurs:
    @pytest.mark.parametrize("lecteur", [import_latispro, import_regressi, readcsv])
    def test_fichier_absent(self, tmp_path, lecteur):
        with pytest.raises(ValueError, match="n'existe pas"):
            lecteur(str(tmp_path / "rien.csv"))

    def test_zero_colonne_refuse_sur_un_fichier_qui_existe(self, tmp_path):
        """Le test d'avant passait sur un fichier absent : toujours vert."""
        f = ecrire(tmp_path / "l.csv", "Temps;EA0\n0;1\n")
        assert Path(f).exists()
        with pytest.raises(ValueError, match="colonnes doit être"):
            import_latispro(f, colonnes=0)
        with pytest.raises(ValueError, match="colonnes doit être"):
            import_regressi(f, colonnes=-1)
