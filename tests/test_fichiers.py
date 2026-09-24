import numpy as np
import pytest

from tpllg.fichiers import fpointformat, import_latispro, import_regressi, readcsv


def test_readcsv_devine_delimiteur_et_virgule(tmp_path):
    f = tmp_path/"m.csv"
    f.write_text("f;Ve\n500;1,00\n2000;1,50\n", encoding="utf8")
    freq, ve = readcsv(str(f))
    assert np.allclose(freq, [500, 2000]) and np.allclose(ve, [1.0, 1.5])


def test_readcsv_avec_types_et_erreur_situee(tmp_path):
    f = tmp_path/"t.csv"
    f.write_text("x;n;nom\n1,5;2;a\n2,5;3;b\n", encoding="utf8")
    x, n, nom = readcsv(str(f), dtypes=[float, int, str])
    assert np.allclose(x, [1.5, 2.5]) and n.dtype.kind == "i" and list(n) == [2, 3] and list(nom) == ["a", "b"]
    f.write_text("x;n\n1,5;2\n2,5;trois\n", encoding="utf8")
    with pytest.raises(ValueError, match="line 3, column 2"):
        readcsv(str(f))


def test_fpointformat():
    assert fpointformat("1,5", float) == "1.5" and fpointformat("1.5", float) == "1.5"
    assert fpointformat("15", int) == "15" and fpointformat("1,5", str) == "1,5"
    with pytest.raises(ValueError):
        fpointformat("1,234.5", float)


def test_import_regressi_rend_une_liste(tmp_path):
    f = tmp_path/"r.txt"
    f.write_text("Regressi\nt\tVs\tVe\ns\tV\tV\n0\t0.5\t1\n0.001\t0.6\t1\n", encoding="utf8")
    t, cols = import_regressi(str(f), colonnes=2)
    assert isinstance(cols, list) and len(cols) == 2
    assert np.allclose(t, [0, 0.001]) and np.allclose(cols[0], [0.5, 0.6]) and np.allclose(cols[1], [1, 1])


def test_import_latispro_nan_sur_cellule_vide(tmp_path):
    f = tmp_path/"l.csv"
    f.write_text("Temps;EA0\n0;1,2\n0,001;\n", encoding="iso8859")
    t, ea0 = import_latispro(str(f), colonnes=2)
    assert np.allclose(t, [0, 0.001]) and np.isnan(ea0[1]) and ea0[0] == 1.2


def test_imports_refusent_un_fichier_absent_ou_zero_colonne(tmp_path):
    absent = str(tmp_path/"rien.csv")
    with pytest.raises(ValueError, match="n'existe pas"):
        import_latispro(absent)
    with pytest.raises(ValueError, match="n'existe pas"):
        import_regressi(absent)
    with pytest.raises(ValueError):
        import_latispro(absent, colonnes=0)
