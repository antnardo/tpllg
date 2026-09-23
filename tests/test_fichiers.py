import numpy as np

from tpllg.fichiers import import_latispro, import_regressi, readcsv


def test_readcsv_devine_delimiteur_et_virgule(tmp_path):
    f = tmp_path/"m.csv"
    f.write_text("f;Ve\n500;1,00\n2000;1,50\n", encoding="utf8")
    freq, ve = readcsv(str(f))
    assert np.allclose(freq, [500, 2000]) and np.allclose(ve, [1.0, 1.5])


def test_import_regressi_rend_une_liste(tmp_path):
    f = tmp_path/"r.txt"
    f.write_text("Regressi\nt\tVs\tVe\ns\tV\tV\n0\t0.5\t1\n0.001\t0.6\t1\n", encoding="utf8")
    t, cols = import_regressi(str(f), colonnes=2)
    assert isinstance(cols, list) and len(cols) == 2
    assert np.allclose(cols[0], [0.5, 0.6])


def test_import_latispro_nan_sur_cellule_vide(tmp_path):
    f = tmp_path/"l.csv"
    f.write_text("Temps;EA0\n0;1,2\n0,001;\n", encoding="iso8859")
    t, ea0 = import_latispro(str(f), colonnes=2)
    assert np.isnan(ea0[1]) and ea0[0] == 1.2
