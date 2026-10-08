"""
Lecture de fichiers de mesures : exports Latis Pro et Regressi, et un lecteur
CSV générique qui reconnaît seul le délimiteur et la virgule décimale
(readcsv, venu de dataanalysis, 2018).

Les trois acceptent un nom de fichier ou un pathlib.Path, et lisent l'UTF-8
comme les fichiers Windows (cp1252), sauf readcsv qui prend l'encodage qu'on
lui donne.

@author: a. marchand
"""

import csv
from pathlib import Path

import numpy as np

__all__ = ["fpointformat", "import_latispro", "import_regressi", "readcsv"]

REGRESSI_HEADER = 3
LATIS_HEADER = 1
_ENCODAGES = ("utf-8-sig", "cp1252", "latin-1")  # le dernier lit tout octet


def _fichier(filename):
    chemin = Path(filename)
    if not chemin.exists():
        raise ValueError(
            f"Le fichier {chemin} n'existe pas : vérifiez le dossier d'exécution (actuellement {Path.cwd()})"
        )
    return chemin


def _lignes(chemin, encodages):
    """Les lignes du fichier, dans le premier encodage qui le lit."""
    for encodage in encodages[:-1]:
        try:
            return chemin.read_text(encoding=encodage).splitlines()
        except UnicodeDecodeError:
            pass
    return chemin.read_text(encoding=encodages[-1]).splitlines()


def _nombre(cellule):
    """Un nombre, à point ou à virgule décimale ; NaN pour une cellule vide ou illisible."""
    try:
        return float(cellule.strip().replace(",", "."))
    except ValueError:
        return np.nan


def _colonnes(lignes, delimiter, colonnes):
    """Les `colonnes` premières colonnes de nombres ; une ligne vide est
    sautée, une cellule absente ou illisible vaut NaN."""
    cols = [[] for _ in range(colonnes)]
    for row in csv.reader(lignes, delimiter=delimiter):
        if not any(cellule.strip() for cellule in row):
            continue
        for j, col in enumerate(cols):
            col.append(_nombre(row[j]) if j < len(row) else np.nan)
    return [np.array(col) for col in cols]


def import_latispro(filename, colonnes=2, delimiter=";"):
    """Les `colonnes` premières colonnes d'un export Latis Pro : une liste de
    tableaux, par exemple [temps, EA0] ou [temps, EA0, temps, EA1].

    Dans Latis Pro : Fichier > Exporter > CSV, et glisser les courbes de
    « Courbes disponibles » dans « Courbes à exporter ». Une ligne d'en-tête,
    des points-virgules, la virgule décimale ; une cellule vide donne NaN,
    et on le dit.
    """
    if colonnes <= 0:
        raise ValueError("Le nombre de colonnes doit être > 0")
    lignes = _lignes(_fichier(filename), _ENCODAGES)
    npcols = _colonnes(lignes[LATIS_HEADER:], delimiter, colonnes)
    for i, col in enumerate(npcols):
        if np.isnan(col).any():
            print(f"WARNING: la colonne {i} contient des valeurs non numériques")
    return npcols


def import_regressi(filename, colonnes=2, delimiter="\t"):
    """Un export Regressi : rend (t, [colonne 1, colonne 2…]), le temps en
    première colonne puis les `colonnes` suivantes.

    Dans Regressi : Fichier > Enregistrer sous > CSV. Trois lignes d'en-tête
    (titre, noms, unités), des tabulations ; le point comme la virgule
    décimale sont lus.
    """
    if colonnes < 0:
        raise ValueError("Le nombre de colonnes doit être >= 0")
    lignes = _lignes(_fichier(filename), _ENCODAGES)
    t, *npcols = _colonnes(lignes[REGRESSI_HEADER:], delimiter, colonnes + 1)
    if np.isnan(t).any():
        print("WARNING: la colonne de temps (0) contient des valeurs non numériques")
    for i, col in enumerate(npcols, start=1):
        if np.isnan(col).any():
            print(f"WARNING: la colonne {i} contient des valeurs non numériques")
    return t, npcols


def _delimiteur(lignes):
    """Le délimiteur des colonnes : le premier de « ; », tabulation, « , »
    présent sur la première ligne non vide — la virgule en dernier, parce
    qu'elle peut être décimale. Aucun : une seule colonne (on rend alors un
    caractère absent du fichier)."""
    premiere = next((ligne for ligne in lignes if ligne.strip()), "")
    for candidat in (";", "\t", ","):
        if candidat in premiere:
            return candidat
    return "\x1f"


def readcsv(filename, encoding="utf8", entete=1, dtypes=None):
    """Les colonnes d'un fichier CSV, une liste de tableaux numpy.

    filename : le nom du fichier (ou un pathlib.Path) ;
    encoding : 'utf8' par défaut, 'cp1252' pour un fichier Windows ancien,
        'mac_roman' pour un Mac ancien ;
    entete : le nombre de lignes à écarter en début de fichier ;
    dtypes : les types des colonnes, par exemple [float, int, str], autant
        que de colonnes ; des float par défaut.

    Le délimiteur (point-virgule, tabulation ou virgule) est reconnu sur la
    première ligne, et une virgule décimale est lue comme un point : un
    fichier sans en-tête à une seule colonne et virgule décimale est donc
    ambigu, donnez-lui une ligne de titre. Une ligne vide est sautée ; une ligne qui n'a pas le bon
    nombre de colonnes, ou une valeur illisible, lève ValueError en disant
    où.
    """
    chemin = _fichier(filename)
    print(f"Lecture de {chemin}", flush=True)
    lignes = chemin.read_text(encoding=encoding).splitlines()
    rows = list(csv.reader(lignes, delimiter=_delimiteur(lignes)))
    for row in rows[:entete]:
        print("Entete exclus : ", row)
    donnees = [(numero, row) for numero, row in enumerate(rows, start=1) if numero > entete and any(row)]
    if not donnees:
        raise ValueError(f"file {chemin} : aucune ligne de données après {entete} ligne(s) d'en-tête")
    cols = len(donnees[0][1])
    print(f"{cols:d} colonnes", flush=True)
    if dtypes is None:
        dtypes = [float] * cols
    elif len(dtypes) != cols or any(dt not in (float, int, str) for dt in dtypes):
        raise ValueError(f"dtypes : {cols} types parmi float, int et str, un par colonne")
    print("Formats de conversion : ", dtypes, flush=True)
    T = [[] for _ in range(cols)]
    for numero, row in donnees:
        if len(row) != cols:
            raise ValueError(f"file {chemin}, line {numero} : {len(row)} colonnes au lieu de {cols}")
        for j, element in enumerate(row):
            try:
                T[j].append(dtypes[j](fpointformat(element.strip(), dtypes[j])))
            except ValueError as erreur:
                raise ValueError(f"file {chemin}, line {numero}, column {j + 1} : {erreur}") from None
    return [np.array(colonne) for colonne in T]


OTHERPOINT = ","
PYTHONPOINT = "."


def fpointformat(s, dtype):
    """La chaîne s prête à convertir en dtype : une virgule décimale devient
    un point ; un nombre qui a les deux est refusé (ValueError)."""
    if dtype is str or OTHERPOINT not in s:
        return s
    if PYTHONPOINT in s:
        raise ValueError(f"{s!r} : un point et une virgule")
    return s.replace(OTHERPOINT, PYTHONPOINT)
