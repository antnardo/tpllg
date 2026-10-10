"""
Lecture et écriture de fichiers de mesures : un CSV de tableur dont on
reconnaît seul le délimiteur et la virgule décimale (lire_csv, qu'ecrire_csv
produit), les exports Latis Pro (lire_latispro) et Regressi
(lire_regressi).

Les trois lecteurs rendent la même chose : une liste de tableaux numpy, un
par colonne, dans l'ordre du fichier, ce qui se dépaquette d'un coup,
`t, u = lire_csv("mesures.csv")`. Ils acceptent un nom de fichier ou un
pathlib.Path, lisent l'UTF-8 comme les fichiers Windows (cp1252), se taisent
par défaut (verbose=True fait imprimer ce qu'ils ont compris du fichier), et
lisent toutes les colonnes sauf à en demander un nombre.

Les noms de 2026.9 fonctionnent encore, avec un avertissement : readcsv et
import_latispro rendent la même liste, import_regressi rend (t, [colonnes])
comme avant, avec `colonnes` le nombre de colonnes après celle du temps.

@author: a. marchand
"""

import csv
import warnings
from pathlib import Path

import numpy as np

from tpllg._interne import deprecie

__all__ = ["ecrire_csv", "lire_csv", "lire_latispro", "lire_regressi"]

REGRESSI_HEADER = 3
LATIS_HEADER = 1
_ENCODAGES = ("utf-8-sig", "cp1252", "latin-1")  # le dernier lit tout octet
_LARGEUR_ECHO = 80


def _fichier(filename):
    chemin = Path(filename)
    if not chemin.exists():
        raise ValueError(
            f"Le fichier {chemin} n'existe pas : vérifiez le dossier d'exécution (actuellement {Path.cwd()})"
        )
    return chemin


def _lignes(chemin, encoding=None):
    """Les lignes du fichier : dans `encoding` s'il est donné, sinon dans le
    premier de _ENCODAGES qui le lit."""
    encodages = _ENCODAGES if encoding is None else (encoding,)
    for encodage in encodages[:-1]:
        try:
            return chemin.read_text(encoding=encodage).splitlines()
        except UnicodeDecodeError:
            pass
    return chemin.read_text(encoding=encodages[-1]).splitlines()


def _echo(verbose, chemin, entetes, nb_colonnes, types=None):
    """Ce que le lecteur a compris du fichier, si on le lui demande : chaque
    ligne d'en-tête tronquée à 80 caractères."""
    if not verbose:
        return
    print(f"Lecture de {chemin}", flush=True)
    for ligne in entetes:
        texte = ligne if len(ligne) <= _LARGEUR_ECHO else ligne[: _LARGEUR_ECHO - 1] + "…"
        print(f"En-tête écarté : {texte}", flush=True)
    types = "" if types is None else ", types " + ", ".join(t.__name__ for t in types)
    print(f"{nb_colonnes} colonne(s){types}", flush=True)


def _nombre(cellule):
    """Un nombre, à point ou à virgule décimale ; NaN pour une cellule vide ou illisible."""
    try:
        return float(cellule.strip().replace(",", "."))
    except ValueError:
        return np.nan


def _colonnes(chemin, lignes, delimiter, colonnes, verbose, entete):
    """Les `colonnes` premières colonnes de nombres des `lignes` (toutes si
    None) ; une ligne vide est sautée, une cellule absente ou illisible vaut
    NaN, et un avertissement le dit."""
    rows = [row for row in csv.reader(lignes[entete:], delimiter=delimiter) if any(c.strip() for c in row)]
    if colonnes is None:
        colonnes = max((len(row) for row in rows), default=0)
    elif colonnes <= 0:
        raise ValueError("Le nombre de colonnes doit être > 0")
    _echo(verbose, chemin, lignes[:entete], colonnes)
    cols = [np.array([_nombre(row[j]) if j < len(row) else np.nan for row in rows]) for j in range(colonnes)]
    for j, col in enumerate(cols):
        if np.isnan(col).any():
            warnings.warn(
                f"{chemin.name} : la colonne {j} contient des cellules vides ou illisibles, lues NaN",
                stacklevel=3,
            )
    return cols


def lire_latispro(filename, colonnes=None, delimiter=";", verbose=False):
    """Un export Latis Pro : une liste de tableaux, un par colonne, par
    exemple [temps, EA0] ou [temps, EA0, temps, EA1] — Latis Pro écrit une
    colonne de temps par courbe. Toutes les colonnes, ou les `colonnes`
    premières.

    Dans Latis Pro : Fichier > Exporter > CSV, et glisser les courbes de
    « Courbes disponibles » dans « Courbes à exporter ». Une ligne d'en-tête,
    des points-virgules, la virgule décimale ; une cellule vide donne NaN,
    avec un avertissement.
    """
    chemin = _fichier(filename)
    return _colonnes(chemin, _lignes(chemin), delimiter, colonnes, verbose, LATIS_HEADER)


def lire_regressi(filename, colonnes=None, delimiter="\t", verbose=False):
    """Un export Regressi : une liste de tableaux, un par colonne, le temps
    en premier, [t, colonne 1, colonne 2…]. Toutes les colonnes, ou les
    `colonnes` premières, celle du temps comprise.

    Dans Regressi : Fichier > Enregistrer sous > CSV. Trois lignes d'en-tête
    (titre, noms, unités), des tabulations ; le point comme la virgule
    décimale sont lus.
    """
    chemin = _fichier(filename)
    return _colonnes(chemin, _lignes(chemin), delimiter, colonnes, verbose, REGRESSI_HEADER)


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


def lire_csv(filename, entete=1, dtypes=None, encoding=None, verbose=False):
    """Les colonnes d'un fichier CSV, une liste de tableaux numpy.

    filename : le nom du fichier (ou un pathlib.Path) ;
    entete : le nombre de lignes à écarter en début de fichier ;
    dtypes : les types des colonnes, par exemple [float, int, str], autant
        que de colonnes ; des float par défaut ;
    encoding : l'encodage, si on le connaît ; sinon l'UTF-8 puis celui de
        Windows (cp1252) sont essayés ;
    verbose : True pour imprimer ce qui a été compris du fichier.

    Le délimiteur (point-virgule, tabulation ou virgule) est reconnu sur la
    première ligne, et une virgule décimale est lue comme un point : un
    fichier sans en-tête à une seule colonne et virgule décimale est donc
    ambigu, donnez-lui une ligne de titre. Une ligne vide est sautée ; une
    ligne qui n'a pas le bon nombre de colonnes, ou une valeur illisible,
    lève ValueError en disant où. ecrire_csv écrit ce format.
    """
    chemin = _fichier(filename)
    lignes = _lignes(chemin, encoding)
    rows = list(csv.reader(lignes, delimiter=_delimiteur(lignes)))
    donnees = [(numero, row) for numero, row in enumerate(rows, start=1) if numero > entete and any(row)]
    if not donnees:
        raise ValueError(f"file {chemin} : aucune ligne de données après {entete} ligne(s) d'en-tête")
    cols = len(donnees[0][1])
    if dtypes is None:
        dtypes = [float] * cols
    elif len(dtypes) != cols or any(dt not in (float, int, str) for dt in dtypes):
        raise ValueError(f"dtypes : {cols} types parmi float, int et str, un par colonne")
    _echo(verbose, chemin, lignes[:entete], cols, dtypes)
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


def ecrire_csv(filename, colonnes, noms=None, delimiter=";", decimale=".", encoding="utf8"):
    """L'inverse de lire_csv : écrit les `colonnes` (des tableaux de même
    longueur, nombres ou textes) dans un fichier CSV, une ligne de titres
    si `noms` est donné (un par colonne), puis une ligne par point.

    delimiter : « ; » par défaut, celui des tableurs français ; decimale :
    « . », ou « , » pour un tableur français. lire_csv relit le fichier tel
    quel (entete=0 s'il n'a pas de titres), un tableur l'ouvre.
    """
    colonnes = [np.asarray(c) for c in colonnes]
    if not colonnes or any(c.ndim != 1 or c.shape != colonnes[0].shape for c in colonnes):
        raise ValueError("ecrire_csv : des colonnes 1D de même longueur, une au moins")
    if noms is not None and len(noms) != len(colonnes):
        raise ValueError(f"{len(noms)} noms pour {len(colonnes)} colonnes")
    if decimale == delimiter:
        raise ValueError("le séparateur décimal ne peut pas être le délimiteur des colonnes")
    lignes = [] if noms is None else [delimiter.join(str(nom) for nom in noms)]
    for cellules in zip(*colonnes):
        lignes.append(delimiter.join(_cellule(c, decimale) for c in cellules))
    Path(filename).write_text("\n".join(lignes) + "\n", encoding=encoding)


def _cellule(valeur, decimale):
    """Un nombre écrit avec toute sa précision et le séparateur décimal voulu ; un texte tel quel."""
    if isinstance(valeur, (str, np.str_)):
        return str(valeur)
    if isinstance(valeur, (bool, np.bool_)) or np.asarray(valeur).dtype.kind in "iu":
        return str(int(valeur))
    return repr(float(valeur)).replace(".", decimale)


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


# --- les noms de 2026.9


def readcsv(filename, encoding="utf8", entete=1, dtypes=None, verbose=False):
    """L'ancien nom (2026.9) de lire_csv : mêmes colonnes, même liste."""
    deprecie("fichiers.readcsv", "lire_csv(filename, entete, dtypes, encoding)")
    return lire_csv(filename, entete=entete, dtypes=dtypes, encoding=encoding, verbose=verbose)


def import_latispro(filename, colonnes=2, delimiter=";", verbose=False):
    """L'ancien nom (2026.9) de lire_latispro : les `colonnes` premières
    colonnes, deux par défaut (lire_latispro les lit toutes)."""
    deprecie("fichiers.import_latispro", "lire_latispro(filename), qui lit toutes les colonnes")
    return lire_latispro(filename, colonnes=colonnes, delimiter=delimiter, verbose=verbose)


def import_regressi(filename, colonnes=2, delimiter="\t", verbose=False):
    """L'ancien nom (2026.9) : rend (t, [colonne 1, colonne 2…]) comme avant,
    `colonnes` étant le nombre de colonnes après celle du temps.
    lire_regressi rend une seule liste, le temps en tête."""
    deprecie("fichiers.import_regressi", "lire_regressi(filename)", ", qui rend [t, colonne 1, colonne 2…]")
    if colonnes < 0:
        raise ValueError("Le nombre de colonnes doit être >= 0")
    t, *reste = lire_regressi(filename, colonnes=colonnes + 1, delimiter=delimiter, verbose=verbose)
    return t, reste
